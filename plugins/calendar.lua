-- The calendar: iCalendar (.ics) files, such as an export from Google Calendar or Outlook. Events are
-- read from every file in `files` and from `file`; new events are written to `file`.
--
--   [plugins.calendar]
--   file = "C:/Users/me/calendar.ics"      # default: calendar.ics in the data folder
--   files = ["C:/Users/me/work.ics"]       # read only
--
-- Recurring events (RRULE) are expanded for daily, weekly (with BYDAY), monthly and yearly rules
-- with INTERVAL, COUNT and UNTIL. Times with a TZID are taken as local time.
local config = ...
local when = require("when")

local own = config.file or ((courier.data_dir or ".") .. "/calendar.ics")
local sources = { own }
for _, f in ipairs(config.files or {}) do sources[#sources + 1] = f end

local function read(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local s = f:read("a")
  f:close()
  return s
end

-- Lines folded at 75 characters continue with a space or tab.
local function unfold(s)
  return (s:gsub("\r\n", "\n"):gsub("\n[ \t]", ""))
end

local function utc_offset(t)
  return os.time(os.date("*t", t)) - os.time(os.date("!*t", t))
end

-- "20261103T093000Z", "20261103T093000", "20261103" -> epoch, all-day?
local function parse_dt(v)
  local y, mo, d = v:match("^(%d%d%d%d)(%d%d)(%d%d)$")
  if y then return os.time({ year = tonumber(y), month = tonumber(mo), day = tonumber(d), hour = 0 }), true end
  local h, mi, s, z
  y, mo, d, h, mi, s, z = v:match("^(%d%d%d%d)(%d%d)(%d%d)T(%d%d)(%d%d)(%d%d)(Z?)$")
  if not y then return nil end
  local t = os.time({ year = tonumber(y), month = tonumber(mo), day = tonumber(d), hour = tonumber(h), min = tonumber(mi), sec = tonumber(s) })
  if z == "Z" then t = t + utc_offset(t) end
  return t, false
end

local function parse(text)
  local events = {}
  for block in unfold(text):gmatch("BEGIN:VEVENT\n(.-)END:VEVENT") do
    local e = {}
    for line in block:gmatch("[^\n]+") do
      local key, params, value = line:match("^([%u%-]+)([^:]*):(.*)$")
      if key == "SUMMARY" then e.title = value:gsub("\\,", ","):gsub("\\;", ";"):gsub("\\n", " ")
      elseif key == "LOCATION" then e.location = value:gsub("\\,", ",")
      elseif key == "DTSTART" then e.start, e.all_day = parse_dt(value)
      elseif key == "DTEND" then e.finish = parse_dt(value)
      elseif key == "RRULE" then
        e.rule = {}
        for k, v in value:gmatch("([%u]+)=([^;]+)") do e.rule[k] = v end
      end
      local _ = params
    end
    if e.start then
      e.title = e.title or "(no title)"
      e.finish = e.finish or (e.start + (e.all_day and 86400 or 3600))
      events[#events + 1] = e
    end
  end
  return events
end

local BYDAY = { SU = 1, MO = 2, TU = 3, WE = 4, TH = 5, FR = 6, SA = 7 }

-- The starts of event `e` that fall in [from, to).
local function occurrences(e, from, to)
  local out = {}
  if not e.rule then
    if e.start < to and e.finish > from then out[1] = e.start end
    return out
  end
  local r = e.rule
  local interval = tonumber(r.INTERVAL or "1")
  local count = tonumber(r.COUNT or "100000")
  local until_t = r.UNTIL and parse_dt(r.UNTIL) or math.huge
  local days = nil
  if r.BYDAY then
    days = {}
    for d in r.BYDAY:gmatch("%a%a") do days[BYDAY[d]] = true end
  end
  local n, k = 0, 0
  local t = e.start
  local d0 = os.date("*t", e.start)
  while t < to and t <= until_t and n < count and k < 5000 do
    k = k + 1
    local candidates = { t }
    if r.FREQ == "WEEKLY" and days then
      candidates = {}
      local week = when.add_days(t, -(os.date("*t", t).wday - 1)) -- the Sunday of t's week
      for i = 0, 6 do
        local c = when.add_days(week, i)
        if days[os.date("*t", c).wday] and c >= e.start then candidates[#candidates + 1] = c end
      end
    end
    for _, c in ipairs(candidates) do
      if c > until_t or n >= count then break end
      n = n + 1
      if c < to and c + (e.finish - e.start) > from then out[#out + 1] = c end
    end
    if r.FREQ == "DAILY" then t = when.add_days(t, interval)
    elseif r.FREQ == "WEEKLY" then t = when.add_days(t, 7 * interval)
    elseif r.FREQ == "MONTHLY" then t = os.time({ year = d0.year, month = d0.month + interval * k, day = d0.day, hour = d0.hour, min = d0.min })
    elseif r.FREQ == "YEARLY" then t = os.time({ year = d0.year + interval * k, month = d0.month, day = d0.day, hour = d0.hour, min = d0.min })
    else break end
  end
  return out
end

local function events_between(from, to)
  local list = {}
  for _, path in ipairs(sources) do
    local text = read(path)
    if text then
      for _, e in ipairs(parse(text)) do
        for _, s in ipairs(occurrences(e, from, to)) do
          list[#list + 1] = { title = e.title, start = s, finish = s + (e.finish - e.start), all_day = e.all_day, location = e.location }
        end
      end
    end
  end
  table.sort(list, function(a, b) return a.start < b.start end)
  return list
end

local function escape(s) return (s:gsub("\\", "\\\\"):gsub(",", "\\,"):gsub(";", "\\;"):gsub("\n", "\\n")) end

return {
  tools = {
    {
      name = "calendar_events",
      description = "The events in the user's calendar on a day, or over several days.",
      parameters = {
        type = "object",
        properties = {
          date = { type = "string", description = "The day as the user said it: \"today\", \"tomorrow\", a weekday (\"monday\") or YYYY-MM-DD" },
          days = { type = "integer", minimum = 1, maximum = 31, description = "How many days from that date (7 for a week)" },
        },
      },
      run = function(a)
        local day, err = when.parse_date(a.date)
        if not day then return { error = err } end
        local n = a.days or 1
        local list = events_between(day, when.add_days(day, n))
        local out, parts = {}, {}
        for i, e in ipairs(list) do
          out[i] = { title = e.title, date = when.iso_date(e.start), start = not e.all_day and when.hhmm(e.start) or nil, all_day = e.all_day, location = e.location }
          if i <= 6 then
            local at = e.all_day and "all day" or ("at " .. when.say_time(e.start))
            parts[#parts + 1] = e.title .. (n > 1 and (" " .. when.say_day(e.start)) or "") .. " " .. at
          end
        end
        local span = n == 1 and when.say_day(day) or ("in the next " .. n .. " days")
        if #list == 0 then return { events = out, say = "Nothing in your calendar " .. span .. "." } end
        local head = (#list == 1 and "One event " or (#list .. " events ")) .. span .. ": "
        return { events = out, say = head .. table.concat(parts, "; ") .. (#list > 6 and "; and more." or ".") }
      end,
    },
    {
      name = "add_event",
      description = "Put an event in the user's calendar.",
      parameters = {
        type = "object",
        properties = {
          title = { type = "string" },
          date = { type = "string", description = "The day as the user said it: \"today\", \"tomorrow\", a weekday (\"monday\") or YYYY-MM-DD" },
          time = { type = "string", description = "HH:MM, 24-hour; leave out for an all-day event" },
          duration_minutes = { type = "integer", minimum = 5 },
        },
        required = { "title", "date" },
      },
      run = function(a)
        local day, err = when.parse_date(a.date)
        if not day then return { error = err } end
        local d = os.date("*t", day)
        local lines
        local start
        if a.time then
          local h, m = when.parse_time(a.time)
          if not h then return { error = "I could not understand the time " .. a.time } end
          start = os.time({ year = d.year, month = d.month, day = d.day, hour = h, min = m })
          local finish = start + (a.duration_minutes or 60) * 60
          lines = { "DTSTART:" .. os.date("%Y%m%dT%H%M%S", start), "DTEND:" .. os.date("%Y%m%dT%H%M%S", finish) }
        else
          start = day
          lines = { "DTSTART;VALUE=DATE:" .. os.date("%Y%m%d", day), "DTEND;VALUE=DATE:" .. os.date("%Y%m%d", when.add_days(day, 1)) }
        end
        local uid = string.format("%d-%d@courier", math.floor(courier.now()), math.random(1, 1e9))
        local event = table.concat({ "BEGIN:VEVENT", "UID:" .. uid, "DTSTAMP:" .. os.date("!%Y%m%dT%H%M%SZ", math.floor(courier.now())),
          "SUMMARY:" .. escape(a.title), lines[1], lines[2], "END:VEVENT" }, "\r\n")
        local text = read(own)
        if text and text:find("END:VCALENDAR") then
          text = text:gsub("END:VCALENDAR%s*$", event .. "\r\nEND:VCALENDAR\r\n")
        else
          text = "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//Courier//EN\r\n" .. event .. "\r\nEND:VCALENDAR\r\n"
        end
        local f, ferr = io.open(own, "w")
        if not f then return { error = "cannot write the calendar: " .. tostring(ferr) } end
        f:write(text)
        f:close()
        local at = a.time and (" at " .. when.say_time(start)) or ""
        return { ok = true, say = "Added " .. a.title .. " " .. when.say_day(start) .. at .. "." }
      end,
    },
  },
  tests = {
    parses_and_expands = function(t)
      local ics = table.concat({
        "BEGIN:VCALENDAR", "BEGIN:VEVENT", "SUMMARY:Dentist", "DTSTART:20261103T093000", "DTEND:20261103T100000", "END:VEVENT",
        "BEGIN:VEVENT", "SUMMARY:Yoga", "DTSTART:20261102T180000", "DTEND:20261102T190000",
        "RRULE:FREQ=WEEKLY;BYDAY=MO,WE;COUNT=4", "END:VEVENT",
        "BEGIN:VEVENT", "SUMMARY:Holiday\\, finally", "DTSTART;VALUE=DATE:20261105", "DTEND;VALUE=DATE:20261106", "END:VEVENT",
        "END:VCALENDAR" }, "\r\n")
      local evs = parse(ics)
      t.eq(#evs, 3, "three events")
      t.eq(evs[3].title, "Holiday, finally", "escaped comma")
      local from = os.time({ year = 2026, month = 11, day = 1, hour = 0 })
      local yoga = occurrences(evs[2], from, when.add_days(from, 14))
      local days = {}
      for i, s in ipairs(yoga) do days[i] = os.date("%m-%d", s) end
      t.eq(days, { "11-02", "11-04", "11-09", "11-11" }, "Mondays and Wednesdays, four times")
      t.eq(#occurrences(evs[1], from, when.add_days(from, 2)), 0, "not yet")
      t.eq(#occurrences(evs[1], from, when.add_days(from, 3)), 1, "the third")
    end,
    adds_and_reads_back = function(t)
      local r = t.call("add_event", { title = "Lunch with Anna", date = "tomorrow", time = "13:00" })
      t.ok(r.ok, "added: " .. tostring(r.error))
      t.ok(r.say:find("Added Lunch with Anna tomorrow at 1 PM", 1, true) == 1, r.say)
      r = t.call("calendar_events", { date = "tomorrow" })
      t.ok(r.say:find("Lunch with Anna at 1 PM", 1, true) ~= nil, r.say)
      t.eq(r.events[#r.events].start, "13:00", "start")
    end,
  },
}
