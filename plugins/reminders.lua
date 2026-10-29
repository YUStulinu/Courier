-- Reminders at a time of day (and a date), or in some minutes. Kept by the host's alarms, on disk.
local when = require("when")

local function due_at(a)
  local now = courier.now()
  if a.after then
    local secs, err = when.parse_duration(a.after)
    if not secs then return nil, err end
    if secs <= 0 then return nil, "the reminder must be in the future" end
    return now + math.floor(secs + 0.5)
  end
  if not a.time then return nil, "when should I remind you? Ask the user for a time" end
  local h, m = when.parse_time(a.time)
  if not h then return nil, "I could not understand the time " .. a.time end
  local day, err = when.parse_date(a.date, now)
  if not day then return nil, err end
  local d = os.date("*t", day)
  local at = os.time({ year = d.year, month = d.month, day = d.day, hour = h, min = m, sec = 0 })
  if at <= now then
    if a.date then return nil, "that time has already passed" end
    at = when.add_days(at, 1) -- "at 7" said at 9 pm means tomorrow
  end
  return at
end

local function list()
  local l = courier.alarms("reminder")
  table.sort(l, function(x, y) return x.at < y.at end)
  return l
end

local function what(alarm) return (alarm.text:gsub("^Reminder: ", ""):gsub("%.$", "")) end

return {
  tools = {
    {
      name = "add_reminder",
      description = "Remind the user of something, or wake them (an alarm), at a time of day or after some time.",
      parameters = {
        type = "object",
        properties = {
          text = { type = "string", description = "What to remind of, as an instruction (\"call Anna\")" },
          time = { type = "string", description = "HH:MM, 24-hour" },
          date = { type = "string", description = "Only if the user said a day: \"tomorrow\", a weekday (\"friday\") or YYYY-MM-DD" },
          after = { type = "string", description = "Instead of a time: how long from now, in the user's words (\"twenty minutes\")" },
        },
        required = { "text" },
      },
      run = function(a)
        local at, err = due_at(a)
        if not at then return { error = err } end
        local text = a.text:gsub("^to ", ""):gsub("%.$", "")
        local id = courier.alarm({ at = at, text = "Reminder: " .. text .. ".", kind = "reminder" })
        return {
          ok = true, id = id, at = when.iso_date(at) .. " " .. when.hhmm(at), in_seconds = at - courier.now(),
          say = "I'll remind you to " .. text .. " " .. when.say_day(at) .. " at " .. when.say_time(at) .. ".",
        }
      end,
    },
    {
      name = "list_reminders",
      description = "The reminders that are set.",
      parameters = { type = "object", properties = {} },
      run = function()
        local l = list()
        if #l == 0 then return { reminders = {}, say = "You have no reminders." } end
        local parts = {}
        for i, r in ipairs(l) do
          if i > 5 then break end
          parts[#parts + 1] = what(r) .. " " .. when.say_day(r.at) .. " at " .. when.say_time(r.at)
        end
        local head = #l == 1 and "One reminder: " or (#l .. " reminders: ")
        return { reminders = l, say = head .. table.concat(parts, "; ") .. "." }
      end,
    },
    {
      name = "delete_reminder",
      description = "Remove a reminder, found by what it is about.",
      parameters = { type = "object", properties = { text = { type = "string" } }, required = { "text" } },
      run = function(a)
        local q = a.text:lower()
        for _, r in ipairs(list()) do
          if r.text:lower():find(q, 1, true) then
            courier.cancel_alarm(r.id)
            return { ok = true, say = "Removed the reminder to " .. what(r) .. "." }
          end
        end
        return { ok = false, say = "I found no reminder about " .. a.text .. "." }
      end,
    },
  },
  tests = {
    -- The tests run at the real time; they only use times relative to it.
    at_a_time = function(t)
      local now = courier.now()
      local later = os.date("*t", now + 7200)
      local r = t.call("add_reminder", { text = "call Anna", time = string.format("%02d:%02d", later.hour, later.min) })
      t.ok(r.ok, "added: " .. tostring(r.error))
      t.ok(r.say:find("I'll remind you to call Anna ", 1, true) == 1, r.say)
      local due = courier.alarms("reminder")[1].at
      t.ok(due > now and due <= now + 7200 + 60 + 86400, "due within a day")
    end,
    in_minutes_and_tomorrow = function(t)
      local r = t.call("add_reminder", { text = "to stretch.", after = "half an hour" })
      t.ok(r.say:find("I'll remind you to stretch ", 1, true) == 1, r.say)
      r = t.call("add_reminder", { text = "water the plants", time = "8:00", date = "tomorrow" })
      t.ok(r.say:find("tomorrow at 8 AM", 1, true) ~= nil, r.say)
      t.ok(t.call("list_reminders", {}).say:find("reminders") ~= nil, "listed")
      t.eq(t.call("delete_reminder", { text = "plants" }).say, "Removed the reminder to water the plants.", "deleted")
    end,
    needs_a_time = function(t)
      t.ok(t.call("add_reminder", { text = "something" }).error:find("when") ~= nil, "asks when")
      t.ok(t.call("add_reminder", { text = "x", time = "25:00" }).error ~= nil, "bad time")
      t.ok(t.call("add_reminder", { text = "x", time = "10:00", date = "2001-01-01" }).error ~= nil, "past")
    end,
    reads_times = function(t)
      local h, m = when.parse_time("6:30 pm")
      t.eq({ h, m }, { 18, 30 }, "6:30 pm")
      h, m = when.parse_time("12am")
      t.eq({ h, m }, { 0, 0 }, "12am")
      t.eq(when.say_time(os.time({ year = 2026, month = 1, day = 1, hour = 18, min = 5 })), "6:05 PM", "say")
      t.eq(when.say_duration(3725), "1 hour, 2 minutes and 5 seconds", "duration")
      local fri = when.parse_date("friday", os.time({ year = 2026, month = 10, day = 29, hour = 12 })) -- a Thursday
      t.eq(os.date("%Y-%m-%d", fri), "2026-10-30", "next friday")
      t.eq(os.date("%Y-%m-%d", when.parse_date("next thursday", os.time({ year = 2026, month = 10, day = 29, hour = 12 }))), "2026-11-05", "a week on")
    end,
  },
}
