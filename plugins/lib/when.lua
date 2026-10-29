-- Dates and times for the plugins: reading what the model passes ("tomorrow", "2026-11-03",
-- "friday", "18:30", "6 pm") and saying times back ("at 6:30 PM", "tomorrow", "on Friday").
-- All in local time, through os.time / os.date.

local M = {}

local WEEKDAYS = { "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday" }
M.WEEKDAYS = WEEKDAYS

local function midnight(t)
  local d = os.date("*t", t)
  return os.time({ year = d.year, month = d.month, day = d.day, hour = 0, min = 0, sec = 0 })
end
M.midnight = midnight

-- Adds whole days, through the calendar (not 86400 s), so daylight saving changes do not shift it.
function M.add_days(t, n)
  local d = os.date("*t", t)
  return os.time({ year = d.year, month = d.month, day = d.day + n, hour = d.hour, min = d.min, sec = d.sec })
end

-- "today", "tomorrow", "yesterday", "YYYY-MM-DD", "MM-DD", a weekday ("friday", "next friday")
-- -> midnight of that day, or nil and why.
function M.parse_date(s, now)
  now = now or courier.now()
  if s == nil or s == "" then return midnight(now) end
  s = s:lower():gsub("^%s+", ""):gsub("%s+$", "")
  if s == "today" or s == "tonight" or s == "now" then return midnight(now) end
  if s == "tomorrow" then return midnight(M.add_days(now, 1)) end
  if s == "yesterday" then return midnight(M.add_days(now, -1)) end
  local y, m, d = s:match("^(%d%d%d%d)%-(%d%d?)%-(%d%d?)$")
  if y then return os.time({ year = tonumber(y), month = tonumber(m), day = tonumber(d), hour = 0 }) end
  m, d = s:match("^(%d%d?)%-(%d%d?)$")
  if m then
    local t = os.date("*t", now)
    local at = os.time({ year = t.year, month = tonumber(m), day = tonumber(d), hour = 0 })
    if at < midnight(now) then at = os.time({ year = t.year + 1, month = tonumber(m), day = tonumber(d), hour = 0 }) end
    return at
  end
  local next_week = s:match("^next%s+(%a+)$")
  local name = next_week or s:match("^on%s+(%a+)$") or s:match("^this%s+(%a+)$") or s
  for i, w in ipairs(WEEKDAYS) do
    if name == w or name == w:sub(1, 3) then
      local today = os.date("*t", now).wday
      local ahead = (i - today) % 7
      if next_week and ahead == 0 then ahead = 7 end
      return midnight(M.add_days(now, ahead))
    end
  end
  return nil, "I could not understand the date " .. s
end

-- "18:30", "6:30", "6 pm", "6:30pm", "noon", "midnight" -> hour, minute (24-hour), or nil.
function M.parse_time(s)
  if s == nil or s == "" then return nil end
  s = s:lower():gsub("%s+", "")
  if s == "noon" or s == "midday" then return 12, 0 end
  if s == "midnight" then return 0, 0 end
  local h, m, ap = s:match("^(%d%d?):(%d%d)([ap]%.?m?%.?)$")
  if not h then h, ap = s:match("^(%d%d?)([ap]%.?m?%.?)$"); m = "0" end
  if not h then h, m = s:match("^(%d%d?):(%d%d)$") end
  if not h then h = s:match("^(%d%d?)$"); m = "0" end
  if not h then return nil end
  h, m = tonumber(h), tonumber(m)
  if ap then
    if h < 1 or h > 12 then return nil end
    if ap:sub(1, 1) == "p" and h < 12 then h = h + 12 end
    if ap:sub(1, 1) == "a" and h == 12 then h = 0 end
  end
  if h > 23 or m > 59 then return nil end
  return h, m
end

-- 18:30 -> "6:30 PM"; 9:00 -> "9 AM".
function M.say_time(t)
  local d = os.date("*t", t)
  local h = d.hour % 12
  if h == 0 then h = 12 end
  local ap = d.hour < 12 and "AM" or "PM"
  if d.min == 0 then return string.format("%d %s", h, ap) end
  return string.format("%d:%02d %s", h, d.min, ap)
end

-- "today", "tomorrow", "on Friday", "on November 3".
function M.say_day(t, now)
  now = now or courier.now()
  local days = math.floor((midnight(t) - midnight(now)) / 86400 + 0.5)
  if days == 0 then return "today" end
  if days == 1 then return "tomorrow" end
  if days == -1 then return "yesterday" end
  if days > 1 and days < 7 then return "on " .. os.date("%A", t) end
  return "on " .. os.date("%B", t) .. " " .. tonumber(os.date("%d", t))
end

-- 125 seconds -> "2 minutes and 5 seconds".
function M.say_duration(secs)
  secs = math.floor(secs + 0.5)
  local h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
  local parts = {}
  local function add(n, unit)
    if n > 0 then parts[#parts + 1] = n .. " " .. unit .. (n == 1 and "" or "s") end
  end
  add(h, "hour"); add(m, "minute"); add(s, "second")
  if #parts == 0 then return "0 seconds" end
  if #parts == 1 then return parts[1] end
  return table.concat(parts, ", ", 1, #parts - 1) .. " and " .. parts[#parts]
end

local NUMBERS = {
  a = 1, an = 1, one = 1, two = 2, three = 3, four = 4, five = 5, six = 6, seven = 7, eight = 8, nine = 9,
  ten = 10, eleven = 11, twelve = 12, thirteen = 13, fourteen = 14, fifteen = 15, sixteen = 16,
  seventeen = 17, eighteen = 18, nineteen = 19, twenty = 20, thirty = 30, forty = 40, fifty = 50,
  sixty = 60, seventy = 70, eighty = 80, ninety = 90, couple = 2, few = 3, half = 0.5, quarter = 0.25,
}
local UNITS = {
  second = 1, seconds = 1, sec = 1, secs = 1, s = 1,
  minute = 60, minutes = 60, min = 60, mins = 60, m = 60,
  hour = 3600, hours = 3600, hr = 3600, hrs = 3600, h = 3600,
}

-- A length of time as people say it -> seconds: "ten minutes", "an hour and a half", "half an hour",
-- "1 hour 20 minutes", "forty-five seconds", "90 sec", "1.5 hours". Nil and why if there is no unit.
function M.parse_duration(s)
  if type(s) == "number" then return s * 60 end -- a bare number is minutes
  local words = {}
  for w in s:lower():gsub("(%d)([a-z])", "%1 %2"):gmatch("[%w%.]+") do words[#words + 1] = w end
  local total, pending, last_unit, found = 0, nil, nil, false
  local i = 1
  while i <= #words do
    local w = words[i]
    local n = tonumber(w) or NUMBERS[w]
    if w == "and" and words[i + 1] == "a" and words[i + 2] == "half" then
      -- "an hour and a half" adds half the unit; "two and a half minutes" half to the number.
      if pending then pending = pending + 0.5 elseif last_unit then total = total + 0.5 * last_unit end
      i = i + 3
    elseif (w == "a" or w == "an") and pending and pending < 1 then
      i = i + 1 -- "half an hour", "a quarter of an hour"
    elseif n then
      -- "forty five": tens and ones add up; "one and a half hours": the half joins the number.
      if pending and pending >= 20 and pending % 10 == 0 and n < 10 then pending = pending + n
      elseif w == "half" and pending then pending = pending + 0.5
      else pending = n end
      i = i + 1
    elseif UNITS[w] then
      total = total + (pending or 1) * UNITS[w]
      last_unit, pending, found = UNITS[w], nil, true
      i = i + 1
    else
      i = i + 1 -- "and", "for", "timer"...
    end
  end
  if not found then return nil, "no length in " .. s end
  return total
end

function M.iso_date(t) return os.date("%Y-%m-%d", t) end
function M.hhmm(t) return os.date("%H:%M", t) end

return M
