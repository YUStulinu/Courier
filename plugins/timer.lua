-- Countdown timers. They ring through the host's alarms, so they survive a plugin reload and a restart.
local when = require("when")

local function timers()
  local list = courier.alarms("timer")
  table.sort(list, function(a, b) return a.at < b.at end)
  return list
end

local function label_of(alarm)
  return alarm.id:match("^timer:%d+$") and alarm.text:match("^The (.-) timer is done") or nil
end

return {
  tools = {
    {
      name = "set_timer",
      description = "Start a countdown timer for a length of time. For a time of day (\"at seven\"), use add_reminder.",
      parameters = {
        type = "object",
        properties = {
          duration = { type = "string", description = "The length in the user's words: \"ten minutes\", \"an hour and a half\", \"90 seconds\"" },
          label = { type = "string", description = "What it is for, if the user said (\"pasta\")" },
        },
        required = { "duration" },
      },
      run = function(a)
        local secs, err = when.parse_duration(a.duration)
        if not secs then return { error = err .. "; for a time of day use add_reminder" } end
        secs = math.floor(secs + 0.5)
        if secs <= 0 then return { error = "the timer needs a length" } end
        if secs > 24 * 3600 then return { error = "timers last at most 24 hours; use a reminder" } end
        local length = when.say_duration(secs)
        local text = a.label and ("The " .. a.label .. " timer is done.") or ("Your timer for " .. length .. " is done.")
        local id = courier.alarm({ at = courier.now() + secs, text = text, kind = "timer" })
        local what = a.label and (a.label:sub(1, 1):upper() .. a.label:sub(2) .. " timer") or "Timer"
        return { ok = true, id = id, seconds = secs, ends = when.hhmm(courier.now() + secs), say = what .. " set for " .. length .. "." }
      end,
    },
    {
      name = "list_timers",
      description = "How long the running timers have left.",
      parameters = { type = "object", properties = {} },
      run = function()
        local list = timers()
        if #list == 0 then return { timers = {}, say = "No timers are running." } end
        local parts = {}
        for _, t in ipairs(list) do
          local left = when.say_duration(math.max(0, t.at - courier.now()))
          local label = label_of(t)
          parts[#parts + 1] = (label and ("the " .. label .. " timer has ") or "") .. left .. " left"
        end
        local say = (#list == 1 and "One timer: " or (#list .. " timers: ")) .. table.concat(parts, "; ") .. "."
        return { timers = list, say = say }
      end,
    },
    {
      name = "cancel_timer",
      description = "Stop a running timer: the one with this label, the next one to ring, or all of them (all = true).",
      parameters = { type = "object", properties = { label = { type = "string" }, all = { type = "boolean" } } },
      run = function(a)
        local list = timers()
        if #list == 0 then return { ok = false, say = "No timers are running." } end
        if a.all then
          for _, t in ipairs(list) do courier.cancel_alarm(t.id) end
          return { ok = true, say = #list == 1 and "Timer cancelled." or ("All " .. #list .. " timers cancelled.") }
        end
        local pick = list[1]
        if a.label then
          pick = nil
          for _, t in ipairs(list) do
            if (label_of(t) or ""):lower():find(a.label:lower(), 1, true) then pick = t break end
          end
          if not pick then return { ok = false, say = "There is no " .. a.label .. " timer." } end
        end
        courier.cancel_alarm(pick.id)
        return { ok = true, say = "Timer cancelled." }
      end,
    },
  },
  tests = {
    set_and_list = function(t)
      local r = t.call("set_timer", { duration = "five minutes" })
      t.eq(r.say, "Timer set for 5 minutes.", "say")
      r = t.call("set_timer", { duration = "a minute and a half", label = "tea" })
      t.eq(r.say, "Tea timer set for 1 minute and 30 seconds.", "labelled")
      r = t.call("list_timers", {})
      t.ok(r.say:find("2 timers") == 1, "two timers: " .. r.say)
      t.ok(r.say:find("the tea timer has 1 minute and 30 seconds left", 1, true) ~= nil, r.say)
      t.eq(t.call("cancel_timer", { label = "tea" }).say, "Timer cancelled.", "cancel tea")
      t.eq(#courier.alarms("timer"), 1, "one left")
      t.eq(t.call("cancel_timer", { all = true }).say, "Timer cancelled.", "cancel all")
    end,
    refuses_nonsense = function(t)
      t.ok(t.call("set_timer", {}).error ~= nil, "no length")
      t.ok(t.call("set_timer", { duration = "30 hours" }).error ~= nil, "too long")
      t.ok(t.call("set_timer", { duration = "seven tomorrow morning" }).error:find("add_reminder") ~= nil, "a time of day")
    end,
    durations = function(t)
      local cases = {
        ["ten minutes"] = 600, ["an hour and a half"] = 5400, ["half an hour"] = 1800, ["1 hour 20 minutes"] = 4800,
        ["forty-five seconds"] = 45, ["90 sec"] = 90, ["1.5 hours"] = 5400, ["two and a half minutes"] = 150,
        ["a quarter of an hour"] = 900, ["one and a half hours"] = 5400, ["5min"] = 300, ["a minute"] = 60,
        ["twenty five minutes"] = 1500,
      }
      for s, want in pairs(cases) do t.eq(when.parse_duration(s), want, s) end
      t.eq(when.parse_duration("seven"), nil, "no unit")
    end,
  },
}
