-- The weather, from Open-Meteo (free, no key). The only plugin that uses the network; with
-- `network = false` in [assistant] it says it cannot.
--
--   [plugins.weather]
--   place = "Bucharest"      # default: [assistant] place
local config = ...
local when = require("when")

-- WMO weather codes, as words.
local CODES = {
  [0] = "clear", [1] = "mostly clear", [2] = "partly cloudy", [3] = "overcast", [45] = "foggy", [48] = "foggy",
  [51] = "light drizzle", [53] = "drizzle", [55] = "heavy drizzle", [56] = "freezing drizzle", [57] = "freezing drizzle",
  [61] = "light rain", [63] = "rain", [65] = "heavy rain", [66] = "freezing rain", [67] = "freezing rain",
  [71] = "light snow", [73] = "snow", [75] = "heavy snow", [77] = "snow grains", [80] = "showers", [81] = "showers",
  [82] = "heavy showers", [85] = "snow showers", [86] = "heavy snow showers", [95] = "thunderstorms",
  [96] = "thunderstorms with hail", [99] = "thunderstorms with hail",
}

local cache = {}

local function url_encode(s)
  return (s:gsub("[^%w%-%._~]", function(c) return string.format("%%%02X", c:byte()) end))
end

local function locate(place)
  if cache[place] then return cache[place] end
  local name = place:match("^([^,]+)") or place
  local body, err = courier.http_get("https://geocoding-api.open-meteo.com/v1/search?count=1&format=json&name=" .. url_encode(name))
  if not body then return nil, err end
  local r = courier.json_decode(body)
  if not r.results or #r.results == 0 then return nil, "I don't know where " .. place .. " is" end
  cache[place] = r.results[1]
  return r.results[1]
end

local function round(x) return math.floor(x + 0.5) end

local function describe(place, f, day_index, is_today)
  local d = f.daily
  local i = day_index + 1
  local code = CODES[d.weather_code[i]] or "mixed weather"
  local lo, hi, rain = round(d.temperature_2m_min[i]), round(d.temperature_2m_max[i]), d.precipitation_probability_max[i]
  local s
  if is_today and f.current then
    local now = CODES[f.current.weather_code] or code
    s = string.format("In %s it is %d degrees and %s; today between %d and %d", place, round(f.current.temperature_2m), now, lo, hi)
  else
    s = string.format("In %s, %s: %s, between %d and %d degrees", place, "%DAY%", code, lo, hi)
  end
  if rain and rain >= 20 then s = s .. string.format(", with a %d%% chance of rain", round(rain)) end
  return s .. "."
end

return {
  tools = {
    {
      name = "get_weather",
      description = "The weather now and the forecast for up to a week, for a place (the user's own if not said).",
      parameters = {
        type = "object",
        properties = {
          place = { type = "string" },
          date = { type = "string", description = "The day as the user said it: \"today\", \"tomorrow\", a weekday (\"monday\") or YYYY-MM-DD" },
        },
      },
      run = function(a)
        local place = a.place or config.place or courier.place
        if not place or place == "" then return { error = "which place? ([assistant] place is not set)" } end
        local day, derr = when.parse_date(a.date)
        if not day then return { error = derr } end
        local offset = math.floor((day - when.midnight(courier.now())) / 86400 + 0.5)
        if offset < 0 or offset > 6 then return { error = "the forecast covers the next seven days" } end
        local loc, err = locate(place)
        if not loc then return { error = err } end
        local body, herr = courier.http_get(string.format(
          "https://api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f&current=temperature_2m,weather_code" ..
          "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=7",
          loc.latitude, loc.longitude))
        if not body then return { error = herr } end
        local f = courier.json_decode(body)
        local say = describe(loc.name, f, offset, offset == 0):gsub("%%DAY%%", when.say_day(day))
        return { place = loc.name, country = loc.country, daily = f.daily, current = f.current, say = say }
      end,
    },
  },
  tests = {
    describes = function(t)
      local f = {
        current = { temperature_2m = 17.6, weather_code = 2 },
        daily = { weather_code = { 61, 0 }, temperature_2m_min = { 9.2, 7 }, temperature_2m_max = { 19.4, 21 }, precipitation_probability_max = { 60, 0 } },
      }
      t.eq(describe("Bucharest", f, 0, true), "In Bucharest it is 18 degrees and partly cloudy; today between 9 and 19, with a 60% chance of rain.", "today")
      t.eq((describe("Bucharest", f, 1, false):gsub("%%DAY%%", "tomorrow")), "In Bucharest, tomorrow: clear, between 7 and 21 degrees.", "tomorrow")
      t.eq(url_encode("São Paulo"), "S%C3%A3o%20Paulo", "encoding")
    end,
    offline_is_an_error = function(t)
      local r = t.call("get_weather", { place = "Nowhere" })
      t.ok(r.error ~= nil, "no network in tests")
    end,
  },
}
