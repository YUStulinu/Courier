-- The computer: volume, opening programs, locking the screen.
--
--   [plugins.computer]
--   apps = { browser = "firefox", editor = "code" }   # extra names, or replacements
local config = ...

-- Spoken name -> what to open. On Windows `start` finds programs by name; the rest are URIs.
local APPS = courier.os == "windows" and {
  browser = "https://", ["web browser"] = "https://", notepad = "notepad", calculator = "calc", calc = "calc",
  explorer = "explorer", ["file explorer"] = "explorer", files = "explorer", settings = "ms-settings:",
  paint = "mspaint", ["task manager"] = "taskmgr", terminal = "wt", ["command prompt"] = "cmd",
  ["vs code"] = "code", ["visual studio code"] = "code", spotify = "spotify:", word = "winword", excel = "excel",
  powerpoint = "powerpnt", outlook = "outlook", mail = "outlook", camera = "microsoft.windows.camera:",
  clock = "ms-clock:", calendar = "outlookcal:",
} or {
  browser = "https://", ["web browser"] = "https://", calculator = "gnome-calculator", files = "nautilus",
  terminal = "x-terminal-emulator", ["vs code"] = "code", settings = "gnome-control-center",
}
for k, v in pairs(config.apps or {}) do APPS[k:lower()] = v end

local function title(s) return (s:gsub("^%l", string.upper)) end

return {
  tools = {
    {
      name = "volume",
      description = "Turn the computer's volume up or down, or mute and unmute it.",
      parameters = {
        type = "object",
        properties = {
          action = { type = "string", enum = { "up", "down", "mute", "unmute" } },
          steps = { type = "integer", minimum = 1, maximum = 25, description = "How much: 1 is a little, 5 is the default, 10 is a lot" },
        },
        required = { "action" },
      },
      run = function(a)
        if a.action == "mute" or a.action == "unmute" then
          courier.key("mute")
          return { ok = true, say = a.action == "mute" and "Muted." or "Sound is back on." }
        end
        local key = a.action == "up" and "volume_up" or "volume_down"
        for _ = 1, a.steps or 5 do courier.key(key) end
        return { ok = true, say = a.action == "up" and "Louder." or "Quieter." }
      end,
    },
    {
      name = "open_app",
      description = "Open a program (browser, notepad, calculator, file explorer, settings, terminal, ...).",
      parameters = { type = "object", properties = { name = { type = "string" } }, required = { "name" } },
      run = function(a)
        local name = a.name:lower():gsub("^the ", ""):gsub(" app$", "")
        local target = APPS[name] or name
        courier.open(target)
        return { ok = true, say = "Opening " .. title(name) .. "." }
      end,
    },
    {
      name = "lock_screen",
      description = "Lock the computer's screen.",
      parameters = { type = "object", properties = {} },
      run = function()
        courier.lock()
        return { ok = true, say = "Locking the screen." }
      end,
    },
  },
  tests = {
    volume_presses_keys = function(t)
      t.eq(t.call("volume", { action = "up", steps = 2 }).say, "Louder.", "up")
      t.eq(t.call("volume", { action = "mute" }).say, "Muted.", "mute")
      t.ok(t.call("volume", { action = "sideways" }).error ~= nil, "enum")
    end,
    apps_by_name = function(t)
      t.eq(t.call("open_app", { name = "the Calculator app" }).say, "Opening Calculator.", "name cleaned")
    end,
  },
}
