-- Music from your own files, played by the assistant (and ducked while it listens or talks). When
-- nothing of ours plays, the controls go to whatever player is running, through the media keys.
--
--   [plugins.music]
--   dir = "D:/Music"     # default: the system's Music folder
local config = ...

local dir = config.dir or courier.music_dir
local EXTS = { "mp3", "flac", "ogg", "wav", "m4a", "opus" }
local index -- built on first use: {path=, words=}

local function words(s)
  local out = {}
  for w in s:lower():gmatch("[%w']+") do out[#out + 1] = w end
  return out
end

local function build()
  index = {}
  if not dir then return end
  for _, p in ipairs(courier.list_files(dir, EXTS, 50000)) do
    -- The path below the music folder: artist, album and title are usually in it.
    local rel = p:sub(#dir + 2):gsub("%.[^.]+$", "")
    index[#index + 1] = { path = p, words = words(rel), name = rel:match("([^/\\]+)$") }
  end
end

local GENERIC = { music = true, song = true, songs = true, something = true, anything = true, some = true, any = true, play = true, me = true, a = true, the = true, by = true, my = true }

-- Tracks whose path has every word of the query (or most of them), best first.
local function find(query)
  if not index then build() end
  local q = {}
  for _, w in ipairs(words(query)) do
    if not GENERIC[w] then q[#q + 1] = w end
  end
  if #q == 0 then
    local all = {}
    for i, t in ipairs(index) do all[i] = t end
    for i = #all, 2, -1 do
      local j = math.random(i)
      all[i], all[j] = all[j], all[i]
    end
    return all, true
  end
  local scored = {}
  for _, t in ipairs(index) do
    local hits = 0
    for _, w in ipairs(q) do
      for _, tw in ipairs(t.words) do
        if tw == w or (#w >= 4 and tw:sub(1, #w) == w) then hits = hits + 1 break end
      end
    end
    if hits > 0 and hits >= math.max(1, #q - 1) then scored[#scored + 1] = { t = t, hits = hits } end
  end
  table.sort(scored, function(a, b) return a.hits > b.hits or (a.hits == b.hits and a.t.path < b.t.path) end)
  local out = {}
  for i, s in ipairs(scored) do out[i] = s.t end
  return out, false
end

local KEYS = { pause = "play_pause", resume = "play_pause", next = "next", previous = "previous", stop = "stop" }

return {
  tools = {
    {
      name = "play_music",
      description = "Play music from the user's music folder: an artist, album, song or genre, or anything.",
      parameters = { type = "object", properties = { query = { type = "string", description = "What to play; empty for anything" } } },
      run = function(a)
        local found, any = find(a.query or "")
        if #found == 0 then
          if not index or #index == 0 then return { error = "the music folder is empty or not set ([plugins.music] dir)" } end
          return { ok = false, say = "I found no music matching " .. a.query .. "." }
        end
        local paths = {}
        for i, t in ipairs(found) do
          if i > 200 then break end
          paths[i] = t.path
        end
        local st = courier.music("play", paths)
        local what = any and "Shuffling your music" or ("Playing " .. found[1].name)
        return { ok = true, tracks = #paths, say = what .. "." , status = st }
      end,
    },
    {
      name = "music_control",
      description = "Pause, resume, stop or skip the music, or tell which song is playing (status).",
      parameters = {
        type = "object",
        properties = { action = { type = "string", enum = { "pause", "resume", "stop", "next", "previous", "status" } } },
        required = { "action" },
      },
      run = function(a)
        local st = courier.music("status")
        local ours = st.playing or st.paused
        if not ours then
          if a.action == "status" then return { ok = true, say = "Nothing is playing." } end
          courier.key(KEYS[a.action])
          return { ok = true, say = "Done." }
        end
        if a.action == "status" then
          return { ok = true, status = st, say = (st.paused and "Paused: " or "Playing ") .. (st.track or "something") .. "." }
        end
        st = courier.music(a.action)
        local say = ({ pause = "Paused.", resume = "Resuming.", stop = "Stopped.", next = "Next: ", previous = "Back to " })[a.action]
        if a.action == "next" or a.action == "previous" then say = say .. (st.track or "the next one") .. "." end
        return { ok = true, status = st, say = say }
      end,
    },
  },
  tests = {
    finds_by_words = function(t)
      index = {
        { path = "M/Miles Davis/Kind of Blue/So What.mp3", words = words("Miles Davis/Kind of Blue/So What"), name = "So What" },
        { path = "M/Bach/Cello Suites/Prelude.flac", words = words("Bach/Cello Suites/Prelude"), name = "Prelude" },
        { path = "M/Jazz/Take Five.mp3", words = words("Jazz/Take Five"), name = "Take Five" },
      }
      local f = find("some miles davis")
      t.eq(f[1].name, "So What", "artist")
      f = find("bach cello")
      t.eq(#f, 1, "one match")
      f = find("anything")
      t.eq(#f, 3, "anything shuffles all")
      t.eq(#find("polka"), 0, "nothing")
      local r = t.call("play_music", { query = "take five" })
      t.eq(r.say, "Playing Take Five.", "plays")
    end,
  },
}
