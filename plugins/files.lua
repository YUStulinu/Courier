-- Finding files by name and, for text files, by what they say; then opening one.
--
--   [plugins.files]
--   roots = ["D:/Work", "C:/Users/me/Documents"]   # default: Documents, Desktop, Downloads, Pictures, Music
local config = ...

local roots = config.roots
if not roots or #roots == 0 then
  roots = {}
  for _, k in ipairs({ "documents_dir", "desktop_dir", "downloads_dir", "pictures_dir", "music_dir" }) do
    if courier[k] then roots[#roots + 1] = courier[k] end
  end
end

local KINDS = {
  document = { "pdf", "doc", "docx", "odt", "rtf", "txt", "md", "xls", "xlsx", "ods", "csv", "ppt", "pptx", "odp" },
  image = { "jpg", "jpeg", "png", "gif", "webp", "heic", "bmp", "tif", "tiff" },
  audio = { "mp3", "flac", "ogg", "wav", "m4a", "aac", "opus" },
  video = { "mp4", "mkv", "mov", "avi", "webm" },
  code = { "py", "rs", "js", "ts", "c", "cpp", "h", "go", "java", "lua", "html", "css" },
}

local last = {}

-- "C:\Users\me\Documents\Taxes" -> "Documents, Taxes": a folder as you would say it.
local function short(folder)
  local parts = {}
  for p in folder:gmatch("[^/\\]+") do parts[#parts + 1] = p end
  local from = math.max(1, #parts - 1)
  return table.concat(parts, ", ", from)
end

return {
  tools = {
    {
      name = "search_files",
      description = "Find the user's files by words in their name or content.",
      parameters = {
        type = "object",
        properties = {
          query = { type = "string", description = "The words to look for (\"tax return 2025\")" },
          kind = { type = "string", enum = { "any", "document", "image", "audio", "video", "code" } },
        },
        required = { "query" },
      },
      run = function(a)
        local hits = courier.find_files({ roots = roots, query = a.query, exts = KINDS[a.kind or "any"], limit = 5 })
        last = hits
        if #hits == 0 then return { files = {}, say = "I found no files about " .. a.query .. "." } end
        local out, parts = {}, {}
        for i, h in ipairs(hits) do
          out[i] = { number = i, name = h.name, folder = short(h.folder), modified = h.modified, snippet = h.snippet }
          if i <= 3 then parts[#parts + 1] = h.name:gsub("%.[^.]+$", "") .. " in " .. short(h.folder) end
        end
        local head = #hits == 1 and "I found one file: " or ("I found " .. #hits .. " files. The best: ")
        return { files = out, say = head .. table.concat(parts, "; ") .. ". Shall I open one?" }
      end,
    },
    {
      name = "open_file",
      description = "Open a file from the last search, by its number (1 is the first).",
      parameters = { type = "object", properties = { number = { type = "integer", minimum = 1, maximum = 5 } } },
      run = function(a)
        local n = a.number or 1
        local h = last[n]
        if not h then return { error = #last == 0 and "search for the file first" or ("the last search found only " .. #last) } end
        courier.open(h.path)
        return { ok = true, say = "Opening " .. h.name:gsub("%.[^.]+$", "") .. "." }
      end,
    },
  },
  tests = {
    folders_are_said_shortly = function(t)
      t.eq(short("C:\\Users\\me\\Documents\\Taxes"), "Documents, Taxes", "two last parts")
      t.eq(short("/home/me"), "home, me", "unix")
    end,
    open_needs_a_search = function(t)
      last = {}
      t.ok(t.call("open_file", { number = 1 }).error:find("search") ~= nil, "asks for a search first")
    end,
  },
}
