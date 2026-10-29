-- An example plugin: the smallest useful one. Copy it to start your own; the assistant picks up new
-- and changed files in this folder while it runs.
return {
  tools = {
    {
      name = "roll_dice",
      description = "Roll dice.",
      parameters = {
        type = "object",
        properties = {
          count = { type = "integer", minimum = 1, maximum = 10 },
          sides = { type = "integer", minimum = 2, maximum = 100 },
        },
      },
      run = function(a)
        local n, sides = a.count or 1, a.sides or 6
        local rolls, total = {}, 0
        for i = 1, n do
          rolls[i] = math.random(sides)
          total = total + rolls[i]
        end
        local say = n == 1 and ("You rolled " .. rolls[1] .. ".")
          or ("You rolled " .. table.concat(rolls, ", ") .. ": " .. total .. " in all.")
        return { rolls = rolls, total = total, say = say }
      end,
    },
    {
      name = "flip_coin",
      description = "Flip a coin.",
      parameters = { type = "object", properties = {} },
      run = function() return { say = math.random(2) == 1 and "Heads." or "Tails." } end,
    },
  },
  tests = {
    in_range = function(t)
      for _ = 1, 50 do
        local r = t.call("roll_dice", { count = 3, sides = 4 })
        t.ok(r.total >= 3 and r.total <= 12, "total " .. r.total)
      end
      local s = t.call("flip_coin", {}).say
      t.ok(s == "Heads." or s == "Tails.", s)
    end,
  },
}
