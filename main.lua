package.path = package.path .. ";lua_modules/share/lua/5.4/?.lua"
require("fennel").install().dofile("main.fnl")
