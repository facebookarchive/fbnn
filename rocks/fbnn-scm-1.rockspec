package = "fbnn"
version = "scm-1"

source = {
   url = "git://github.com/facebook/fbnn.git"
}

description = {
   summary = "Facebook extensions for the neural network package",
   detailed = [[
   ]],
   homepage = "https://github.com/facebook/fbnn",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
   "torch >= 7.0",
   "nn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)";
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
