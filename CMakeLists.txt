CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(LUAROCKS_PREFIX)
  MESSAGE(STATUS "Installing fbnn through Luarocks")
  STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
  MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()
FIND_PACKAGE(Torch REQUIRED)

FILE(GLOB luasrc fbnn/*.lua)
SET(src
  init.c
  )
ADD_TORCH_PACKAGE(fbnn "${src}" "${luasrc}" "facebook's extensions to nn")
