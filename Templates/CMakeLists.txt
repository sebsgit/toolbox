cmake_minimum_required (VERSION 3.0.2)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

include_directories (${CMAKE_CURRENT_SOURCE_DIR})

set (SOURCES main.c)

add_executable (a.out ${SOURCES})
if (UNIX)
	target_link_libraries(a.out "lm")
endif (UNIX)
