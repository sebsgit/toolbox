cmake_minimum_required(VERSION 3.14)

project(CUQt LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

set(CUQT_SOURCES "")

set(CUQT_CUDA_SOURCES
    src/cuqt.cpp
    src/cuqtdevice.cpp
    src/cuqtdebugoperators.cpp
    src/cuqtstream.cpp
)

set(CUQT_HEADERS
    include/cuqt.h
    include/cuqtdevice.h
    include/cuqtstream.h)

add_library(CUQt SHARED
    ${CUQT_HEADERS}
    ${CUQT_SOURCES}
    ${CUQT_CUDA_SOURCES}
)

set_source_files_properties(${CUQT_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

target_link_libraries(CUQt PRIVATE Qt${QT_VERSION_MAJOR}::Core)

target_include_directories(CUQt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_compile_definitions(CUQt PRIVATE CUQT_LIBRARY)

add_subdirectory(tests)
