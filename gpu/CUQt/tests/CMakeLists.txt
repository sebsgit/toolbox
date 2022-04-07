cmake_minimum_required(VERSION 3.14)

project(CUQt LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Test)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Test)

enable_testing()

set(CUQT_CUDA_TEST_SOURCES testdevicememory.cpp testtextures.cpp)
set(TEST_SOURCES testbase.h testdevice.cpp testdevice.h teststream.h teststream.cpp testdevicememory.h testtextures.h ${CUQT_CUDA_TEST_SOURCES})

set_source_files_properties(${CUQT_CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(cuqt_test main.cpp ${TEST_SOURCES})

target_include_directories(cuqt_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include/")

target_link_libraries(cuqt_test PRIVATE Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::Test CUQt)
