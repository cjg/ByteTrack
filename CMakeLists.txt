cmake_minimum_required(VERSION 3.24)
project(ByteTrack)

set(CMAKE_CXX_STANDARD 11)

include(FetchContent)

option(BUILD_EXAMPLES "Build examples (requires OpenCV)" OFF)

find_package(Eigen3 REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_library(ByteTrack
        include/BYTETracker.h
        src/BYTETracker.cpp
        src/STrack.cpp
        src/kalmanFilter.cpp
        src/lapjv.cpp
        src/utils.cpp)

if (BUILD_EXAMPLES)
    find_package(OpenCV REQUIRED)
    FetchContent_Declare(json
            URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
    FetchContent_MakeAvailable(json)
    add_executable(byte-track-example
            example/example.cpp)
    target_link_libraries(byte-track-example ByteTrack ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
endif ()