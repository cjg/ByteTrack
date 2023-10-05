cmake_minimum_required(VERSION 3.24)
project(ByteTrack)

set(CMAKE_CXX_STANDARD 11)

include(FetchContent)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_library(ByteTrack
        include/BYTETracker.h
        src/BYTETracker.cpp
        src/STrack.cpp
        src/kalmanFilter.cpp
        src/lapjv.cpp
        src/utils.cpp)

add_executable(byte-track-example
        example/example.cpp)
target_link_libraries(byte-track-example ByteTrack ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

