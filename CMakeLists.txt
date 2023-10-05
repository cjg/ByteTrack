cmake_minimum_required(VERSION 3.24)
project(ByteTrack)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_library(ByteTrack
        src/BYTETracker.cpp
        src/STrack.cpp
        src/kalmanFilter.cpp
        src/lapjv.cpp
        src/utils.cpp)

add_executable(byte-track-example
        example/example.cpp)
target_link_libraries(byte-track-example ByteTrack)

