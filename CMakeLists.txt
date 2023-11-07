cmake_minimum_required(VERSION 3.24)
project(ByteTrack)

set(CMAKE_CXX_STANDARD 11)

include(FetchContent)

option(BUILD_EXAMPLES "Build examples (requires OpenCV)" OFF)

find_package(Eigen3 REQUIRED)

#include_directories(include)

add_library(ByteTrack
        include/BYTETracker.h
        src/BYTETracker.cpp
        src/STrack.cpp
        src/kalmanFilter.cpp
        src/lapjv.cpp
        src/utils.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${EIGEN3_INCLUDE_DIR}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include>
)
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}-config)
install(EXPORT ${PROJECT_NAME}-config
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
install(FILES
        ${CMAKE_CURRENT_LIST_DIR}/include/BYTETracker.h
        ${CMAKE_CURRENT_LIST_DIR}/include/kalmanFilter.h
        ${CMAKE_CURRENT_LIST_DIR}/include/lapjv.h
        ${CMAKE_CURRENT_LIST_DIR}/include/STrack.h
        TYPE INCLUDE)

if (BUILD_EXAMPLES)
    find_package(OpenCV REQUIRED)
    FetchContent_Declare(json
            URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
    FetchContent_MakeAvailable(json)
    add_executable(byte-track-example
            example/example.cpp)
    target_link_libraries(byte-track-example ByteTrack ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
endif ()