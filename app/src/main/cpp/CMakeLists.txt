cmake_minimum_required(VERSION 3.10.2)
project("lensometer_engine")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni)
find_package(OpenCV REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/eigen)

add_library(
        lensometer_engine
        SHARED
        lensometer_engine.cpp
        depth16_filter.cpp
	phase_unwrapper.cpp
)

target_link_libraries(
        lensometer_engine
        ${OpenCV_LIBS}
        log
        jnigraphics
)
