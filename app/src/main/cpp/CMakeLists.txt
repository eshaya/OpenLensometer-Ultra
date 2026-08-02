# File: lensometer-engine/src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.10.2)[cite: 1]

project("lensometer_engine")[cite: 1]

# NDK r26c supports modern C++ features required by Eigen
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point directly to your OpenCV SDK path as you had it configured
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni)[cite: 1]
find_package(OpenCV REQUIRED)[cite: 1]

# Include Eigen exactly as you mapped it
include_directories(${CMAKE_SOURCE_DIR}/eigen)[cite: 1]
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add all the cpp files you explicitly listed
add_library(
        lensometer_engine[cite: 1]
        SHARED[cite: 1]
        lensometer_engine.cpp[cite: 1]
        depth16_filter.cpp[cite: 1]
        phase_unwrapper.cpp[cite: 1]
        zernike_fitter.cpp[cite: 1]
        charuco_calibrator.cpp[cite: 1]
)

# Link everything together
target_link_libraries(
        lensometer_engine[cite: 1]
        ${OpenCV_LIBS}[cite: 1]
        log[cite: 1]
        jnigraphics[cite: 1] 
        android # Kept this as it is required for the new DirectByteBuffer zero-copy access
)