# File: lensometer-engine/src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.10.2)

project("lensometer_engine")

# NDK r26c supports modern C++ features required by Eigen
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# CONDITIONAL DEPENDENCIES (OpenCV & Eigen)
# ---------------------------------------------------------
if(ANDROID)
    # Gradle Android Build: Point to your local Android paths
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni)
    include_directories(${CMAKE_SOURCE_DIR}/eigen)
else()
    # Local Linux Tests: Use system paths
    include_directories(/usr/include/eigen3)
endif()

# Find and include for both environments
find_package(OpenCV REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add all the cpp files for the Android library
# ==========================================
# ANDROID JNI LIBRARY
# ==========================================
if(ANDROID)
    # Add all the cpp files for the Android library
    add_library(
            lensometer_engine
            SHARED
            lensometer_engine.cpp
            alignment_detector.cpp
            frame_stack.cpp
            lens_measurement.cpp
            grid_ftp.cpp
            aberration_removal.cpp
            depth16_filter.cpp
            phase_unwrapper.cpp
            zernike_fitter.cpp
            charuco_calibrator.cpp
            wavefront_calibrator.cpp
    )

    # Link everything together for the Android build
    target_link_libraries(
            lensometer_engine
            ${OpenCV_LIBS}
            log
            jnigraphics 
            android 
    )
endif()
# ==========================================
# Unit Testing Configuration (Host Build Only)
# ==========================================
if(NOT ANDROID)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        release-1.12.1
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Create the host test executable
    add_executable(
        lensometer_tests
        test_lensometer.cpp
        alignment_detector.cpp
        frame_stack.cpp
        lens_measurement.cpp
        grid_ftp.cpp
        zernike_fitter.cpp
        phase_unwrapper.cpp
        wavefront_calibrator.cpp 
        charuco_calibrator.cpp
    )

    target_link_libraries(
        lensometer_tests
        gtest_main
        ${OpenCV_LIBS}
    )
    
    include(GoogleTest)
    gtest_discover_tests(lensometer_tests)
endif()
