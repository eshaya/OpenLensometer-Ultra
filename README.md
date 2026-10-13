# OpenLensometer-Ultra

Android app (Kotlin + C++/OpenCV) that measures a spectacle lens by looking through it at
patterns shown by `monitor-pattern-player.html` on a computer monitor.

## Building

The OpenCV Android SDK is not in the repository (it is about 1 GB). Download
OpenCV 4.10.0 for Android (`opencv-4.10.0-android-sdk.zip`) from
https://github.com/opencv/opencv/releases/tag/4.10.0 and unpack it so that this path exists:

    app/src/main/cpp/opencv/sdk/native/jni/OpenCVConfig.cmake

Then build and install with

    ./gradlew :app:assembleDebug
    adb install -r app/build/outputs/apk/debug/app-debug.apk
