# OpenLensometer-Ultra

Android app (Kotlin + C++/OpenCV) that measures a spectacle lens by looking through it at
patterns shown by `monitor-pattern-player.html` on a computer monitor.

## Using it

Install the app from the latest APK on the
[Releases page](https://github.com/eshaya/OpenLensometer-Ultra/releases) (needs a phone with ARCore).

1. Open `monitor-pattern-player.html` in a web browser on the computer whose monitor will show
   the patterns. Its first page explains the setup and the whole procedure.
2. Put the phone on a tripod with its back camera facing the browser window, about 50 cm away.
3. Start the app on the phone and follow the directions on the monitor and on the phone.

## Building

The OpenCV Android SDK is not in the repository (it is about 1 GB). Download
OpenCV 4.10.0 for Android (`opencv-4.10.0-android-sdk.zip`) from
https://github.com/opencv/opencv/releases/tag/4.10.0 and unpack it so that this path exists:

    app/src/main/cpp/opencv/sdk/native/jni/OpenCVConfig.cmake

Then build and install with

    ./gradlew :app:assembleDebug
    adb install -r app/build/outputs/apk/debug/app-debug.apk
