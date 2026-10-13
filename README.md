# OpenLensometer-Ultra

Android app (Kotlin + C++/OpenCV) that measures a spectacle lens by looking through it at
patterns shown by `monitor-pattern-player.html` on a computer monitor.

> **Notice:** OpenLensometer is experimental software for educational and research use. It is
> not a medical device and has not been cleared or approved by the FDA or any other regulator.
> Its readings may be wrong and are not a prescription. It is no substitute for an eye
> examination or a professional lensmeter. Do not use it to order glasses or to make decisions
> about eye health. Provided "as is", without warranty of any kind (see LICENSE).

## How it works

The app uses the phone's camera to measure the optical shape of one lens at a time: its
sphericity (sphere power), its ellipticity (cylinder power) and the direction of that
ellipticity (axis). For bifocal and progressive lenses it also finds the near-vision add.

The lens is held in front of the camera with a grid of lines behind it, displayed by a web
browser on a computer monitor. The lens bends the lines it covers, and the app photographs
the grid through the lens and works out the lens's power from how the lines are distorted.
A tripod holds the phone still so the grid seen through the lens can be compared with the
grid seen without it.

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
