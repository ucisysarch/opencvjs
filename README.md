# OpenCV.js

This is a JavaScript binding that exposes OpenCV library to the web. This project is made possible by support of Intel corporation.

### How to Build
0. Get the source codes
```
git clone https://github.com/sajjadt/opencvjs.git --recursive
```

1. Install emscripten 1.35.0 using [Emscripten SDK](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).
```
./emsdk install emsdk-1.35.0-64bit
./emsdk activate emsdk-1.35.0-64bit
```

2. Patch the Emscripten
```
patch -p1 < PATH/TO/patch_emscripten_1_35_0.diff
```
3. Compile OpenCV and generate bindings by executing make.py script
```
  python make.py
```

### Exported OpenCV subset
Classes and functions that are intended for binding generators (i.e. come with wrapping macros such as CV_EXPORTS_W and CV_WRAP) are exposed. Hence, supported OpenCV subset is comparable to OpenCV python binding.

Currently, the following modules are supported:

1. Core
2. Image processing
3. Photo
4. Shape
5. Video
6. Object detection
7. Features framework
8. Image codecs



### Examples
Test suite contains several examples demonstrating how different modules can be used.
