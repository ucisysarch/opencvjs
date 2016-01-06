# OpenCV.js

This is a JavaScript binding that exposes OpenCV library to the web platform. This project is made possible by support of Intel Corporation.

### How to Build
0. Get the source codes
```
git clone https://github.com/sajjadt/opencvjs.git --recursive
```

1. Install Emscripten
2. Execute make.py
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

### How it looks


### Examples
[Image processing module](http://sajjadt.github.io/opencvjs/examples/img_proc.html).

[Object detection module](http://sajjadt.github.io/opencvjs/examples/obj_detect.html).

[Features framework module](http://sajjadt.github.io/opencvjs/examples/features_2d.html).
