## Work in Progres... 
# opencvjs

This is a JavaScript binding for popular OpenCV library using Emscripten compiler and Embind. 

### Usage
0. Get the source code
```
git clone https://github.com/sajjadt/opencvjs.git --recursive
```

1. Install Emscripten
2. Execute make.py
```
  python make.py
```


### Examples
[Image processing module](http://sajjadt.github.io/opencvjs/examples/img_proc.html).
[Object detection module](http://sajjadt.github.io/opencvjs/examples/obj_detect.html).
[Features framework module](http://sajjadt.github.io/opencvjs/examples/features_2d.html).


### Availabe OpenCV Entities
1. Exported classes and functions intended for wrapper generators (i.e. CV_EXPORTS_W, ...)
2. Enums

### Notes
3. Default parameteres are also supported
4. Final script is compressed using Zee.js

