# OpenCV.js

This is a JavaScript binding that exposes OpenCV library to the web. This project is made possible by support of Intel corporation.

### How to Build
1. Get the source code
  
  ```
  git clone https://github.com/ucisysarch/opencvjs.git --recursive
  ```
2. Install emscripten 1.35.0. Other versions can be used as well, but currently, the patch is provided for only this version. You can obtain emscripten by using [Emscripten SDK](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).
  
  ```
  ./emsdk install sdk-1.35.0-64bit
  ./emsdk activate sdk-1.35.0-64bit
  ```
3. Patch the Emscripten
  
  ```
  patch -p1 < PATH/TO/patch_emscripten_1_35_0.diff
  ```
4. Compile OpenCV and generate bindings by executing make.py script.
  
  ```
    python make.py
  ```

### Exported OpenCV Subset
Classes and functions that are intended for binding generators (i.e. come with wrapping macros such as CV_EXPORTS_W and CV_WRAP) are exposed. Hence, supported OpenCV subset is comparable to OpenCV for Python. Also, enums with exception of anonymous enums are also exported.

Currently, the following modules are supported. You can modify the make script to exclude certain modules.

1. Core
2. Image processing
3. Photo
4. Shape
5. Video
6. Object detection
7. Features framework
8. Image codecs

### At a glance
The following example demonstrates how to apply a gaussian blur filter on an image. Note that everything is wrapped in a JavaScript module ('cv'). 

```Javascript
  // Gaussian Blur
  var mat1 = cv.Mat.ones(7, 7, cv.CV_8UC1),
      mat2 = new cv.Mat();

  cv.GaussianBlur(mat1, mat2, [3, 3], 0, 0, cv.BORDER_DEFAULT);
  
  mat1.delete();
  mat2.delete();
```
Next example shows how to calculate image keypoints and their descriptors using ORB (Oriented Brief) method.
```Javascript
  var numFeatures = 900,
	    scaleFactor = 1.2,
	    numLevels = 8,
	    edgeThreshold = 31,
		  firstLevel =0,
		  WTA_K= 2,
		  scoreType = 0, //ORB::HARRIS_SCORE
		  patchSize = 31,
		  fastThreshold=20,
		  keyPoints = new cv.KeyPointVector(),
		  descriptors = new cv.Mat();
  
	var orb = new cv.ORB(numFeatures, scaleFactor, numLevels, edgeThreshold, firstLevel,
									     WTA_K, scoreType, patchSize, fastThreshold);

  // image and mask are of type cv.Mat
	orb.detect(image, keyPoints, mask);
	orb.compute(image, keyPoints, descriptors);

	keyPoints.delete();
	descriptors.delete();
	orb.delete();
```

Functions work on cv::Mat and various vectors. The following vectors are registered and can be used.

```cpp
  register_vector<int>("IntVector");
  register_vector<unsigned char>("UCharVector"););
  register_vector<float>("FloatVector");
  register_vector<std::vector<Point>>("PointVectorVector");
  register_vector<cv::Point>("PointVector");
  register_vector<cv::Mat>("MatVector");
  register_vector<cv::KeyPoint>("KeyPointVector");
  register_vector<cv::Rect>("RectVector");
  register_vector<cv::Point2f>("Point2fVector");
```

### More Examples
Test suite contains several tests and examples demonstrating how the API can be used.

### Memory management
All the allocated objects should be freed manually by calling delete() method. To avoid manual memory management for basic types, the following data types are exported as JavaScript value arrays.

```
cv.Size
cv.Point
```

### File System Access
If your OpenCV application needs to access a file, for instance a dataset or a previoulsy trained classifier, you can modify the make script and attach the files by using emscripten "--preload-file" flag. 


### Limitations
1. MatExpr is not exported.
2. No support for default parameters yet.
2. Constructor overloading are implemented by number of paramteres and not their types. Hence, only following Mat constructors are exported.

```cpp
  cv::Mat()
  cv::Mat(const std::vector<unsigned char>& data)
  cv::Mat(Size size, int type)
  cv::Mat(int rows, int cols, int type)
  cv::Mat(Size size, int type, void* data, size_t step)
```
