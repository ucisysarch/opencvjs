
width = 640;
height = 480;

function process_frame(video, ctx) {
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        // Draw frame
        ctx.drawImage(video, 0, 0, width, height);

        // Process the frame
        console.log("Processing frame");
        var raw = ctx.getImageData(0, 0, width, height);


        var img = cv.matFromArray(raw, 24),
            out_img = new cv.Mat();

        var t1 = performance.now();
        var f = kernels[Control.operation];
        out_img = f(img);
        var t2 = performance.now();
        console.log("Operation done in ", t2-t1, " ms");


        show_image(out_img, ctx);
        img.delete();
        out_img.delete();
    }
}


kernels = {};


function start_processing(stream) {
  var video = document.getElementById("webcam1");
  try {
      video.src = url.createObjectURL(stream);
  } catch (error) {
      video.src = stream;
  }
  setTimeout(function() {
          video.play();
      }, 500);

  frames = 0;
  var canvas = document.getElementById("canvas1");
  ctx = canvas.getContext("2d");

  var id = setInterval(function(){
    frames++; process_frame(video, ctx);

 }, 15);

}

function run_stuff() {
  var video_constraints =  {
      width: {exact: 640},
      height: {exact: 480}
   };
  getUserMedia(
    {
        audio: false,
        video: video_constraints
    }
    , start_processing, function (e) {
        console.error(e);
    });
}

function show_image(mat, canvas_context) {
  var data = mat.data(); 	// output is a Uint8Array that aliases directly into the Emscripten heap

  channels = mat.channels();
  channelSize = mat.elemSize1();

  canvas_context.clearRect(0, 0, width, height);

  imdata = canvas_context.createImageData(mat.cols, mat.rows);

  for (var i = 0,j=0; i < data.length; i += channels, j+=4) {
    imdata.data[j] = data[i];
    imdata.data[j + 1] = data[i+1%channels];
    imdata.data[j + 2] = data[i+2%channels];
    imdata.data[j + 3] = 255;
  }
  canvas_context.putImageData(imdata, 0, 0);
}


kernels['makeGray'] = function (src) {
  var res = new cv.Mat();
  cv.cvtColor(src, res, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
  return res;
}

kernels['makeHSV'] = function (src) {
  var tmp = new cv.Mat();
  var res = new cv.Mat();
  cv.cvtColor(src, tmp, cv.ColorConversionCodes.COLOR_RGBA2RGB.value, 0);
  cv.cvtColor(tmp, res, cv.ColorConversionCodes.COLOR_RGB2HSV.value, 0);
  tmp.delete();
  return res;
}

kernels['makeYUV'] = function (src) {
  var res = new cv.Mat();
  cv.cvtColor(src, res, cv.ColorConversionCodes.COLOR_RGB2YUV.value, 0);
  return res;
}

kernels['makeBGRA'] = function (src) {
  var res = new cv.Mat();
  cv.cvtColor(src, res, cv.ColorConversionCodes.COLOR_RGBA2BGRA.value, 0);
  return res;
}

kernels['blur'] = function(src) {
  var res = new cv.Mat();
  var threshold = Control.blurThreshold;
  cv.cvtColor(src, res, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
  cv.blur(res, res, [Control.blurSize, Control.blurSize], [-1, -1], cv.BORDER_DEFAULT);
  return res;
}

kernels['gaussianBlur'] = function(src) {
  var res = new cv.Mat();
  var size = [2*Control.blurSize+1, 2*Control.blurSize+1];
  cv.cvtColor(src, res, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
  cv.GaussianBlur(res, res, size, 0, 0, cv.BORDER_DEFAULT);
  return res;
}

kernels['medianBlur'] = function(src){
  var res = new cv.Mat();
  cv.cvtColor(src, res, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
  cv.medianBlur(res, res, 2*Control.blurSize+1);
  return res;
}

kernels['erode'] = function(src) {
  var res = new cv.Mat();
  var borderValue = cv.Scalar.all(Number.MAX_VALUE);

  var erosion_type = cv.MorphShapes.MORPH_RECT.value;
  var erosion_size = [2*Control.erosion_size+1, 2*Control.erosion_size+1];
  var element = cv.getStructuringElement(erosion_type, erosion_size, [-1, -1]);
  cv.cvtColor(src, res, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
  cv.erode(res, res, element, [-1, -1], 1, cv.BORDER_CONSTANT, borderValue);
  return res;
}

kernels['dilate'] = function(src) {
  var res = new cv.Mat();
  var borderValue = cv.Scalar.all(Number.MIN_VALUE);

  var erosion_type = cv.MorphShapes.MORPH_RECT.value;
  var erosion_size = [2*Control.erosion_size+1, 2*Control.erosion_size+1];
  var element = cv.getStructuringElement(erosion_type, erosion_size, [-1, -1]);
  cv.cvtColor(src, res, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
  cv.dilate(res, res, element, [-1, -1], 1, cv.BORDER_CONSTANT, borderValue);
  return res;
}


kernels['canny'] = function(src) {
  var res = new cv.Mat();
  var thresh = Control.canny_threshold;

  var blurred_img = kernels['gaussianBlur'](src);
  cv.Canny(blurred_img, res, thresh, thresh*2, 3, 0);
  blurred_img.delete();
  return res;
}

kernels['contours'] = function (src) {
  var cthresh = Control.canny_threshold;

  var canny_output = kernels.canny(src);

  /// Find contours
  var contours = new cv.MatVector();
  var hierarchy = new cv.Mat();
  cv.findContours(canny_output, contours, hierarchy, 3, 2, [0, 0]);

  // Convex hull
  var hull = new cv.MatVector();
  for( i = 0; i < contours.size(); i++ )
  {
    var item = new cv.Mat();
    cv.convexHull(contours.get(i), item, false, true);
    hull.push_back(item);
    item.delete();
  }

  // Draw contours + hull results
  var size = canny_output.size();
  var res = cv.Mat.zeros(size.get(0), size.get(1), cv.CV_8UC4);
  for(i = 0; i< contours.size(); i++ )
  {
    var color = new cv.Scalar(Math.random()*255, Math.random()*255, Math.random()*255);
    cv.drawContours(res, contours, i, color, 2, 8, hierarchy, 0, [0, 0]);
    var green = new cv.Scalar(30, 150, 30);
    cv.drawContours(res, hull, i, green, 1, 8, new cv.Mat(), 0, [0, 0]);
    color.delete();
    green.delete();
  }

  hull.delete();
  contours.delete();
  hierarchy.delete();
  canny_output.delete();
  return res;
}
