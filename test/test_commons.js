var Commons = {
  // Utility functions
	clamp: function(num, min, max) {
		return num < min ? min : num > max ? max : num;
	},

	createAlphaMat: function(mat) {
			let UCHAR_MAX =  255;
			for (var i = 0; i < mat.rows; ++i) {
					for (var j = 0; j < mat.cols; ++j) {
							var rgba = mat.ptr(i, j);
							rgba[0] = this.clamp((mat.rows - i)/(mat.rows) * UCHAR_MAX, 0, UCHAR_MAX); // Red
							rgba[1] = this.clamp((mat.cols - j)/(mat.cols) * UCHAR_MAX, 0, UCHAR_MAX); // Green
							rgba[2] = UCHAR_MAX; // Blue
							rgba[3] = this.clamp(0.5 * (rgba[1] + rgba[2]), 0, UCHAR_MAX); // Alpha
					}
			}
	},

	showImage: function(mat) {
		let canvas = document.createElement('canvas');
		canvas.style.zIndex   = 8;
		canvas.style.border   = "1px solid";
		document.body.appendChild(canvas);


		ctx = canvas.getContext("2d");
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		canvas.width = mat.cols;
		canvas.height = mat.rows;

		let imdata = ctx.createImageData(mat.cols, mat.rows);
		let data = mat.data(),
				channels = mat.channels(),
				channelSize = mat.elemSize1();

		for (var i = 0, j=0; i < data.length; i += channels, j+=4) {
			imdata.data[j] = data[i];
			imdata.data[j + 1] = data[i+1%channels];
			imdata.data[j + 2] = data[i+2%channels];
			imdata.data[j + 3] = 255;
		}

		ctx.putImageData(imdata, 0, 0);
	}
};
