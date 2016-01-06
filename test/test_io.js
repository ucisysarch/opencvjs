
module ("IO")
QUnit.test("Test IO", function(assert) {
	const CV_LOAD_IMAGE_COLOR = 1;
	// Imwrite/Imread
	{
		  let mat = new cv.Mat(48, 64, cv.CV_8UC4);
			Commons.createAlphaMat(mat);

			compressionParams = new cv.IntVector();
			compressionParams.push_back(16);
			compressionParams.push_back(9);

			cv.imwrite("alpha.png", mat, compressionParams);

			let mat2 = cv.imread("alpha.png", 1); // RGB
			assert.equal(mat.total(), mat2.total());
			assert.equal(mat2.channels(), 3);

			let mat3 = cv.imread("alpha.png", 0); //Grayscale
			assert.equal(mat.total(), mat3.total());
			assert.equal(mat3.channels(), 1);


			mat.delete();
			mat2.delete();
			mat3.delete();
			compressionParams.delete();
	}
	// Imencode/Imdecode
	{
			let mat = new cv.Mat(480, 640, cv.CV_8UC4),
					buff = new cv.UCharVector(),
					param = new cv.IntVector();

			Commons.createAlphaMat(mat);
			param.push_back(1); // CV_IMWRITE_JPEG_QUALITY
			param.push_back(95);
			cv.imencode(".png", mat, buff, param);

			let mat2 = cv.imdecode(new cv.Mat(buff), CV_LOAD_IMAGE_COLOR);

			assert.equal(mat.total(), mat2.total())

			mat.delete();
			buff.delete();
			mat2.delete();
			param.delete();
	}
	// Show image
	{
			let mat = new cv.Mat([50, 50], cv.CV_8UC4);

			Commons.createAlphaMat(mat);
			Commons.showImage(mat);

			mat.delete();
	}

});
