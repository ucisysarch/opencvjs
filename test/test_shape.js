
module ("Shapes")
QUnit.test("Test transformers", function(assert) {
// ShapeTransformer
{
	let regParamSize = 0;

	let transformer = new cv.ThinPlateSplineShapeTransformer(regParamSize);

	assert.equal(transformer.getRegularizationParameter(), 0);

	transformer.setRegularizationParameter(1);
	assert.equal(transformer.getRegularizationParameter(), 1);


	transformer.delete();

}
// AffineTransformer
{
	let transformer = new cv.AffineTransformer(true);
	assert.equal(transformer.getFullAffine(), true);
	transformer.delete();
}

});


QUnit.test("Test Histogram Extractor", function(assert) {
{
	let flag = cv.DistanceTypes.DIST_L2.value,
			numDummies = 20,
			cost = 80;

	let extractor = new cv.NormHistogramCostExtractor(flag, numDummies, cost);

	//assert.equal(extractor.getNormFlag(), flag);
	assert.equal(extractor.getNDummies(), numDummies);
	assert.equal(extractor.getDefaultCost(), cost);

	let matDim = 10;

	let mat1 = cv.Mat.eye([matDim, matDim], cv.CV_8UC4),
	 		mat2 = cv.Mat.ones([matDim, matDim], cv.CV_8UC4),
			mat3 = new cv.Mat();

	extractor.buildCostMatrix(mat1, mat2, mat3);

	assert.equal(mat3.rows, matDim + numDummies);
	assert.equal(mat3.channels(), 1);
	assert.equal(mat3.elemSize1(), 4);

	mat1.delete();
	mat2.delete();
	mat3.delete();
	extractor.delete();
}


});
