
module ("Video")
//QUnit.test("Tracking", function(assert) {
// meanShift
//{
//}
// buildOpticalFlowPyramid
//{
//}
// DualTVL1OpticalFlow
//{
//}
//});

QUnit.test("Background Segmentation", function(assert) {

// BackgroundSubtractorMOG2
{
	let history= 600,
			varThreshold=15,
      detectShadows=true;

	let mog2 = new cv.BackgroundSubtractorMOG2(history, varThreshold, detectShadows);

	assert.equal(mog2.getVarThreshold(), 15);
	assert.equal(mog2.getDetectShadows(), true);

	mog2.delete();
}

// BackgroundSubtractorKNN
{
	let history = 500,
			dist2Threshold = 350,
			numSamples = 10,
			detectShadows = false;

	let bsknn = new cv.BackgroundSubtractorKNN(history, dist2Threshold, detectShadows);
	bsknn.setNSamples(numSamples);
	assert.equal(bsknn.getDetectShadows(), detectShadows);
	assert.equal(bsknn.getHistory(), history);
	assert.equal(bsknn.getDist2Threshold(), dist2Threshold);
	assert.equal(bsknn.getNSamples(), numSamples);

	bsknn.delete();
}

// BackgroundSubtractorMOG2
{
	let history = 500,
			numMixtures = 8,
			varThreshold = 16,
			detectShadows = true;

	var bsmog2 = new cv.BackgroundSubtractorMOG2(history, varThreshold, detectShadows);
	bsmog2.setNMixtures(numMixtures);
	assert.equal(bsmog2.getDetectShadows(), detectShadows, "BSMOG2.getDetectShadows");
	assert.equal(bsmog2.getHistory(), history, "BSMOG2.getHistory");
	assert.equal(bsmog2.getVarThreshold(), varThreshold, "BSMOG2.getVarThreshold");
	assert.equal(bsmog2.getNMixtures(), numMixtures, "BSMOG2.getNMixtures");

	bsmog2.delete();
}

});
