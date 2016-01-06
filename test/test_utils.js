

QUnit.test("Test vectors", function(assert) {
	var pointVector = new cv.PointVector();
	for ( var i = 0 ; i< 100; ++i){
			pointVector.push_back([i, 2*i]);
	}

	assert.equal(pointVector.size(), 100);

	var index = 10;
	var item = pointVector.get(index);
	assert.equal(item[0], index);
	assert.equal(item[1], 2*index);

	pointVector.delete();
});
