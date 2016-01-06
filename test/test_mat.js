/*/////////////////////////////////////////////////////////////////////////////
AUTHOR: Sajjad Taheri sajjadt[at]uci[at]edu

                             LICENSE AGREEMENT
Copyright (c) 2015, University of california, Irvine

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:
   This product includes software developed by the UC Irvine.
4. Neither the name of the UC Irvine nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UC IRVINE ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL UC IRVINE OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/////////////////////////////////////////////////////////////////////////////*/

module ("Core")
QUnit.test("test_mat_creation", function(assert) {
  // Mat constructors.
  // Mat::Mat(int rows, int cols, int type)
  {
    let mat = new cv.Mat(10, 20, cv.CV_8UC3);

    assert.equal(mat.type(), cv.CV_8UC3);
    assert.equal(mat.depth(), cv.CV_8U);
    assert.equal(mat.channels(), 3);
    assert.ok(mat.empty() === false);

    let size = mat.size();
    assert.ok(size.size() === 2);
    assert.equal(size.get(0), 10);
    assert.equal(size.get(1), 20);

    size.delete();
    mat.delete();
  }

  // Mat::Mat(const Mat &)
  {
    //  : Copy from another Mat
    //let mat1 = new cv.Mat(10, 20, cv.CV_8UC3);
    //let mat2 = new cv.Mat(mat1);

    //assert.equal(mat2.type(), mat1.type());
    //assert.equal(mat2.depth(), mat1.depth());
    //assert.equal(mat2.channels(), mat1.channels());
    //assert.equal(mat2.empty(), mat1.empty());

    //let size1 = mat1.size();
    //let size2 = mat2.size();
    //assert.ok(size1.size() === size2.size());
    //assert.ok(size1.get(0) === size2.get(0));
    //assert.ok(size1.get(1) === size2.get(1));

    //mat1.delete();
    //mat2.delete();
  }

  // Mat::Mat(Size size, int type, void *data, size_t step=AUTO_STEP)
  {
    // 10 * 10 and one channel
    let data = cv._malloc(10 * 10 * 1);
    let mat = new cv.Mat([10, 10], cv.CV_8UC1, data, 0);

    assert.equal(mat.type(), cv.CV_8UC1);
    assert.equal(mat.depth(), cv.CV_8U);
    assert.equal(mat.channels(), 1);
    assert.ok(mat.empty() === false);

    let size = mat.size();
    assert.ok(size.size() === 2);
    assert.ok(size.get(0) === 10);
    assert.ok(size.get(1) === 10);

    size.delete();
    mat.delete();
  }

  //  Mat::create(int, int, int)
  {
    let mat = new cv.Mat();
    mat.create(10, 5, cv.CV_8UC3);
    let size = mat.size();

    assert.ok(mat.type() === cv.CV_8UC3);
    assert.ok(size.get(0) === 10);
    assert.ok(size.get(1) === 5);
    assert.ok(mat.channels() === 3);

    size.delete();
    mat.delete();
  }
  //  Mat::create(Size, int)
  {
    let mat = new cv.Mat();
    mat.create([10, 5], cv.CV_8UC4);
    let size = mat.size();

    assert.ok(mat.type() === cv.CV_8UC4);
    assert.ok(size.get(0) === 10);
    assert.ok(size.get(1) === 5);
    assert.ok(mat.channels() === 4);

    size.delete();
    mat.delete();
  }
});

QUnit.test("test_mat_ptr", function(assert) {
  const RValue = 3;
  const GValue = 7;
  const BValue = 197;

  // cv.CV_8UC1 + Mat::ptr(int).
  {
    let mat = new cv.Mat(10, 10, cv.CV_8UC1);
    let view = mat.data();

    // Alter matrix[2, 1].
    let step = 10;
    view[2 * step + 1] = RValue;

    // Access matrix[2, 1].
    view = mat.ptr(2);

    assert.equal(view[1], RValue);
  }

  // cv.CV_8UC3 + Mat::ptr(int).
  {
    let mat = new cv.Mat(10, 10, cv.CV_8UC3);
    let view = mat.data();

    // Alter matrix[2, 1].
    let step = 3 * 10;
    view[2 * step + 3] = RValue;
    view[2 * step + 3 + 1] = GValue;
    view[2 * step + 3 + 2] = BValue;

    // Access matrix[2, 1].
    view = mat.ptr(2);

    assert.equal(view[3], RValue);
    assert.equal(view[3 + 1], GValue);
    assert.equal(view[3 + 2], BValue);
  }

  // cv.CV_8UC3 + Mat::ptr(int, int).
  {
    let mat = new cv.Mat(10, 10, cv.CV_8UC3);
    let view = mat.data();

    // Alter matrix[2, 1].
    let step = 3 * 10;
    view[2 * step + 3] = RValue;
    view[2 * step + 3 + 1] = GValue;
    view[2 * step + 3 + 2] = BValue;

    // Access matrix[2, 1].
    view = mat.ptr(2, 1);

    assert.equal(view[0], RValue);
    assert.equal(view[1], GValue);
    assert.equal(view[2], BValue);
  }
});

QUnit.test("test_mat_zeros", function(assert) {
  // Mat::zeros(int, int, int)
  {
    let mat = cv.Mat.zeros(10, 10, cv.CV_8UC1);
    let view = mat.data();

    // TBD
    // Figurr out why array.prototype is undefined. Since that's undifined, I can't
    // use any array member function, such as every/forEach.
    //assert.ok(view.every(function(x) { return x === 0; }));

    var total = 0;
    for (let i = 0; i < 100; i++) {
      total += view[i];
    }

    assert.ok(total === 0);

    mat.delete();
  }

  // Mat::zeros(Size, int)
  {
    let mat = cv.Mat.zeros([10, 10], cv.CV_8UC1);
    let view = mat.data();

    var total = 0;
    for (let i = 0; i < 100; i++) {
      total += view[i];
    }

    assert.ok(total === 0);

    mat.delete();
  }
});

QUnit.test("test_mat_ones", function(assert) {
  // Mat::ones(int, int, int)
  {
    var mat = cv.Mat.ones(10, 10, cv.CV_8UC1);
    var view = mat.data();

    var total = 0;
    for (let i = 0; i < 100; i++) {
      total += view[i];
    }

    assert.ok(total === 100);
  }
  // Mat::ones(Size, int)
  {
    var mat = cv.Mat.ones([10, 10], cv.CV_8UC1);
    var view = mat.data();

    var total = 0;
    for (let i = 0; i < 100; i++) {
      total += view[i];
    }

    assert.ok(total === 100);
  }
});

QUnit.test("test_mat_eye", function(assert) {
  // Mat::eye(int, int, int)
  {
    var mat = cv.Mat.eye(10, 10, cv.CV_8UC1);
    var view = mat.data();

    var total = 0;
    for (let i = 0; i < 100; i++) {
      total += view[i];
    }

    assert.ok(total === 10);
  }

  // Mat::eye(Size, int)
  {
    var mat = cv.Mat.eye([10, 10], cv.CV_8UC1);
    var view = mat.data();

    var total = 0;
    for (let i = 0; i < 100; i++) {
      total += view[i];
    }

    assert.ok(total === 10);
  }
});

QUnit.test("test_mat_miscs", function(assert) {
  // Mat::col(int)
  {
    let mat = cv.Mat.ones(5, 5, cv.CV_8UC2);
    let col = mat.col(1);
    let view = col.data();
    assert.equal(view[0], 1);
    assert.equal(view[4], 1);

    col.delete();
    mat.delete();
  }

  // Mat::row(int)
  {
    let mat = cv.Mat.zeros(5, 5, cv.CV_8UC2);
    let row = mat.row(1);
    let view = row.data();
    assert.equal(view[0], 0);
    assert.equal(view[4], 0);

    row.delete();
    mat.delete();
  }

  // Mat::convertTo(Mat, int, double, double)
  {
    let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
    let grayMat = cv.Mat.zeros(5, 5, cv.CV_8UC1);

    mat.convertTo(grayMat, cv.CV_8U, 2, 1);
    // dest = 2 * source(x, y) + 1.
    let view = grayMat.data();
    assert.equal(view[0], (1 * 2) + 1);

    grayMat.delete();
    mat.delete();
  }

  // C++
  //   void split(InputArray, OutputArrayOfArrays)
  // Embind
  //   void split(VecotrMat, VectorMat)
  {
    const R =7;
    const G =13;
    const B =29;

    let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
    let view = mat.data();
    view[0] = R;
    view[1] = G;
    view[2] = B;

    let bgr_planes = new cv.MatVector();
    cv.split(mat, bgr_planes);
    assert.equal(bgr_planes.size(), 3);

    let rMat = bgr_planes.get(0);
    view = rMat.data();
    assert.equal(view[0], R);

    let gMat = bgr_planes.get(1);
    view = gMat.data();
    assert.equal(view[0], G);

    let bMat = bgr_planes.get(2);
    view = bMat.data();
    assert.equal(view[0], B);

    mat.delete();
    rMat.delete();
    gMat.delete();
    bMat.delete();
  }

  // C++
  //   size_t Mat::elemSize() const
  {
    let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
    assert.equal(mat.elemSize(), 3);
    assert.equal(mat.elemSize1(), 1);

    let mat2 = cv.Mat.zeros(5, 5, cv.CV_8UC1);
    assert.equal(mat2.elemSize(), 1);
    assert.equal(mat2.elemSize1(), 1);

    let mat3 = cv.Mat.eye(5, 5, cv.CV_16UC3);
    assert.equal(mat3.elemSize(), 2 * 3);
    assert.equal(mat3.elemSize1(), 2);

    mat.delete();
    mat2.delete();
    mat3.delete();
  }

	//   double Mat::dot(const Mat&) const
  {
    let mat = cv.Mat.ones(5, 5, cv.CV_8UC1);
		let mat2 = cv.Mat.eye(5, 5, cv.CV_8UC1);

    assert.equal(mat.dot(mat), 25);
		assert.equal(mat.dot(mat2), 5);
		assert.equal(mat2.dot(mat2), 5);

    mat.delete();
    mat2.delete();
  }

	//   Element-wise multiplication
	//   double Mat::mul(const Mat&) const
	{
		let mat = cv.Mat.ones(5, 5, cv.CV_8UC1);
		let mat2 = cv.Mat.eye(5, 5, cv.CV_8UC1);

		let mat3 = mat.mul(mat2, 3);

		let size = mat3.size();
		assert.equal(size.get(0), 5);
		assert.equal(size.get(1), 5);

		var arr = mat3.data();
		var total = 0;
		for(var i=0; i < 25; i+=1) {
			total += arr[i];
		}
		assert.equal(total, 5*3);

		mat.delete();
		mat2.delete();
		mat3.delete();
	}

	//   clone
  {
    let mat = cv.Mat.ones(5, 5, cv.CV_8UC1);
		let mat2 = mat.clone();

		let arr = mat.data(),
				arr2 = mat2.data();

		assert.equal(mat.channels, mat2.channels);
		assert.equal(mat.size()[0], mat2.size()[0]);
		assert.equal(mat.size()[1], mat2.size()[1]);

		var equal = (arr.length == arr2.length) && arr.every(function(element, index) {
				return element === arr2[index];
		});
		assert.equal(equal, true);


    mat.delete();
    mat2.delete();
  }
});


QUnit.test("test_memory_view", function(assert) {

{
		let data = new Uint8Array([0, 0, 0, 255, 0, 1, 2, 3]),
	      dataPtr = cv._malloc(8);

		let dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr, 8);
		dataHeap.set(new Uint8Array(data.buffer));

		let mat = new cv.Mat([8, 1], cv.CV_8UC1, dataPtr, 0);


		let unsignedCharView = new Uint8Array(data.buffer),
				charView = new Int8Array(data.buffer),
				shortView = new Int16Array(data.buffer),
				unsignedShortView = new Uint16Array(data.buffer),
				intView = new Int32Array(data.buffer),
				float32View = new Float32Array(data.buffer),
				float64View = new Float64Array(data.buffer);


		assert.deepEqual(unsignedCharView, mat.data());
		assert.deepEqual(charView, mat.data8S());
		assert.deepEqual(shortView, mat.data16s());
		assert.deepEqual(unsignedShortView, mat.data16u());
		assert.deepEqual(intView, mat.data32s());
		assert.deepEqual(float32View, mat.data32f());
		assert.deepEqual(float64View, mat.data64f());
}

});
