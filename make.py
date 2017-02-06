#!/usr/bin/python
import os, sys, re, json, shutil
from subprocess import Popen, PIPE, STDOUT

# Startup
exec(open(os.path.expanduser('~/.emscripten'), 'r').read())

try:
    EMSCRIPTEN_ROOT
except:
    print "ERROR: Missing EMSCRIPTEN_ROOT (which should be equal to emscripten's root dir) in ~/.emscripten"
    sys.exit(1)

#Popen('source ' + emenv)
sys.path.append(EMSCRIPTEN_ROOT)
import tools.shared as emscripten

# Settings
'''
          Settings.INLINING_LIMIT = 0
          Settings.DOUBLE_MODE = 0
          Settings.PRECISE_I64_MATH = 0
          Settings.CORRECT_SIGNS = 0
          Settings.CORRECT_OVERFLOWS = 0
          Settings.CORRECT_ROUNDINGS = 0
'''

#TODO No Filesystem
#emcc_args = sys.argv[1:] or '-O3 --llvm-lto 1 -s NO_EXIT_RUNTIME=1 -s ASSERTIOSN=1 -s AGGRESSIVE_VARIABLE_ELIMINATION=1 -s NO_DYNAMIC_EXECUTION=0 --memory-init-file 0 -s NO_FILESYSTEM#=1 -s NO_BROWSER=1'.split(' ')

# TODO: -msse2, SIMD.js is complaint with SSE2

emcc_args = '-O3 --llvm-lto 1 -s ASSERTIONS=0 -s AGGRESSIVE_VARIABLE_ELIMINATION=0 -s NO_DYNAMIC_EXECUTION=0 --memory-init-file 0 -s NO_FILESYSTEM=0'.split(' ')


print
print '--------------------------------------------------'
print 'Building opencv.js, build type:', emcc_args
print '--------------------------------------------------'
print


stage_counter = 0
def stage(text):
    global stage_counter
    stage_counter += 1
    text = 'Stage %d: %s' % (stage_counter, text)
    print
    print '=' * len(text)
    print text
    print '=' * len(text)
    print

# Main
try:
    this_dir = os.getcwd()
    os.chdir('opencv')
    if not os.path.exists('build'):
        os.makedirs('build')
    os.chdir('build')

    stage('OpenCV Configuration')
    configuration = ['cmake',
                     '-DCMAKE_BUILD_TYPE=RELEASE',
                     '-DBUILD_DOCS=OFF',
                     '-DBUILD_EXAMPLES=OFF',
                     '-DBUILD_PACKAGE=OFF',
                     '-DBUILD_WITH_DEBUG_INFO=OFF',
                     '-DBUILD_opencv_bioinspired=OFF',
                     '-DBUILD_opencv_calib3d=OFF',
                     '-DBUILD_opencv_cuda=OFF',
                     '-DBUILD_opencv_cudaarithm=OFF',
                     '-DBUILD_opencv_cudabgsegm=OFF',
                     '-DBUILD_opencv_cudacodec=OFF',
                     '-DBUILD_opencv_cudafeatures2d=OFF',
                     '-DBUILD_opencv_cudafilters=OFF',
                     '-DBUILD_opencv_cudaimgproc=OFF',
                     '-DBUILD_opencv_cudaoptflow=OFF',
                     '-DBUILD_opencv_cudastereo=OFF',
                     '-DBUILD_opencv_cudawarping=OFF',
                     '-DBUILD_opencv_gpu=OFF',
                     '-DBUILD_opencv_gpuarithm=OFF',
                     '-DBUILD_opencv_gpubgsegm=OFF',
                     '-DBUILD_opencv_gpucodec=OFF',
                     '-DBUILD_opencv_gpufeatures2d=OFF',
                     '-DBUILD_opencv_gpufilters=OFF',
                     '-DBUILD_opencv_gpuimgproc=OFF',
                     '-DBUILD_opencv_gpuoptflow=OFF',
                     '-DBUILD_opencv_gpustereo=OFF',
                     '-DBUILD_opencv_gpuwarping=OFF',
                     '-BUILD_opencv_hal=OFF',
                     '-DBUILD_opencv_highgui=ON',
                     '-DBUILD_opencv_java=OFF',
                     '-DBUILD_opencv_legacy=OFF',
                     '-DBUILD_opencv_ml=ON',
                     '-DBUILD_opencv_nonfree=OFF',
                     '-DBUILD_opencv_optim=OFF',
                     '-DBUILD_opencv_photo=ON',
                     '-DBUILD_opencv_shape=ON',
                     '-DBUILD_opencv_objdetect=ON',
                     '-DBUILD_opencv_softcascade=ON',
                     '-DBUILD_opencv_stitching=OFF',
                     '-DBUILD_opencv_superres=OFF',
                     '-DBUILD_opencv_ts=OFF',
                     '-DBUILD_opencv_videostab=OFF',
                     '-DENABLE_PRECOMPILED_HEADERS=OFF',
                     '-DWITH_1394=OFF',
                     '-DWITH_CUDA=OFF',
                     '-DWITH_CUFFT=OFF',
                     '-DWITH_EIGEN=OFF',
                     '-DWITH_FFMPEG=OFF',
                     '-DWITH_GIGEAPI=OFF',
                     '-DWITH_GSTREAMER=OFF',
                     '-DWITH_GTK=OFF',
                     '-DWITH_JASPER=OFF',
                     '-DWITH_JPEG=ON',
                     '-DWITH_OPENCL=OFF',
                     '-DWITH_OPENCLAMDBLAS=OFF',
                     '-DWITH_OPENCLAMDFFT=OFF',
                     '-DWITH_OPENEXR=OFF',
                     '-DWITH_PNG=ON',
                     '-DWITH_PVAPI=OFF',
                     '-DWITH_TIFF=OFF',
                     '-DWITH_LIBV4L=OFF',
                     '-DWITH_WEBP=OFF',
                     '-DWITH_PTHREADS_PF=OFF',
                     '-DBUILD_opencv_apps=OFF',
                     '-DBUILD_PERF_TESTS=OFF',
                     '-DBUILD_TESTS=OFF',
                     '-DBUILD_SHARED_LIBS=OFF',
                     '-DWITH_IPP=OFF',
                     '-DENABLE_SSE=OFF',
                     '-DENABLE_SSE2=OFF',
                     '-DENABLE_SSE3=OFF',
                     '-DENABLE_SSE41=OFF',
                     '-DENABLE_SSE42=OFF',
                     '-DENABLE_AVX=OFF',
                     '-DENABLE_AVX2=OFF',
                     '-DCMAKE_CXX_FLAGS=%s' % ' '.join(emcc_args),
                     '-DCMAKE_EXE_LINKER_FLAGS=%s' % ' '.join(emcc_args),
                     '-DCMAKE_CXX_FLAGS_DEBUG=%s' % ' '.join(emcc_args),
                     '-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_C_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_MODULE_LINKER_FLAGS_RELEASE=%s' % ' '.join(emcc_args),
                     '-DCMAKE_MODULE_LINKER_FLAGS_DEBUG=%s' % ' '.join(emcc_args),
                     '-DCMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE=%s' % ' '.join(emcc_args),
                     '-DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_SHARED_LINKER_FLAGS_DEBUG=%s' % ' '.join(emcc_args),
                     '..']
    emscripten.Building.configure(configuration)


    stage('Making OpenCV')

    emcc_args += ('-s TOTAL_MEMORY=%d' % (128*1024*1024)).split(' ') # default 128MB.
    emcc_args += '-s ALLOW_MEMORY_GROWTH=1'.split(' ')  # resizable heap
    emcc_args += '-s EXPORT_NAME="cv" -s MODULARIZE=1'.split(' ')


    emscripten.Building.make(['make', '-j4'])

    stage('Generating Bindings')
    INCLUDE_DIRS = [
             os.path.join('..', 'modules', 'core', 'include'),
             os.path.join('..', 'modules', 'flann', 'include'),
             os.path.join('..', 'modules', 'ml', 'include'),
             os.path.join('..', 'modules', 'photo', 'include'),
             os.path.join('..', 'modules', 'shape', 'include'),
             os.path.join('..', 'modules', 'imgproc', 'include'),
             os.path.join('..', 'modules', 'calib3d', 'include'),
             os.path.join('..', 'modules', 'features2d', 'include'),
             os.path.join('..', 'modules', 'video', 'include'),
             os.path.join('..', 'modules', 'objdetect', 'include'),
             os.path.join('..', 'modules', 'imgcodecs', 'include'),
             os.path.join('..', 'modules', 'hal', 'include')
    ]
    include_dir_args = ['-I'+item for item in INCLUDE_DIRS]
    emcc_binding_args = ['--bind']
    emcc_binding_args += include_dir_args

    emscripten.Building.emcc('../../bindings.cpp', emcc_binding_args, 'bindings.bc')
    assert os.path.exists('bindings.bc')

    stage('Building OpenCV.js')
    opencv = os.path.join('..', '..', 'build', 'cv.js')
    data = os.path.join('..', '..', 'build', 'cv.data')

    tests = os.path.join('..', '..', 'test')

    input_files = [
                'bindings.bc',
                os.path.join('lib','libopencv_core.a'),
                os.path.join('lib','libopencv_imgproc.a'),
                os.path.join('lib','libopencv_imgcodecs.a'),

                os.path.join('lib','libopencv_ml.a'),
                os.path.join('lib','libopencv_flann.a'),
                os.path.join('lib','libopencv_objdetect.a'),
                os.path.join('lib','libopencv_features2d.a') ,

                os.path.join('lib','libopencv_shape.a'),
                os.path.join('lib','libopencv_photo.a'),
                os.path.join('lib','libopencv_video.a'),

                # external libraries
                os.path.join('3rdparty', 'lib', 'liblibjpeg.a'),
                os.path.join('3rdparty', 'lib', 'liblibpng.a'),
                os.path.join('3rdparty', 'lib', 'libzlib.a'),
                #os.path.join('3rdparty', 'lib', 'liblibtiff.a'),
                #os.path.join('3rdparty', 'lib', 'liblibwebp.a'),
                ]

    emscripten.Building.link(input_files, 'libOpenCV.bc')
    emcc_args += '--preload-file ../../test/data/'.split(' ') #For testing purposes
    emcc_args += ['--bind']
#emcc_args += ['--memoryprofiler']

    emscripten.Building.emcc('libOpenCV.bc', emcc_args, opencv)
    stage('Wrapping')
    with open(opencv, 'r+b') as file:
        out = file.read()
        file.seek(0)
        # inspired by https://github.com/umdjs/umd/blob/95563fd6b46f06bda0af143ff67292e7f6ede6b7/templates/returnExportsGlobal.js
        file.write(("""
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define(function () {
            return (root.cv = factory());
        });
    } else if (typeof module === 'object' && module.exports) {
        // Node. Does not work with strict CommonJS, but
        // only CommonJS-like environments that support module.exports,
        // like Node.
        module.exports = factory();
    } else {
        // Browser globals
        root.cv = factory();
    }
}(this, function () {
    %s
    return cv(Module);
}));
""" % (out,)).lstrip())


    shutil.copy2(opencv, tests)
    if os.path.exists(data):
        shutil.copy2(data, tests)

finally:
    os.chdir(this_dir)
