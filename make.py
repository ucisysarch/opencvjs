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

emcc_args = sys.argv[1:] or '-O3 --llvm-lto 1 -s NO_EXIT_RUNTIME=1 -s AGGRESSIVE_VARIABLE_ELIMINATION=1 -s NO_DYNAMIC_EXECUTION=0 --memory-init-file 0 -s NO_FILESYSTEM=1 -s NO_BROWSER=1'.split(' ')

emcc_args += [ '-s', 'TOTAL_MEMORY=%d' % (64*1024*1024)] # default 64MB. Compile with ALLOW_MEMORY_GROWTH if you want a growable heap (slower though).
#emcc_args += ['-s', 'ALLOW_MEMORY_GROWTH=1'] # resizable heap, with some amount of slowness

emcc_args += '-s EXPORT_NAME="CV" -s MODULARIZE=1'.split(' ')

print
print '--------------------------------------------------'
print 'Building webcv.js, build type:', emcc_args
print '--------------------------------------------------'
print

'''
import os, sys, re
infile = open(sys.argv[1], 'r').read()
outfile = open(sys.argv[2], 'w')
t1 = infile
while True:
  t2 = re.sub(r'\(\n?!\n?1\n?\+\n?\(\n?!\n?1\n?\+\n?(\w)\n?\)\n?\)', lambda m: '(!1+' + m.group(1) + ')', t1)
  print len(infile), len(t2)
  if t1 == t2: break
  t1 = t2
outfile.write(t2)
'''

# Utilities

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

    configuration = ['cmake' , '-DBUILD_DOCS=OFF', '-DBUILD_PACKAGE=OFF', '-DBUILD_WITH_DEBUG_INFO=OFF' ,'-DBUILD_opencv_bioinspired=OFF' ,'-DBUILD_opencv_calib3d=OFF',  '-DBUILD_opencv_cuda=OFF','-DBUILD_opencv_cudaarithm=OFF', '-DBUILD_opencv_cudabgsegm=OFF', '-DBUILD_opencv_cudacodec=OFF', '-DBUILD_opencv_cudafeatures2d=OFF', '-DBUILD_opencv_cudafilters=OFF', '-DBUILD_opencv_cudaimgproc=OFF','-DBUILD_opencv_cudaoptflow=OFF', '-DBUILD_opencv_cudastereo=OFF' ,'-DBUILD_opencv_cudawarping=OFF', '-DBUILD_opencv_gpu=OFF', '-DBUILD_opencv_gpuarithm=OFF','-DBUILD_opencv_gpubgsegm=OFF', '-DBUILD_opencv_gpucodec=OFF', '-DBUILD_opencv_gpufeatures2d=OFF', '-DBUILD_opencv_gpufilters=OFF', '-DBUILD_opencv_gpuimgproc=OFF', '-DBUILD_opencv_gpuoptflow=OFF' ,'-DBUILD_opencv_gpustereo=OFF','-DBUILD_opencv_gpuwarping=OFF', '-DBUILD_opencv_highgui=ON', '-DBUILD_opencv_java=OFF', '-DBUILD_opencv_legacy=OFF', '-DBUILD_opencv_ml=ON', '-DBUILD_opencv_nonfree=OFF', '-DBUILD_opencv_optim=OFF', '-DBUILD_opencv_photo=OFF', '-DBUILD_opencv_shape=OFF', '-DBUILD_opencv_softcascade=OFF', '-DBUILD_opencv_stitching=OFF', '-DBUILD_opencv_superres=OFF', '-DBUILD_opencv_ts=OFF',  '-DBUILD_opencv_videostab=OFF', '-DENABLE_PRECOMPILED_HEADERS=ON', '-DWITH_1394=OFF','-DWITH_CUDA=OFF', '-DWITH_CUFFT=OFF', '-DWITH_EIGEN=OFF', '-DWITH_FFMPEG=OFF', '-DWITH_GIGEAPI=OFF', '-DWITH_GSTREAMER=OFF', '-DWITH_GTK=OFF', '-DWITH_JASPER=OFF', '-DWITH_JPEG=ON', '-DWITH_OPENCL=OFF', '-DWITH_OPENCLAMDBLAS=OFF', '-DWITH_OPENCLAMDFFT=OFF', '-DWITH_OPENEXR=OFF', '-DWITH_PNG=ON', '-DWITH_PVAPI=OFF', '-DWITH_TIFF=OFF', '-DWITH_LIBV4L=OFF', '-DWITH_WEBP=OFF' ,'-DBUILD_opencv_apps=OFF', '-DBUILD_PERF_TESTS=OFF',  '-DBUILD_SHARED_LIBS=OFF' , '..']


    emscripten.Building.configure(configuration)

    stage('OpenCV Make')

    emscripten.Building.make(['make', '-j'])


    stage('Generate Bindings')
    INCLUDE_DIRS = [
             os.path.join('..', 'modules' , 'core' , 'include'),
             os.path.join('..', 'modules' , 'flann' , 'include'),
             os.path.join('..', 'modules' , 'ml' , 'include'),
             os.path.join('..', 'modules' , 'imgproc' , 'include'),
             os.path.join('..', 'modules' , 'calib3d' , 'include'),
             os.path.join('..', 'modules' , 'features2d' , 'include'),
             os.path.join('..', 'modules' , 'video' , 'include'),
             os.path.join('..', 'modules' , 'objdetect' , 'include'),
             os.path.join('..', 'modules' , 'imgcodecs' , 'include'),
             os.path.join('..', 'modules' , 'videoio' , 'include'),
             os.path.join('..', 'modules' , 'highgui' , 'include') ,
             os.path.join('..', 'modules' , 'hal' , 'include')
    ]
    include_dir_args = ['-I'+item for item in INCLUDE_DIRS]
    print  include_dir_args 
    emcc_binding_args = [ '--bind' ]
    emcc_binding_args += include_dir_args
    #TODO Generate separate bindings for each module
    emscripten.Building.emcc('../../bindings.cpp', emcc_binding_args ,  'bindings.bc')
    assert os.path.exists('bindings.bc') 

    stage('Generate JS Libraries')

    core = os.path.join('..', '..', 'builds', 'core.js')
    imgproc = os.path.join('..', '..', 'builds', 'imgproc.js')
    imgcodecs = os.path.join('..', '..', 'builds', 'imgcodecs.js')
    ml =  os.path.join('..', '..', 'builds', 'ml.js')
    flann = os.path.join('..', '..', 'builds', 'flann.js')
    objdetect =  os.path.join('..', '..', 'builds', 'objdetect.js')
    features2d =  os.path.join('..', '..', 'builds', 'features2d.js')
    opencv =  os.path.join('..', '..', 'builds', 'opencv.html')
    
    
    #TODO emscripten.Building.emcc('libopencv_imgproc.bc', emcc_args + ['--pre-js', 'bindings_symbols.js')],  imgproc)
    
    
    
    emscripten.Building.emcc(os.path.join( 'lib' , 'libopencv_core.a' ), emcc_args + ['--bind' ,'bindings.bc' ] ,  core)  
    emscripten.Building.emcc(os.path.join('lib' ,'libopencv_imgproc.a'), emcc_args +  ['--bind' ,'bindings.bc' ]  ,  imgproc)
    emscripten.Building.emcc(os.path.join('lib' ,'libopencv_imgcodecs.a'), emcc_args +  ['--bind' ,'bindings.bc' ]  ,  imgcodecs)
    emscripten.Building.emcc(os.path.join('lib' ,'libopencv_ml.a'), emcc_args +  ['--bind' ,'bindings.bc' ]  ,  ml)
    emscripten.Building.emcc(os.path.join('lib' ,'libopencv_flann.a'), emcc_args +  ['--bind' ,'bindings.bc' ]  ,  flann)
    emscripten.Building.emcc(os.path.join('lib' ,'libopencv_objdetect.a'), emcc_args +  ['--bind' ,'bindings.bc' ]  ,  objdetect)
    emscripten.Building.emcc(os.path.join('lib' ,'libopencv_features2d.a'), emcc_args +  ['--bind' ,'bindings.bc' ]  ,  features2d)
    
    
    assert os.path.exists(core), 'Failed to create script code'
    assert os.path.exists(imgproc), 'Failed to create script code'
    assert os.path.exists(imgcodecs), 'Failed to create script code'
    assert os.path.exists(ml), 'Failed to create script code'
    assert os.path.exists(flann), 'Failed to create script code'
    assert os.path.exists(objdetect), 'Failed to create script code'
    assert os.path.exists(features2d), 'Failed to create script code'
    
    #TODO remove this
    input_files = [
                os.path.join('lib' ,'libopencv_core.a'),
                os.path.join('lib' ,'libopencv_imgproc.a'),
                os.path.join('lib' ,'libopencv_imgcodecs.a'),
                os.path.join('lib' ,'libopencv_ml.a'),
                os.path.join('lib' ,'libopencv_flann.a'),
                os.path.join('lib' ,'libopencv_objdetect.a'), 
                os.path.join('lib' ,'libopencv_features2d.a')
                ]
    emscripten.Building.emcc( input_files[0] , input_files[1:] + emcc_args  +  ['--bind' ,'bindings.bc' ] ,  opencv )
    
    
    stage('wrap')

    wrapped = '''
    // This is CV::Core, a JavaScript binding for OpenCV Core Library.
    ''' + open(core).read() + '''
    cv = CV();
    '''

    open(core, 'w').write(wrapped)
    
    wrapped = '''
     // This is CV::ImgProc, a port of OpenCV ImgProc Library to JavaScript.
    ''' + open(imgproc).read() + '''
    cv = CV();
    '''
    open(imgproc, 'w').write(wrapped)
    
finally:
    os.chdir(this_dir)
