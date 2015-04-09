#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <emscripten/bind.h>
using namespace emscripten;
using namespace cv;

    EMSCRIPTEN_BINDINGS(testBinding) {
    

    emscripten::class_<cv::CLAHE>("CLAHE")
    
    .function("setTilesGridSize", select_overload<void(Size)>(&cv::CLAHE::setTilesGridSize))

    .function("collectGarbage", select_overload<void()>(&cv::CLAHE::collectGarbage))

    .function("setClipLimit", select_overload<void(double)>(&cv::CLAHE::setClipLimit))

    .function("getTilesGridSize", select_overload<Size()const>(&cv::CLAHE::getTilesGridSize))

    .function("getClipLimit", select_overload<double()const>(&cv::CLAHE::getClipLimit))

    .function("apply", select_overload<void(InputArray,OutputArray)>(&cv::CLAHE::apply))

    ;

    emscripten::class_<cv::LineSegmentDetector>("LineSegmentDetector")
    
    .function("compareSegments", select_overload<int(const Size&,InputArray,InputArray,InputOutputArray)>(&cv::LineSegmentDetector::compareSegments))

    .function("detect", select_overload<void(InputArray,OutputArray,OutputArray,OutputArray,OutputArray)>(&cv::LineSegmentDetector::detect))

    .function("drawSegments", select_overload<void(InputOutputArray,InputArray)>(&cv::LineSegmentDetector::drawSegments))

    ;

    emscripten::class_<cv::Mat>("Mat")
    
    .function("elemSize1", select_overload<size_t()const>(&cv::Mat::elemSize1))

    .function("assignTo", select_overload<void(Mat&,int)const>(&cv::Mat::assignTo))

    .function("inv", select_overload<MatExpr(int)const>(&cv::Mat::inv))

    .function("channels", select_overload<int()const>(&cv::Mat::channels))

    .function("isContinuous", select_overload<bool()const>(&cv::Mat::isContinuous))

    .function("checkVector", select_overload<int(int,int,bool)const>(&cv::Mat::checkVector))

    .function("convertTo", select_overload<void(OutputArray,int,double,double)const>(&cv::Mat::convertTo))

    .function("total", select_overload<size_t()const>(&cv::Mat::total))

    .function("row", select_overload<Mat(int)const>(&cv::Mat::row))

    .class_function("eye",select_overload<MatExpr(int,int,int)>(&cv::Mat::eye))

    .class_function("eye",select_overload<MatExpr(Size,int)>(&cv::Mat::eye))

    .constructor<  >()

    .constructor< int,int,int >()

    .constructor< int,int,int,const Scalar& >()

    .constructor< const Mat& >()

    .constructor< int,int,int,void*,size_t >()

    .constructor< const Mat&,const Rect& >()

    .function("reshape", select_overload<Mat(int,int)const>(&cv::Mat::reshape))

    .function("reshape", select_overload<Mat(int,int,const int*)const>(&cv::Mat::reshape), allow_raw_pointers())

    .function("create", select_overload<void(int,int,int)>(&cv::Mat::create))

    .function("create", select_overload<void(Size,int)>(&cv::Mat::create))

    .function("rowRange", select_overload<Mat(int,int)const>(&cv::Mat::rowRange))

    .function("rowRange", select_overload<Mat(const Range&)const>(&cv::Mat::rowRange))

    .function("cross", select_overload<Mat(InputArray)const>(&cv::Mat::cross))

    .function("copyTo", select_overload<void(OutputArray)const>(&cv::Mat::copyTo))

    .function("copyTo", select_overload<void(OutputArray,InputArray)const>(&cv::Mat::copyTo))

    .class_function("zeros",select_overload<MatExpr(int,int,int)>(&cv::Mat::zeros))

    .class_function("zeros",select_overload<MatExpr(Size,int)>(&cv::Mat::zeros))

    .class_function("zeros",select_overload<MatExpr(int,const int*,int)>(&cv::Mat::zeros), allow_raw_pointers())

    .function("elemSize", select_overload<size_t()const>(&cv::Mat::elemSize))

    .function("mul", select_overload<MatExpr(InputArray,double)const>(&cv::Mat::mul))

    .function("type", select_overload<int()const>(&cv::Mat::type))

    .function("empty", select_overload<bool()const>(&cv::Mat::empty))

    .function("colRange", select_overload<Mat(int,int)const>(&cv::Mat::colRange))

    .function("colRange", select_overload<Mat(const Range&)const>(&cv::Mat::colRange))

    .function("step1", select_overload<size_t(int)const>(&cv::Mat::step1))

    .function("clone", select_overload<Mat()const>(&cv::Mat::clone))

    .class_function("ones",select_overload<MatExpr(int,int,int)>(&cv::Mat::ones))

    .class_function("ones",select_overload<MatExpr(Size,int)>(&cv::Mat::ones))

    .class_function("ones",select_overload<MatExpr(int,const int*,int)>(&cv::Mat::ones), allow_raw_pointers())

    .function("depth", select_overload<int()const>(&cv::Mat::depth))

    .function("t", select_overload<MatExpr()const>(&cv::Mat::t))

    .function("col", select_overload<Mat(int)const>(&cv::Mat::col))

    .function("dot", select_overload<double(InputArray)const>(&cv::Mat::dot))

    .property("rows", &cv::Mat::rows)

    .property("cols", &cv::Mat::cols)

    .function("getData", &cv::Mat::getData , allow_raw_pointers() )

    ;

    emscripten::class_<cv::Subdiv2D>("Subdiv2D")
    
    .function("insert", select_overload<int(Point2f)>(&cv::Subdiv2D::insert))

    .function("insert", select_overload<void(const std::vector<Point2f>&)>(&cv::Subdiv2D::insert))

    .function("edgeOrg", select_overload<int(int, Point2f*)const>(&cv::Subdiv2D::edgeOrg), allow_raw_pointers())

    .function("rotateEdge", select_overload<int(int,int)const>(&cv::Subdiv2D::rotateEdge))

    .function("initDelaunay", select_overload<void(Rect)>(&cv::Subdiv2D::initDelaunay))

    .constructor<  >()

    .constructor< Rect >()

    .function("getEdge", select_overload<int(int,int)const>(&cv::Subdiv2D::getEdge))

    .function("getTriangleList", select_overload<void( std::vector<Vec6f>&)const>(&cv::Subdiv2D::getTriangleList))

    .function("nextEdge", select_overload<int(int)const>(&cv::Subdiv2D::nextEdge))

    .function("edgeDst", select_overload<int(int, Point2f*)const>(&cv::Subdiv2D::edgeDst), allow_raw_pointers())

    .function("getEdgeList", select_overload<void( std::vector<Vec4f>&)const>(&cv::Subdiv2D::getEdgeList))

    .function("getVertex", select_overload<Point2f(int, int*)const>(&cv::Subdiv2D::getVertex), allow_raw_pointers())

    .function("getVoronoiFacetList", select_overload<void(const std::vector<int>&, std::vector<std::vector<Point2f> >&, std::vector<Point2f>&)>(&cv::Subdiv2D::getVoronoiFacetList))

    .function("symEdge", select_overload<int(int)const>(&cv::Subdiv2D::symEdge))

    .function("findNearest", select_overload<int(Point2f, Point2f*)>(&cv::Subdiv2D::findNearest), allow_raw_pointers())

    ;

    emscripten::class_<cv::ml::ANN_MLP>("ml_ANN_MLP")
    
    ;

    emscripten::class_<cv::ml::Boost>("ml_Boost")
    
    ;

    emscripten::class_<cv::ml::DTrees>("ml_DTrees")
    
    ;

    emscripten::class_<cv::ml::EM>("ml_EM")
    
    .function("predict2", select_overload<Vec2d(InputArray,OutputArray)const>(&cv::ml::EM::predict2))

    ;

    emscripten::class_<cv::ml::KNearest>("ml_KNearest")
    
    ;

    emscripten::class_<cv::ml::NormalBayesClassifier>("ml_NormalBayesClassifier")
    
    ;

    emscripten::class_<cv::ml::ParamGrid>("ml_ParamGrid")
    
    .property("minVal", &cv::ml::ParamGrid::minVal)

    .property("maxVal", &cv::ml::ParamGrid::maxVal)

    .property("logStep", &cv::ml::ParamGrid::logStep)

    ;

    emscripten::class_<cv::ml::RTrees>("ml_RTrees")
    
    ;

    emscripten::class_<cv::ml::SVM>("ml_SVM")
    
    .function("getSupportVectors", select_overload<Mat()const>(&cv::ml::SVM::getSupportVectors))

    ;

   
    function("Canny", select_overload<void(InputArray,OutputArray,double,double,int,bool)>(&cv::Canny));

    function("GaussianBlur", select_overload<void(InputArray,OutputArray,Size,double,double,int)>(&cv::GaussianBlur));

    function("HoughCircles", select_overload<void(InputArray,OutputArray,int,double,double,double,double,int,int)>(&cv::HoughCircles));

    function("HoughLines", select_overload<void(InputArray,OutputArray,double,double,int,double,double,double,double)>(&cv::HoughLines));

    function("HoughLinesP", select_overload<void(InputArray,OutputArray,double,double,int,double,double)>(&cv::HoughLinesP));

    function("HuMoments", select_overload<void(const Moments&,OutputArray)>(&cv::HuMoments));

    function("LUT", select_overload<void(InputArray,InputArray,OutputArray)>(&cv::LUT));

    function("Laplacian", select_overload<void(InputArray,OutputArray,int,int,double,double,int)>(&cv::Laplacian));

    function("Mahalanobis", select_overload<double(InputArray,InputArray,InputArray)>(&cv::Mahalanobis));

    function("PCABackProject", select_overload<void(InputArray,InputArray,InputArray,OutputArray)>(&cv::PCABackProject));

    function("PCACompute", select_overload<void(InputArray,InputOutputArray,OutputArray,int)>(&cv::PCACompute));

    function("PCACompute", select_overload<void(InputArray,InputOutputArray,OutputArray,double)>(&cv::PCACompute));

    function("PCAProject", select_overload<void(InputArray,InputArray,InputArray,OutputArray)>(&cv::PCAProject));

    function("PSNR", select_overload<double(InputArray,InputArray)>(&cv::PSNR));

    function("SVBackSubst", select_overload<void(InputArray,InputArray,InputArray,InputArray,OutputArray)>(&cv::SVBackSubst));

    function("SVDecomp", select_overload<void(InputArray,OutputArray,OutputArray,OutputArray,int)>(&cv::SVDecomp));

    function("Scharr", select_overload<void(InputArray,OutputArray,int,int,int,double,double,int)>(&cv::Scharr));

    function("Sobel", select_overload<void(InputArray,OutputArray,int,int,int,int,double,double,int)>(&cv::Sobel));

    function("absdiff", select_overload<void(InputArray,InputArray,OutputArray)>(&cv::absdiff));

    function("accumulate", select_overload<void(InputArray,InputOutputArray,InputArray)>(&cv::accumulate));

    function("accumulateProduct", select_overload<void(InputArray,InputArray,InputOutputArray,InputArray)>(&cv::accumulateProduct));

    function("accumulateSquare", select_overload<void(InputArray,InputOutputArray,InputArray)>(&cv::accumulateSquare));

    function("accumulateWeighted", select_overload<void(InputArray,InputOutputArray,double,InputArray)>(&cv::accumulateWeighted));

    function("adaptiveThreshold", select_overload<void(InputArray,OutputArray,double,int,int,int,double)>(&cv::adaptiveThreshold));

    function("add", select_overload<void(InputArray,InputArray,OutputArray,InputArray,int)>(&cv::add));

    function("addWeighted", select_overload<void(InputArray,double,InputArray,double,double,OutputArray,int)>(&cv::addWeighted));

    function("applyColorMap", select_overload<void(InputArray,OutputArray,int)>(&cv::applyColorMap));

    function("approxPolyDP", select_overload<void(InputArray,OutputArray,double,bool)>(&cv::approxPolyDP));

    function("arcLength", select_overload<double(InputArray,bool)>(&cv::arcLength));

    function("arrowedLine", select_overload<void(InputOutputArray,Point,Point,const Scalar&,int,int,int,double)>(&cv::arrowedLine));

    function("batchDistance", select_overload<void(InputArray,InputArray,OutputArray,int,OutputArray,int,int,InputArray,int,bool)>(&cv::batchDistance));

    function("bilateralFilter", select_overload<void(InputArray,OutputArray,int,double,double,int)>(&cv::bilateralFilter));

    function("bitwise_and", select_overload<void(InputArray,InputArray,OutputArray,InputArray)>(&cv::bitwise_and));

    function("bitwise_not", select_overload<void(InputArray,OutputArray,InputArray)>(&cv::bitwise_not));

    function("bitwise_or", select_overload<void(InputArray,InputArray,OutputArray,InputArray)>(&cv::bitwise_or));

    function("bitwise_xor", select_overload<void(InputArray,InputArray,OutputArray,InputArray)>(&cv::bitwise_xor));

    function("blur", select_overload<void(InputArray,OutputArray,Size,Point,int)>(&cv::blur));

    function("borderInterpolate", select_overload<int(int,int,int)>(&cv::borderInterpolate));

    function("boundingRect", select_overload<Rect(InputArray)>(&cv::boundingRect));

    function("boxFilter", select_overload<void(InputArray,OutputArray,int,Size,Point,bool,int)>(&cv::boxFilter));

    function("boxPoints", select_overload<void(RotatedRect,OutputArray)>(&cv::boxPoints));

    function("calcBackProject", select_overload<void(InputArrayOfArrays,const std::vector<int>&,InputArray,OutputArray,const std::vector<float>&,double)>(&cv::calcBackProject));

    function("calcCovarMatrix", select_overload<void(InputArray,OutputArray,InputOutputArray,int,int)>(&cv::calcCovarMatrix));

    function("calcHist", select_overload<void(InputArrayOfArrays,const std::vector<int>&,InputArray,OutputArray,const std::vector<int>&,const std::vector<float>&,bool)>(&cv::calcHist));

    function("cartToPolar", select_overload<void(InputArray,InputArray,OutputArray,OutputArray,bool)>(&cv::cartToPolar));

    function("checkRange", select_overload<bool(InputArray,bool, Point*,double,double)>(&cv::checkRange), allow_raw_pointers());

    function("circle", select_overload<void(InputOutputArray,Point,int,const Scalar&,int,int,int)>(&cv::circle));

    function("clipLine", select_overload<bool(Rect,  Point&,  Point&)>(&cv::clipLine));

    function("compare", select_overload<void(InputArray,InputArray,OutputArray,int)>(&cv::compare));

    function("compareHist", select_overload<double(InputArray,InputArray,int)>(&cv::compareHist));

    function("completeSymm", select_overload<void(InputOutputArray,bool)>(&cv::completeSymm));

    function("connectedComponents", select_overload<int(InputArray,OutputArray,int,int)>(&cv::connectedComponents));

    function("connectedComponentsWithStats", select_overload<int(InputArray,OutputArray,OutputArray,OutputArray,int,int)>(&cv::connectedComponentsWithStats));

    function("contourArea", select_overload<double(InputArray,bool)>(&cv::contourArea));

    function("convertMaps", select_overload<void(InputArray,InputArray,OutputArray,OutputArray,int,bool)>(&cv::convertMaps));

    function("convertScaleAbs", select_overload<void(InputArray,OutputArray,double,double)>(&cv::convertScaleAbs));

    function("convexHull", select_overload<void(InputArray,OutputArray,bool,bool)>(&cv::convexHull));

    function("convexityDefects", select_overload<void(InputArray,InputArray,OutputArray)>(&cv::convexityDefects));

    function("copyMakeBorder", select_overload<void(InputArray,OutputArray,int,int,int,int,int,const Scalar&)>(&cv::copyMakeBorder));

    function("cornerEigenValsAndVecs", select_overload<void(InputArray,OutputArray,int,int,int)>(&cv::cornerEigenValsAndVecs));

    function("cornerHarris", select_overload<void(InputArray,OutputArray,int,int,double,int)>(&cv::cornerHarris));

    function("cornerMinEigenVal", select_overload<void(InputArray,OutputArray,int,int,int)>(&cv::cornerMinEigenVal));

    function("cornerSubPix", select_overload<void(InputArray,InputOutputArray,Size,Size,TermCriteria)>(&cv::cornerSubPix));

    function("countNonZero", select_overload<int(InputArray)>(&cv::countNonZero));

    function("createCLAHE", select_overload<Ptr<CLAHE>(double,Size)>(&cv::createCLAHE));

    function("createHanningWindow", select_overload<void(OutputArray,Size,int)>(&cv::createHanningWindow));

    function("createLineSegmentDetector", select_overload<Ptr<LineSegmentDetector>(int,double,double,double,double,double,double,int)>(&cv::createLineSegmentDetector));

    function("cvtColor", select_overload<void(InputArray,OutputArray,int,int)>(&cv::cvtColor));

    function("dct", select_overload<void(InputArray,OutputArray,int)>(&cv::dct));

    function("demosaicing", select_overload<void(InputArray,OutputArray,int,int)>(&cv::demosaicing));

    function("destroyAllWindows", select_overload<void()>(&cv::destroyAllWindows));

    function("destroyWindow", select_overload<void(const String&)>(&cv::destroyWindow));

    function("determinant", select_overload<double(InputArray)>(&cv::determinant));

    function("dft", select_overload<void(InputArray,OutputArray,int,int)>(&cv::dft));

    function("dilate", select_overload<void(InputArray,OutputArray,InputArray,Point,int,int,const Scalar&)>(&cv::dilate));

    function("distanceTransform", select_overload<void(InputArray,OutputArray,int,int,int)>(&cv::distanceTransform));

    function("distanceTransformWithLabels", select_overload<void(InputArray,OutputArray,OutputArray,int,int,int)>(&cv::distanceTransform));

    function("divide", select_overload<void(InputArray,InputArray,OutputArray,double,int)>(&cv::divide));

    function("divide", select_overload<void(double,InputArray,OutputArray,int)>(&cv::divide));

    function("drawContours", select_overload<void(InputOutputArray,InputArrayOfArrays,int,const Scalar&,int,int,InputArray,int,Point)>(&cv::drawContours));

    function("eigen", select_overload<bool(InputArray,OutputArray,OutputArray)>(&cv::eigen));

    function("ellipse", select_overload<void(InputOutputArray,Point,Size,double,double,double,const Scalar&,int,int,int)>(&cv::ellipse));

    function("ellipse", select_overload<void(InputOutputArray,const RotatedRect&,const Scalar&,int,int)>(&cv::ellipse));

    function("ellipse2Poly", select_overload<void(Point,Size,int,int,int,int, std::vector<Point>&)>(&cv::ellipse2Poly));

    function("equalizeHist", select_overload<void(InputArray,OutputArray)>(&cv::equalizeHist));

    function("erode", select_overload<void(InputArray,OutputArray,InputArray,Point,int,int,const Scalar&)>(&cv::erode));

    function("exp", select_overload<void(InputArray,OutputArray)>(&cv::exp));

    function("extractChannel", select_overload<void(InputArray,OutputArray,int)>(&cv::extractChannel));

    function("fillConvexPoly", select_overload<void(InputOutputArray,InputArray,const Scalar&,int,int)>(&cv::fillConvexPoly));

    function("fillPoly", select_overload<void(InputOutputArray,InputArrayOfArrays,const Scalar&,int,int,Point)>(&cv::fillPoly));

    function("filter2D", select_overload<void(InputArray,OutputArray,int,InputArray,Point,double,int)>(&cv::filter2D));

    function("findContours", select_overload<void(InputOutputArray,OutputArrayOfArrays,OutputArray,int,int,Point)>(&cv::findContours));

    function("findNonZero", select_overload<void(InputArray,OutputArray)>(&cv::findNonZero));

    function("fitEllipse", select_overload<RotatedRect(InputArray)>(&cv::fitEllipse));

    function("fitLine", select_overload<void(InputArray,OutputArray,int,double,double,double)>(&cv::fitLine));

    function("flip", select_overload<void(InputArray,OutputArray,int)>(&cv::flip));

    function("floodFill", select_overload<int(InputOutputArray,InputOutputArray,Point,Scalar, Rect*,Scalar,Scalar,int)>(&cv::floodFill), allow_raw_pointers());

    function("gemm", select_overload<void(InputArray,InputArray,double,InputArray,double,OutputArray,int)>(&cv::gemm));

    function("getAffineTransform", select_overload<Mat(InputArray,InputArray)>(&cv::getAffineTransform));

    function("getDefaultNewCameraMatrix", select_overload<Mat(InputArray,Size,bool)>(&cv::getDefaultNewCameraMatrix));

    function("getDerivKernels", select_overload<void(OutputArray,OutputArray,int,int,int,bool,int)>(&cv::getDerivKernels));

    function("getGaborKernel", select_overload<Mat(Size,double,double,double,double,double,int)>(&cv::getGaborKernel));

    function("getGaussianKernel", select_overload<Mat(int,double,int)>(&cv::getGaussianKernel));

    function("getOptimalDFTSize", select_overload<int(int)>(&cv::getOptimalDFTSize));

    function("getPerspectiveTransform", select_overload<Mat(InputArray,InputArray)>(&cv::getPerspectiveTransform));

    function("getRectSubPix", select_overload<void(InputArray,Size,Point2f,OutputArray,int)>(&cv::getRectSubPix));

    function("getRotationMatrix2D", select_overload<Mat(Point2f,double,double)>(&cv::getRotationMatrix2D));

    function("getStructuringElement", select_overload<Mat(int,Size,Point)>(&cv::getStructuringElement));

    function("getTextSize", select_overload<Size(const String&,int,double,int, int*)>(&cv::getTextSize), allow_raw_pointers());

    function("getTrackbarPos", select_overload<int(const String&,const String&)>(&cv::getTrackbarPos));

    function("getWindowProperty", select_overload<double(const String&,int)>(&cv::getWindowProperty));

    function("goodFeaturesToTrack", select_overload<void(InputArray,OutputArray,int,double,double,InputArray,int,bool,double)>(&cv::goodFeaturesToTrack));

    function("grabCut", select_overload<void(InputArray,InputOutputArray,Rect,InputOutputArray,InputOutputArray,int,int)>(&cv::grabCut));

    function("hconcat", select_overload<void(InputArrayOfArrays,OutputArray)>(&cv::hconcat));

    function("idct", select_overload<void(InputArray,OutputArray,int)>(&cv::idct));

    function("idft", select_overload<void(InputArray,OutputArray,int,int)>(&cv::idft));

    function("imdecode", select_overload<Mat(InputArray,int)>(&cv::imdecode));

    function("imencode", select_overload<bool(const String&,InputArray, std::vector<uchar>&,const std::vector<int>&)>(&cv::imencode));

    function("imread", select_overload<Mat(const String&,int)>(&cv::imread));

    function("imreadmulti", select_overload<bool(const String&,std::vector<Mat>&,int)>(&cv::imreadmulti));

    function("imshow", select_overload<void(const String&,InputArray)>(&cv::imshow));

    function("imwrite", select_overload<bool(const String&,InputArray,const std::vector<int>&)>(&cv::imwrite));

    function("inRange", select_overload<void(InputArray,InputArray,InputArray,OutputArray)>(&cv::inRange));

    function("initUndistortRectifyMap", select_overload<void(InputArray,InputArray,InputArray,InputArray,Size,int,OutputArray,OutputArray)>(&cv::initUndistortRectifyMap));

    function("initWideAngleProjMap", select_overload<float(InputArray,InputArray,Size,int,int,OutputArray,OutputArray,int,double)>(&cv::initWideAngleProjMap));

    function("insertChannel", select_overload<void(InputArray,InputOutputArray,int)>(&cv::insertChannel));

    function("integral", select_overload<void(InputArray,OutputArray,int)>(&cv::integral));

    function("integral2", select_overload<void(InputArray,OutputArray,OutputArray,int,int)>(&cv::integral));

    function("integral3", select_overload<void(InputArray,OutputArray,OutputArray,OutputArray,int,int)>(&cv::integral));

    function("intersectConvexConvex", select_overload<float(InputArray,InputArray,OutputArray,bool)>(&cv::intersectConvexConvex));

    function("invert", select_overload<double(InputArray,OutputArray,int)>(&cv::invert));

    function("invertAffineTransform", select_overload<void(InputArray,OutputArray)>(&cv::invertAffineTransform));

    function("isContourConvex", select_overload<bool(InputArray)>(&cv::isContourConvex));

    function("kmeans", select_overload<double(InputArray,int,InputOutputArray,TermCriteria,int,int,OutputArray)>(&cv::kmeans));

    function("line", select_overload<void(InputOutputArray,Point,Point,const Scalar&,int,int,int)>(&cv::line));

    function("linearPolar", select_overload<void(InputArray,OutputArray,Point2f,double,int)>(&cv::linearPolar));

    function("log", select_overload<void(InputArray,OutputArray)>(&cv::log));

    function("logPolar", select_overload<void(InputArray,OutputArray,Point2f,double,int)>(&cv::logPolar));

    function("magnitude", select_overload<void(InputArray,InputArray,OutputArray)>(&cv::magnitude));

    function("matchShapes", select_overload<double(InputArray,InputArray,int,double)>(&cv::matchShapes));

    function("matchTemplate", select_overload<void(InputArray,InputArray,OutputArray,int,InputArray)>(&cv::matchTemplate));

    function("max", select_overload<void(InputArray,InputArray,OutputArray)>(&cv::max));

    function("mean", select_overload<Scalar(InputArray,InputArray)>(&cv::mean));

    function("meanStdDev", select_overload<void(InputArray,OutputArray,OutputArray,InputArray)>(&cv::meanStdDev));

    function("medianBlur", select_overload<void(InputArray,OutputArray,int)>(&cv::medianBlur));

    function("merge", select_overload<void(InputArrayOfArrays,OutputArray)>(&cv::merge));

    function("min", select_overload<void(InputArray,InputArray,OutputArray)>(&cv::min));

    function("minAreaRect", select_overload<RotatedRect(InputArray)>(&cv::minAreaRect));

    function("minEnclosingTriangle", select_overload<double(InputArray, OutputArray)>(&cv::minEnclosingTriangle));

    function("minMaxLoc", select_overload<void(InputArray, double*, double*, Point*, Point*,InputArray)>(&cv::minMaxLoc), allow_raw_pointers());

    function("mixChannels", select_overload<void(InputArrayOfArrays,InputOutputArrayOfArrays,const std::vector<int>&)>(&cv::mixChannels));

    function("moments", select_overload<Moments(InputArray,bool)>(&cv::moments));

    function("morphologyEx", select_overload<void(InputArray,OutputArray,int,InputArray,Point,int,int,const Scalar&)>(&cv::morphologyEx));

    function("moveWindow", select_overload<void(const String&,int,int)>(&cv::moveWindow));

    function("mulSpectrums", select_overload<void(InputArray,InputArray,OutputArray,int,bool)>(&cv::mulSpectrums));

    function("mulTransposed", select_overload<void(InputArray,OutputArray,bool,InputArray,double,int)>(&cv::mulTransposed));

    function("multiply", select_overload<void(InputArray,InputArray,OutputArray,double,int)>(&cv::multiply));

    function("namedWindow", select_overload<void(const String&,int)>(&cv::namedWindow));

    function("norm", select_overload<double(InputArray,int,InputArray)>(&cv::norm));

    function("norm", select_overload<double(InputArray,InputArray,int,InputArray)>(&cv::norm));

    function("normalize", select_overload<void(InputArray,InputOutputArray,double,double,int,int,InputArray)>(&cv::normalize));

    function("patchNaNs", select_overload<void(InputOutputArray,double)>(&cv::patchNaNs));

    function("perspectiveTransform", select_overload<void(InputArray,OutputArray,InputArray)>(&cv::perspectiveTransform));

    function("phase", select_overload<void(InputArray,InputArray,OutputArray,bool)>(&cv::phase));

    function("phaseCorrelate", select_overload<Point2d(InputArray,InputArray,InputArray, double*)>(&cv::phaseCorrelate), allow_raw_pointers());

    function("pointPolygonTest", select_overload<double(InputArray,Point2f,bool)>(&cv::pointPolygonTest));

    function("polarToCart", select_overload<void(InputArray,InputArray,OutputArray,OutputArray,bool)>(&cv::polarToCart));

    function("polylines", select_overload<void(InputOutputArray,InputArrayOfArrays,bool,const Scalar&,int,int,int)>(&cv::polylines));

    function("pow", select_overload<void(InputArray,double,OutputArray)>(&cv::pow));

    function("preCornerDetect", select_overload<void(InputArray,OutputArray,int,int)>(&cv::preCornerDetect));

    function("putText", select_overload<void(InputOutputArray,const String&,Point,int,double,Scalar,int,int,bool)>(&cv::putText));

    function("pyrDown", select_overload<void(InputArray,OutputArray,const Size&,int)>(&cv::pyrDown));

    function("pyrMeanShiftFiltering", select_overload<void(InputArray,OutputArray,double,double,int,TermCriteria)>(&cv::pyrMeanShiftFiltering));

    function("pyrUp", select_overload<void(InputArray,OutputArray,const Size&,int)>(&cv::pyrUp));

    function("randShuffle", select_overload<void(InputOutputArray,double,RNG*)>(&cv::randShuffle), allow_raw_pointers());

    function("randn", select_overload<void(InputOutputArray,InputArray,InputArray)>(&cv::randn));

    function("randu", select_overload<void(InputOutputArray,InputArray,InputArray)>(&cv::randu));

    function("rectangle", select_overload<void(InputOutputArray,Point,Point,const Scalar&,int,int,int)>(&cv::rectangle));

    function("reduce", select_overload<void(InputArray,OutputArray,int,int,int)>(&cv::reduce));

    function("remap", select_overload<void(InputArray,OutputArray,InputArray,InputArray,int,int,const Scalar&)>(&cv::remap));

    function("repeat", select_overload<void(InputArray,int,int,OutputArray)>(&cv::repeat));

    function("resize", select_overload<void(InputArray,OutputArray,Size,double,double,int)>(&cv::resize));

    function("resizeWindow", select_overload<void(const String&,int,int)>(&cv::resizeWindow));

    function("rotatedRectangleIntersection", select_overload<int(const RotatedRect&,const RotatedRect&,OutputArray)>(&cv::rotatedRectangleIntersection));

    function("scaleAdd", select_overload<void(InputArray,double,InputArray,OutputArray)>(&cv::scaleAdd));

    function("sepFilter2D", select_overload<void(InputArray,OutputArray,int,InputArray,InputArray,Point,double,int)>(&cv::sepFilter2D));

    function("setIdentity", select_overload<void(InputOutputArray,const Scalar&)>(&cv::setIdentity));

    function("setTrackbarMax", select_overload<void(const String&,const String&,int)>(&cv::setTrackbarMax));

    function("setTrackbarPos", select_overload<void(const String&,const String&,int)>(&cv::setTrackbarPos));

    function("setWindowProperty", select_overload<void(const String&,int,double)>(&cv::setWindowProperty));

    function("setWindowTitle", select_overload<void(const String&,const String&)>(&cv::setWindowTitle));

    function("solve", select_overload<bool(InputArray,InputArray,OutputArray,int)>(&cv::solve));

    function("solveCubic", select_overload<int(InputArray,OutputArray)>(&cv::solveCubic));

    function("solvePoly", select_overload<double(InputArray,OutputArray,int)>(&cv::solvePoly));

    function("sort", select_overload<void(InputArray,OutputArray,int)>(&cv::sort));

    function("sortIdx", select_overload<void(InputArray,OutputArray,int)>(&cv::sortIdx));

    function("split", select_overload<void(InputArray,OutputArrayOfArrays)>(&cv::split));

    function("sqrBoxFilter", select_overload<void(InputArray,OutputArray,int,Size,Point,bool,int)>(&cv::sqrBoxFilter));

    function("sqrt", select_overload<void(InputArray,OutputArray)>(&cv::sqrt));

    function("startWindowThread", select_overload<int()>(&cv::startWindowThread));

    function("subtract", select_overload<void(InputArray,InputArray,OutputArray,InputArray,int)>(&cv::subtract));

    function("sumElems", select_overload<Scalar(InputArray)>(&cv::sum));

    function("threshold", select_overload<double(InputArray,OutputArray,double,double,int)>(&cv::threshold));

    function("trace", select_overload<Scalar(InputArray)>(&cv::trace));

    function("transform", select_overload<void(InputArray,OutputArray,InputArray)>(&cv::transform));

    function("transpose", select_overload<void(InputArray,OutputArray)>(&cv::transpose));

    function("undistort", select_overload<void(InputArray,OutputArray,InputArray,InputArray,InputArray)>(&cv::undistort));

    function("undistortPoints", select_overload<void(InputArray,OutputArray,InputArray,InputArray,InputArray,InputArray)>(&cv::undistortPoints));

    function("vconcat", select_overload<void(InputArrayOfArrays,OutputArray)>(&cv::vconcat));

    function("waitKey", select_overload<int(int)>(&cv::waitKey));

    function("warpAffine", select_overload<void(InputArray,OutputArray,InputArray,Size,int,int,const Scalar&)>(&cv::warpAffine));

    function("warpPerspective", select_overload<void(InputArray,OutputArray,InputArray,Size,int,int,const Scalar&)>(&cv::warpPerspective));

    function("watershed", select_overload<void(InputArray,InputOutputArray)>(&cv::watershed));

    function("finish", select_overload<void()>(&cv::ocl::finish));

    function("haveAmdBlas", select_overload<bool()>(&cv::ocl::haveAmdBlas));

    function("haveAmdFft", select_overload<bool()>(&cv::ocl::haveAmdFft));

    function("haveOpenCL", select_overload<bool()>(&cv::ocl::haveOpenCL));

    function("setUseOpenCL", select_overload<void(bool)>(&cv::ocl::setUseOpenCL));

    function("useOpenCL", select_overload<bool()>(&cv::ocl::useOpenCL));

    emscripten::enum_<cv::AdaptiveThresholdTypes>("cv_AdaptiveThresholdTypes")
    
    .value("ADAPTIVE_THRESH_MEAN_C",cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C)

    .value("ADAPTIVE_THRESH_GAUSSIAN_C",cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C)

    ;

    emscripten::enum_<cv::ColorConversionCodes>("cv_ColorConversionCodes")
    
    .value("COLOR_BGR2BGRA",cv::ColorConversionCodes::COLOR_BGR2BGRA)

    .value("COLOR_RGB2RGBA",cv::ColorConversionCodes::COLOR_RGB2RGBA)

    .value("COLOR_BGRA2BGR",cv::ColorConversionCodes::COLOR_BGRA2BGR)

    .value("COLOR_RGBA2RGB",cv::ColorConversionCodes::COLOR_RGBA2RGB)

    .value("COLOR_BGR2RGBA",cv::ColorConversionCodes::COLOR_BGR2RGBA)

    .value("COLOR_RGB2BGRA",cv::ColorConversionCodes::COLOR_RGB2BGRA)

    .value("COLOR_RGBA2BGR",cv::ColorConversionCodes::COLOR_RGBA2BGR)

    .value("COLOR_BGRA2RGB",cv::ColorConversionCodes::COLOR_BGRA2RGB)

    .value("COLOR_BGR2RGB",cv::ColorConversionCodes::COLOR_BGR2RGB)

    .value("COLOR_RGB2BGR",cv::ColorConversionCodes::COLOR_RGB2BGR)

    .value("COLOR_BGRA2RGBA",cv::ColorConversionCodes::COLOR_BGRA2RGBA)

    .value("COLOR_RGBA2BGRA",cv::ColorConversionCodes::COLOR_RGBA2BGRA)

    .value("COLOR_BGR2GRAY",cv::ColorConversionCodes::COLOR_BGR2GRAY)

    .value("COLOR_RGB2GRAY",cv::ColorConversionCodes::COLOR_RGB2GRAY)

    .value("COLOR_GRAY2BGR",cv::ColorConversionCodes::COLOR_GRAY2BGR)

    .value("COLOR_GRAY2RGB",cv::ColorConversionCodes::COLOR_GRAY2RGB)

    .value("COLOR_GRAY2BGRA",cv::ColorConversionCodes::COLOR_GRAY2BGRA)

    .value("COLOR_GRAY2RGBA",cv::ColorConversionCodes::COLOR_GRAY2RGBA)

    .value("COLOR_BGRA2GRAY",cv::ColorConversionCodes::COLOR_BGRA2GRAY)

    .value("COLOR_RGBA2GRAY",cv::ColorConversionCodes::COLOR_RGBA2GRAY)

    .value("COLOR_BGR2BGR565",cv::ColorConversionCodes::COLOR_BGR2BGR565)

    .value("COLOR_RGB2BGR565",cv::ColorConversionCodes::COLOR_RGB2BGR565)

    .value("COLOR_BGR5652BGR",cv::ColorConversionCodes::COLOR_BGR5652BGR)

    .value("COLOR_BGR5652RGB",cv::ColorConversionCodes::COLOR_BGR5652RGB)

    .value("COLOR_BGRA2BGR565",cv::ColorConversionCodes::COLOR_BGRA2BGR565)

    .value("COLOR_RGBA2BGR565",cv::ColorConversionCodes::COLOR_RGBA2BGR565)

    .value("COLOR_BGR5652BGRA",cv::ColorConversionCodes::COLOR_BGR5652BGRA)

    .value("COLOR_BGR5652RGBA",cv::ColorConversionCodes::COLOR_BGR5652RGBA)

    .value("COLOR_GRAY2BGR565",cv::ColorConversionCodes::COLOR_GRAY2BGR565)

    .value("COLOR_BGR5652GRAY",cv::ColorConversionCodes::COLOR_BGR5652GRAY)

    .value("COLOR_BGR2BGR555",cv::ColorConversionCodes::COLOR_BGR2BGR555)

    .value("COLOR_RGB2BGR555",cv::ColorConversionCodes::COLOR_RGB2BGR555)

    .value("COLOR_BGR5552BGR",cv::ColorConversionCodes::COLOR_BGR5552BGR)

    .value("COLOR_BGR5552RGB",cv::ColorConversionCodes::COLOR_BGR5552RGB)

    .value("COLOR_BGRA2BGR555",cv::ColorConversionCodes::COLOR_BGRA2BGR555)

    .value("COLOR_RGBA2BGR555",cv::ColorConversionCodes::COLOR_RGBA2BGR555)

    .value("COLOR_BGR5552BGRA",cv::ColorConversionCodes::COLOR_BGR5552BGRA)

    .value("COLOR_BGR5552RGBA",cv::ColorConversionCodes::COLOR_BGR5552RGBA)

    .value("COLOR_GRAY2BGR555",cv::ColorConversionCodes::COLOR_GRAY2BGR555)

    .value("COLOR_BGR5552GRAY",cv::ColorConversionCodes::COLOR_BGR5552GRAY)

    .value("COLOR_BGR2XYZ",cv::ColorConversionCodes::COLOR_BGR2XYZ)

    .value("COLOR_RGB2XYZ",cv::ColorConversionCodes::COLOR_RGB2XYZ)

    .value("COLOR_XYZ2BGR",cv::ColorConversionCodes::COLOR_XYZ2BGR)

    .value("COLOR_XYZ2RGB",cv::ColorConversionCodes::COLOR_XYZ2RGB)

    .value("COLOR_BGR2YCrCb",cv::ColorConversionCodes::COLOR_BGR2YCrCb)

    .value("COLOR_RGB2YCrCb",cv::ColorConversionCodes::COLOR_RGB2YCrCb)

    .value("COLOR_YCrCb2BGR",cv::ColorConversionCodes::COLOR_YCrCb2BGR)

    .value("COLOR_YCrCb2RGB",cv::ColorConversionCodes::COLOR_YCrCb2RGB)

    .value("COLOR_BGR2HSV",cv::ColorConversionCodes::COLOR_BGR2HSV)

    .value("COLOR_RGB2HSV",cv::ColorConversionCodes::COLOR_RGB2HSV)

    .value("COLOR_BGR2Lab",cv::ColorConversionCodes::COLOR_BGR2Lab)

    .value("COLOR_RGB2Lab",cv::ColorConversionCodes::COLOR_RGB2Lab)

    .value("COLOR_BGR2Luv",cv::ColorConversionCodes::COLOR_BGR2Luv)

    .value("COLOR_RGB2Luv",cv::ColorConversionCodes::COLOR_RGB2Luv)

    .value("COLOR_BGR2HLS",cv::ColorConversionCodes::COLOR_BGR2HLS)

    .value("COLOR_RGB2HLS",cv::ColorConversionCodes::COLOR_RGB2HLS)

    .value("COLOR_HSV2BGR",cv::ColorConversionCodes::COLOR_HSV2BGR)

    .value("COLOR_HSV2RGB",cv::ColorConversionCodes::COLOR_HSV2RGB)

    .value("COLOR_Lab2BGR",cv::ColorConversionCodes::COLOR_Lab2BGR)

    .value("COLOR_Lab2RGB",cv::ColorConversionCodes::COLOR_Lab2RGB)

    .value("COLOR_Luv2BGR",cv::ColorConversionCodes::COLOR_Luv2BGR)

    .value("COLOR_Luv2RGB",cv::ColorConversionCodes::COLOR_Luv2RGB)

    .value("COLOR_HLS2BGR",cv::ColorConversionCodes::COLOR_HLS2BGR)

    .value("COLOR_HLS2RGB",cv::ColorConversionCodes::COLOR_HLS2RGB)

    .value("COLOR_BGR2HSV_FULL",cv::ColorConversionCodes::COLOR_BGR2HSV_FULL)

    .value("COLOR_RGB2HSV_FULL",cv::ColorConversionCodes::COLOR_RGB2HSV_FULL)

    .value("COLOR_BGR2HLS_FULL",cv::ColorConversionCodes::COLOR_BGR2HLS_FULL)

    .value("COLOR_RGB2HLS_FULL",cv::ColorConversionCodes::COLOR_RGB2HLS_FULL)

    .value("COLOR_HSV2BGR_FULL",cv::ColorConversionCodes::COLOR_HSV2BGR_FULL)

    .value("COLOR_HSV2RGB_FULL",cv::ColorConversionCodes::COLOR_HSV2RGB_FULL)

    .value("COLOR_HLS2BGR_FULL",cv::ColorConversionCodes::COLOR_HLS2BGR_FULL)

    .value("COLOR_HLS2RGB_FULL",cv::ColorConversionCodes::COLOR_HLS2RGB_FULL)

    .value("COLOR_LBGR2Lab",cv::ColorConversionCodes::COLOR_LBGR2Lab)

    .value("COLOR_LRGB2Lab",cv::ColorConversionCodes::COLOR_LRGB2Lab)

    .value("COLOR_LBGR2Luv",cv::ColorConversionCodes::COLOR_LBGR2Luv)

    .value("COLOR_LRGB2Luv",cv::ColorConversionCodes::COLOR_LRGB2Luv)

    .value("COLOR_Lab2LBGR",cv::ColorConversionCodes::COLOR_Lab2LBGR)

    .value("COLOR_Lab2LRGB",cv::ColorConversionCodes::COLOR_Lab2LRGB)

    .value("COLOR_Luv2LBGR",cv::ColorConversionCodes::COLOR_Luv2LBGR)

    .value("COLOR_Luv2LRGB",cv::ColorConversionCodes::COLOR_Luv2LRGB)

    .value("COLOR_BGR2YUV",cv::ColorConversionCodes::COLOR_BGR2YUV)

    .value("COLOR_RGB2YUV",cv::ColorConversionCodes::COLOR_RGB2YUV)

    .value("COLOR_YUV2BGR",cv::ColorConversionCodes::COLOR_YUV2BGR)

    .value("COLOR_YUV2RGB",cv::ColorConversionCodes::COLOR_YUV2RGB)

    .value("COLOR_YUV2RGB_NV12",cv::ColorConversionCodes::COLOR_YUV2RGB_NV12)

    .value("COLOR_YUV2BGR_NV12",cv::ColorConversionCodes::COLOR_YUV2BGR_NV12)

    .value("COLOR_YUV2RGB_NV21",cv::ColorConversionCodes::COLOR_YUV2RGB_NV21)

    .value("COLOR_YUV2BGR_NV21",cv::ColorConversionCodes::COLOR_YUV2BGR_NV21)

    .value("COLOR_YUV420sp2RGB",cv::ColorConversionCodes::COLOR_YUV420sp2RGB)

    .value("COLOR_YUV420sp2BGR",cv::ColorConversionCodes::COLOR_YUV420sp2BGR)

    .value("COLOR_YUV2RGBA_NV12",cv::ColorConversionCodes::COLOR_YUV2RGBA_NV12)

    .value("COLOR_YUV2BGRA_NV12",cv::ColorConversionCodes::COLOR_YUV2BGRA_NV12)

    .value("COLOR_YUV2RGBA_NV21",cv::ColorConversionCodes::COLOR_YUV2RGBA_NV21)

    .value("COLOR_YUV2BGRA_NV21",cv::ColorConversionCodes::COLOR_YUV2BGRA_NV21)

    .value("COLOR_YUV420sp2RGBA",cv::ColorConversionCodes::COLOR_YUV420sp2RGBA)

    .value("COLOR_YUV420sp2BGRA",cv::ColorConversionCodes::COLOR_YUV420sp2BGRA)

    .value("COLOR_YUV2RGB_YV12",cv::ColorConversionCodes::COLOR_YUV2RGB_YV12)

    .value("COLOR_YUV2BGR_YV12",cv::ColorConversionCodes::COLOR_YUV2BGR_YV12)

    .value("COLOR_YUV2RGB_IYUV",cv::ColorConversionCodes::COLOR_YUV2RGB_IYUV)

    .value("COLOR_YUV2BGR_IYUV",cv::ColorConversionCodes::COLOR_YUV2BGR_IYUV)

    .value("COLOR_YUV2RGB_I420",cv::ColorConversionCodes::COLOR_YUV2RGB_I420)

    .value("COLOR_YUV2BGR_I420",cv::ColorConversionCodes::COLOR_YUV2BGR_I420)

    .value("COLOR_YUV420p2RGB",cv::ColorConversionCodes::COLOR_YUV420p2RGB)

    .value("COLOR_YUV420p2BGR",cv::ColorConversionCodes::COLOR_YUV420p2BGR)

    .value("COLOR_YUV2RGBA_YV12",cv::ColorConversionCodes::COLOR_YUV2RGBA_YV12)

    .value("COLOR_YUV2BGRA_YV12",cv::ColorConversionCodes::COLOR_YUV2BGRA_YV12)

    .value("COLOR_YUV2RGBA_IYUV",cv::ColorConversionCodes::COLOR_YUV2RGBA_IYUV)

    .value("COLOR_YUV2BGRA_IYUV",cv::ColorConversionCodes::COLOR_YUV2BGRA_IYUV)

    .value("COLOR_YUV2RGBA_I420",cv::ColorConversionCodes::COLOR_YUV2RGBA_I420)

    .value("COLOR_YUV2BGRA_I420",cv::ColorConversionCodes::COLOR_YUV2BGRA_I420)

    .value("COLOR_YUV420p2RGBA",cv::ColorConversionCodes::COLOR_YUV420p2RGBA)

    .value("COLOR_YUV420p2BGRA",cv::ColorConversionCodes::COLOR_YUV420p2BGRA)

    .value("COLOR_YUV2GRAY_420",cv::ColorConversionCodes::COLOR_YUV2GRAY_420)

    .value("COLOR_YUV2GRAY_NV21",cv::ColorConversionCodes::COLOR_YUV2GRAY_NV21)

    .value("COLOR_YUV2GRAY_NV12",cv::ColorConversionCodes::COLOR_YUV2GRAY_NV12)

    .value("COLOR_YUV2GRAY_YV12",cv::ColorConversionCodes::COLOR_YUV2GRAY_YV12)

    .value("COLOR_YUV2GRAY_IYUV",cv::ColorConversionCodes::COLOR_YUV2GRAY_IYUV)

    .value("COLOR_YUV2GRAY_I420",cv::ColorConversionCodes::COLOR_YUV2GRAY_I420)

    .value("COLOR_YUV420sp2GRAY",cv::ColorConversionCodes::COLOR_YUV420sp2GRAY)

    .value("COLOR_YUV420p2GRAY",cv::ColorConversionCodes::COLOR_YUV420p2GRAY)

    .value("COLOR_YUV2RGB_UYVY",cv::ColorConversionCodes::COLOR_YUV2RGB_UYVY)

    .value("COLOR_YUV2BGR_UYVY",cv::ColorConversionCodes::COLOR_YUV2BGR_UYVY)

    .value("COLOR_YUV2RGB_Y422",cv::ColorConversionCodes::COLOR_YUV2RGB_Y422)

    .value("COLOR_YUV2BGR_Y422",cv::ColorConversionCodes::COLOR_YUV2BGR_Y422)

    .value("COLOR_YUV2RGB_UYNV",cv::ColorConversionCodes::COLOR_YUV2RGB_UYNV)

    .value("COLOR_YUV2BGR_UYNV",cv::ColorConversionCodes::COLOR_YUV2BGR_UYNV)

    .value("COLOR_YUV2RGBA_UYVY",cv::ColorConversionCodes::COLOR_YUV2RGBA_UYVY)

    .value("COLOR_YUV2BGRA_UYVY",cv::ColorConversionCodes::COLOR_YUV2BGRA_UYVY)

    .value("COLOR_YUV2RGBA_Y422",cv::ColorConversionCodes::COLOR_YUV2RGBA_Y422)

    .value("COLOR_YUV2BGRA_Y422",cv::ColorConversionCodes::COLOR_YUV2BGRA_Y422)

    .value("COLOR_YUV2RGBA_UYNV",cv::ColorConversionCodes::COLOR_YUV2RGBA_UYNV)

    .value("COLOR_YUV2BGRA_UYNV",cv::ColorConversionCodes::COLOR_YUV2BGRA_UYNV)

    .value("COLOR_YUV2RGB_YUY2",cv::ColorConversionCodes::COLOR_YUV2RGB_YUY2)

    .value("COLOR_YUV2BGR_YUY2",cv::ColorConversionCodes::COLOR_YUV2BGR_YUY2)

    .value("COLOR_YUV2RGB_YVYU",cv::ColorConversionCodes::COLOR_YUV2RGB_YVYU)

    .value("COLOR_YUV2BGR_YVYU",cv::ColorConversionCodes::COLOR_YUV2BGR_YVYU)

    .value("COLOR_YUV2RGB_YUYV",cv::ColorConversionCodes::COLOR_YUV2RGB_YUYV)

    .value("COLOR_YUV2BGR_YUYV",cv::ColorConversionCodes::COLOR_YUV2BGR_YUYV)

    .value("COLOR_YUV2RGB_YUNV",cv::ColorConversionCodes::COLOR_YUV2RGB_YUNV)

    .value("COLOR_YUV2BGR_YUNV",cv::ColorConversionCodes::COLOR_YUV2BGR_YUNV)

    .value("COLOR_YUV2RGBA_YUY2",cv::ColorConversionCodes::COLOR_YUV2RGBA_YUY2)

    .value("COLOR_YUV2BGRA_YUY2",cv::ColorConversionCodes::COLOR_YUV2BGRA_YUY2)

    .value("COLOR_YUV2RGBA_YVYU",cv::ColorConversionCodes::COLOR_YUV2RGBA_YVYU)

    .value("COLOR_YUV2BGRA_YVYU",cv::ColorConversionCodes::COLOR_YUV2BGRA_YVYU)

    .value("COLOR_YUV2RGBA_YUYV",cv::ColorConversionCodes::COLOR_YUV2RGBA_YUYV)

    .value("COLOR_YUV2BGRA_YUYV",cv::ColorConversionCodes::COLOR_YUV2BGRA_YUYV)

    .value("COLOR_YUV2RGBA_YUNV",cv::ColorConversionCodes::COLOR_YUV2RGBA_YUNV)

    .value("COLOR_YUV2BGRA_YUNV",cv::ColorConversionCodes::COLOR_YUV2BGRA_YUNV)

    .value("COLOR_YUV2GRAY_UYVY",cv::ColorConversionCodes::COLOR_YUV2GRAY_UYVY)

    .value("COLOR_YUV2GRAY_YUY2",cv::ColorConversionCodes::COLOR_YUV2GRAY_YUY2)

    .value("COLOR_YUV2GRAY_Y422",cv::ColorConversionCodes::COLOR_YUV2GRAY_Y422)

    .value("COLOR_YUV2GRAY_UYNV",cv::ColorConversionCodes::COLOR_YUV2GRAY_UYNV)

    .value("COLOR_YUV2GRAY_YVYU",cv::ColorConversionCodes::COLOR_YUV2GRAY_YVYU)

    .value("COLOR_YUV2GRAY_YUYV",cv::ColorConversionCodes::COLOR_YUV2GRAY_YUYV)

    .value("COLOR_YUV2GRAY_YUNV",cv::ColorConversionCodes::COLOR_YUV2GRAY_YUNV)

    .value("COLOR_RGBA2mRGBA",cv::ColorConversionCodes::COLOR_RGBA2mRGBA)

    .value("COLOR_mRGBA2RGBA",cv::ColorConversionCodes::COLOR_mRGBA2RGBA)

    .value("COLOR_RGB2YUV_I420",cv::ColorConversionCodes::COLOR_RGB2YUV_I420)

    .value("COLOR_BGR2YUV_I420",cv::ColorConversionCodes::COLOR_BGR2YUV_I420)

    .value("COLOR_RGB2YUV_IYUV",cv::ColorConversionCodes::COLOR_RGB2YUV_IYUV)

    .value("COLOR_BGR2YUV_IYUV",cv::ColorConversionCodes::COLOR_BGR2YUV_IYUV)

    .value("COLOR_RGBA2YUV_I420",cv::ColorConversionCodes::COLOR_RGBA2YUV_I420)

    .value("COLOR_BGRA2YUV_I420",cv::ColorConversionCodes::COLOR_BGRA2YUV_I420)

    .value("COLOR_RGBA2YUV_IYUV",cv::ColorConversionCodes::COLOR_RGBA2YUV_IYUV)

    .value("COLOR_BGRA2YUV_IYUV",cv::ColorConversionCodes::COLOR_BGRA2YUV_IYUV)

    .value("COLOR_RGB2YUV_YV12",cv::ColorConversionCodes::COLOR_RGB2YUV_YV12)

    .value("COLOR_BGR2YUV_YV12",cv::ColorConversionCodes::COLOR_BGR2YUV_YV12)

    .value("COLOR_RGBA2YUV_YV12",cv::ColorConversionCodes::COLOR_RGBA2YUV_YV12)

    .value("COLOR_BGRA2YUV_YV12",cv::ColorConversionCodes::COLOR_BGRA2YUV_YV12)

    .value("COLOR_BayerBG2BGR",cv::ColorConversionCodes::COLOR_BayerBG2BGR)

    .value("COLOR_BayerGB2BGR",cv::ColorConversionCodes::COLOR_BayerGB2BGR)

    .value("COLOR_BayerRG2BGR",cv::ColorConversionCodes::COLOR_BayerRG2BGR)

    .value("COLOR_BayerGR2BGR",cv::ColorConversionCodes::COLOR_BayerGR2BGR)

    .value("COLOR_BayerBG2RGB",cv::ColorConversionCodes::COLOR_BayerBG2RGB)

    .value("COLOR_BayerGB2RGB",cv::ColorConversionCodes::COLOR_BayerGB2RGB)

    .value("COLOR_BayerRG2RGB",cv::ColorConversionCodes::COLOR_BayerRG2RGB)

    .value("COLOR_BayerGR2RGB",cv::ColorConversionCodes::COLOR_BayerGR2RGB)

    .value("COLOR_BayerBG2GRAY",cv::ColorConversionCodes::COLOR_BayerBG2GRAY)

    .value("COLOR_BayerGB2GRAY",cv::ColorConversionCodes::COLOR_BayerGB2GRAY)

    .value("COLOR_BayerRG2GRAY",cv::ColorConversionCodes::COLOR_BayerRG2GRAY)

    .value("COLOR_BayerGR2GRAY",cv::ColorConversionCodes::COLOR_BayerGR2GRAY)

    .value("COLOR_BayerBG2BGR_VNG",cv::ColorConversionCodes::COLOR_BayerBG2BGR_VNG)

    .value("COLOR_BayerGB2BGR_VNG",cv::ColorConversionCodes::COLOR_BayerGB2BGR_VNG)

    .value("COLOR_BayerRG2BGR_VNG",cv::ColorConversionCodes::COLOR_BayerRG2BGR_VNG)

    .value("COLOR_BayerGR2BGR_VNG",cv::ColorConversionCodes::COLOR_BayerGR2BGR_VNG)

    .value("COLOR_BayerBG2RGB_VNG",cv::ColorConversionCodes::COLOR_BayerBG2RGB_VNG)

    .value("COLOR_BayerGB2RGB_VNG",cv::ColorConversionCodes::COLOR_BayerGB2RGB_VNG)

    .value("COLOR_BayerRG2RGB_VNG",cv::ColorConversionCodes::COLOR_BayerRG2RGB_VNG)

    .value("COLOR_BayerGR2RGB_VNG",cv::ColorConversionCodes::COLOR_BayerGR2RGB_VNG)

    .value("COLOR_BayerBG2BGR_EA",cv::ColorConversionCodes::COLOR_BayerBG2BGR_EA)

    .value("COLOR_BayerGB2BGR_EA",cv::ColorConversionCodes::COLOR_BayerGB2BGR_EA)

    .value("COLOR_BayerRG2BGR_EA",cv::ColorConversionCodes::COLOR_BayerRG2BGR_EA)

    .value("COLOR_BayerGR2BGR_EA",cv::ColorConversionCodes::COLOR_BayerGR2BGR_EA)

    .value("COLOR_BayerBG2RGB_EA",cv::ColorConversionCodes::COLOR_BayerBG2RGB_EA)

    .value("COLOR_BayerGB2RGB_EA",cv::ColorConversionCodes::COLOR_BayerGB2RGB_EA)

    .value("COLOR_BayerRG2RGB_EA",cv::ColorConversionCodes::COLOR_BayerRG2RGB_EA)

    .value("COLOR_BayerGR2RGB_EA",cv::ColorConversionCodes::COLOR_BayerGR2RGB_EA)

    .value("COLOR_COLORCVT_MAX",cv::ColorConversionCodes::COLOR_COLORCVT_MAX)

    ;

    emscripten::enum_<cv::ColormapTypes>("cv_ColormapTypes")
    
    .value("COLORMAP_AUTUMN",cv::ColormapTypes::COLORMAP_AUTUMN)

    .value("COLORMAP_BONE",cv::ColormapTypes::COLORMAP_BONE)

    .value("COLORMAP_JET",cv::ColormapTypes::COLORMAP_JET)

    .value("COLORMAP_WINTER",cv::ColormapTypes::COLORMAP_WINTER)

    .value("COLORMAP_RAINBOW",cv::ColormapTypes::COLORMAP_RAINBOW)

    .value("COLORMAP_OCEAN",cv::ColormapTypes::COLORMAP_OCEAN)

    .value("COLORMAP_SUMMER",cv::ColormapTypes::COLORMAP_SUMMER)

    .value("COLORMAP_SPRING",cv::ColormapTypes::COLORMAP_SPRING)

    .value("COLORMAP_COOL",cv::ColormapTypes::COLORMAP_COOL)

    .value("COLORMAP_HSV",cv::ColormapTypes::COLORMAP_HSV)

    .value("COLORMAP_PINK",cv::ColormapTypes::COLORMAP_PINK)

    .value("COLORMAP_HOT",cv::ColormapTypes::COLORMAP_HOT)

    ;

    emscripten::enum_<cv::ConnectedComponentsTypes>("cv_ConnectedComponentsTypes")
    
    .value("CC_STAT_LEFT",cv::ConnectedComponentsTypes::CC_STAT_LEFT)

    .value("CC_STAT_TOP",cv::ConnectedComponentsTypes::CC_STAT_TOP)

    .value("CC_STAT_WIDTH",cv::ConnectedComponentsTypes::CC_STAT_WIDTH)

    .value("CC_STAT_HEIGHT",cv::ConnectedComponentsTypes::CC_STAT_HEIGHT)

    .value("CC_STAT_AREA",cv::ConnectedComponentsTypes::CC_STAT_AREA)

    .value("CC_STAT_MAX",cv::ConnectedComponentsTypes::CC_STAT_MAX)

    ;

    emscripten::enum_<cv::ContourApproximationModes>("cv_ContourApproximationModes")
    
    .value("CHAIN_APPROX_NONE",cv::ContourApproximationModes::CHAIN_APPROX_NONE)

    .value("CHAIN_APPROX_SIMPLE",cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE)

    .value("CHAIN_APPROX_TC89_L1",cv::ContourApproximationModes::CHAIN_APPROX_TC89_L1)

    .value("CHAIN_APPROX_TC89_KCOS",cv::ContourApproximationModes::CHAIN_APPROX_TC89_KCOS)

    ;

    emscripten::enum_<cv::CovarFlags>("cv_CovarFlags")
    
    .value("COVAR_SCRAMBLED",cv::CovarFlags::COVAR_SCRAMBLED)

    .value("COVAR_NORMAL",cv::CovarFlags::COVAR_NORMAL)

    .value("COVAR_USE_AVG",cv::CovarFlags::COVAR_USE_AVG)

    .value("COVAR_SCALE",cv::CovarFlags::COVAR_SCALE)

    .value("COVAR_ROWS",cv::CovarFlags::COVAR_ROWS)

    .value("COVAR_COLS",cv::CovarFlags::COVAR_COLS)

    ;

    emscripten::enum_<cv::DistanceTransformLabelTypes>("cv_DistanceTransformLabelTypes")
    
    .value("DIST_LABEL_CCOMP",cv::DistanceTransformLabelTypes::DIST_LABEL_CCOMP)

    .value("DIST_LABEL_PIXEL",cv::DistanceTransformLabelTypes::DIST_LABEL_PIXEL)

    ;

    emscripten::enum_<cv::DistanceTransformMasks>("cv_DistanceTransformMasks")
    
    .value("DIST_MASK_3",cv::DistanceTransformMasks::DIST_MASK_3)

    .value("DIST_MASK_5",cv::DistanceTransformMasks::DIST_MASK_5)

    .value("DIST_MASK_PRECISE",cv::DistanceTransformMasks::DIST_MASK_PRECISE)

    ;

    emscripten::enum_<cv::DistanceTypes>("cv_DistanceTypes")
    
    .value("DIST_USER",cv::DistanceTypes::DIST_USER)

    .value("DIST_L1",cv::DistanceTypes::DIST_L1)

    .value("DIST_L2",cv::DistanceTypes::DIST_L2)

    .value("DIST_C",cv::DistanceTypes::DIST_C)

    .value("DIST_L12",cv::DistanceTypes::DIST_L12)

    .value("DIST_FAIR",cv::DistanceTypes::DIST_FAIR)

    .value("DIST_WELSCH",cv::DistanceTypes::DIST_WELSCH)

    .value("DIST_HUBER",cv::DistanceTypes::DIST_HUBER)

    ;

    emscripten::enum_<cv::FloodFillFlags>("cv_FloodFillFlags")
    
    .value("FLOODFILL_FIXED_RANGE",cv::FloodFillFlags::FLOODFILL_FIXED_RANGE)

    .value("FLOODFILL_MASK_ONLY",cv::FloodFillFlags::FLOODFILL_MASK_ONLY)

    ;

    emscripten::enum_<cv::GrabCutClasses>("cv_GrabCutClasses")
    
    .value("GC_BGD",cv::GrabCutClasses::GC_BGD)

    .value("GC_FGD",cv::GrabCutClasses::GC_FGD)

    .value("GC_PR_BGD",cv::GrabCutClasses::GC_PR_BGD)

    .value("GC_PR_FGD",cv::GrabCutClasses::GC_PR_FGD)

    ;

    emscripten::enum_<cv::GrabCutModes>("cv_GrabCutModes")
    
    .value("GC_INIT_WITH_RECT",cv::GrabCutModes::GC_INIT_WITH_RECT)

    .value("GC_INIT_WITH_MASK",cv::GrabCutModes::GC_INIT_WITH_MASK)

    .value("GC_EVAL",cv::GrabCutModes::GC_EVAL)

    ;

    emscripten::enum_<cv::HersheyFonts>("cv_HersheyFonts")
    
    .value("FONT_HERSHEY_SIMPLEX",cv::HersheyFonts::FONT_HERSHEY_SIMPLEX)

    .value("FONT_HERSHEY_PLAIN",cv::HersheyFonts::FONT_HERSHEY_PLAIN)

    .value("FONT_HERSHEY_DUPLEX",cv::HersheyFonts::FONT_HERSHEY_DUPLEX)

    .value("FONT_HERSHEY_COMPLEX",cv::HersheyFonts::FONT_HERSHEY_COMPLEX)

    .value("FONT_HERSHEY_TRIPLEX",cv::HersheyFonts::FONT_HERSHEY_TRIPLEX)

    .value("FONT_HERSHEY_COMPLEX_SMALL",cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL)

    .value("FONT_HERSHEY_SCRIPT_SIMPLEX",cv::HersheyFonts::FONT_HERSHEY_SCRIPT_SIMPLEX)

    .value("FONT_HERSHEY_SCRIPT_COMPLEX",cv::HersheyFonts::FONT_HERSHEY_SCRIPT_COMPLEX)

    .value("FONT_ITALIC",cv::HersheyFonts::FONT_ITALIC)

    ;

    emscripten::enum_<cv::HistCompMethods>("cv_HistCompMethods")
    
    .value("HISTCMP_CORREL",cv::HistCompMethods::HISTCMP_CORREL)

    .value("HISTCMP_CHISQR",cv::HistCompMethods::HISTCMP_CHISQR)

    .value("HISTCMP_INTERSECT",cv::HistCompMethods::HISTCMP_INTERSECT)

    .value("HISTCMP_BHATTACHARYYA",cv::HistCompMethods::HISTCMP_BHATTACHARYYA)

    .value("HISTCMP_HELLINGER",cv::HistCompMethods::HISTCMP_HELLINGER)

    .value("HISTCMP_CHISQR_ALT",cv::HistCompMethods::HISTCMP_CHISQR_ALT)

    .value("HISTCMP_KL_DIV",cv::HistCompMethods::HISTCMP_KL_DIV)

    ;

    emscripten::enum_<cv::HoughModes>("cv_HoughModes")
    
    .value("HOUGH_STANDARD",cv::HoughModes::HOUGH_STANDARD)

    .value("HOUGH_PROBABILISTIC",cv::HoughModes::HOUGH_PROBABILISTIC)

    .value("HOUGH_MULTI_SCALE",cv::HoughModes::HOUGH_MULTI_SCALE)

    .value("HOUGH_GRADIENT",cv::HoughModes::HOUGH_GRADIENT)

    ;

    emscripten::enum_<cv::InterpolationFlags>("cv_InterpolationFlags")
    
    .value("INTER_NEAREST",cv::InterpolationFlags::INTER_NEAREST)

    .value("INTER_LINEAR",cv::InterpolationFlags::INTER_LINEAR)

    .value("INTER_CUBIC",cv::InterpolationFlags::INTER_CUBIC)

    .value("INTER_AREA",cv::InterpolationFlags::INTER_AREA)

    .value("INTER_LANCZOS4",cv::InterpolationFlags::INTER_LANCZOS4)

    .value("INTER_MAX",cv::InterpolationFlags::INTER_MAX)

    .value("WARP_FILL_OUTLIERS",cv::InterpolationFlags::WARP_FILL_OUTLIERS)

    .value("WARP_INVERSE_MAP",cv::InterpolationFlags::WARP_INVERSE_MAP)

    ;

    emscripten::enum_<cv::InterpolationMasks>("cv_InterpolationMasks")
    
    .value("INTER_BITS",cv::InterpolationMasks::INTER_BITS)

    .value("INTER_BITS2",cv::InterpolationMasks::INTER_BITS2)

    .value("INTER_TAB_SIZE",cv::InterpolationMasks::INTER_TAB_SIZE)

    .value("INTER_TAB_SIZE2",cv::InterpolationMasks::INTER_TAB_SIZE2)

    ;

    emscripten::enum_<cv::KmeansFlags>("cv_KmeansFlags")
    
    .value("KMEANS_RANDOM_CENTERS",cv::KmeansFlags::KMEANS_RANDOM_CENTERS)

    .value("KMEANS_PP_CENTERS",cv::KmeansFlags::KMEANS_PP_CENTERS)

    .value("KMEANS_USE_INITIAL_LABELS",cv::KmeansFlags::KMEANS_USE_INITIAL_LABELS)

    ;

    emscripten::enum_<cv::LineSegmentDetectorModes>("cv_LineSegmentDetectorModes")
    
    .value("LSD_REFINE_NONE",cv::LineSegmentDetectorModes::LSD_REFINE_NONE)

    .value("LSD_REFINE_STD",cv::LineSegmentDetectorModes::LSD_REFINE_STD)

    .value("LSD_REFINE_ADV",cv::LineSegmentDetectorModes::LSD_REFINE_ADV)

    ;

    emscripten::enum_<cv::LineTypes>("cv_LineTypes")
    
    .value("FILLED",cv::LineTypes::FILLED)

    .value("LINE_4",cv::LineTypes::LINE_4)

    .value("LINE_8",cv::LineTypes::LINE_8)

    .value("LINE_AA",cv::LineTypes::LINE_AA)

    ;

    emscripten::enum_<cv::MorphShapes>("cv_MorphShapes")
    
    .value("MORPH_RECT",cv::MorphShapes::MORPH_RECT)

    .value("MORPH_CROSS",cv::MorphShapes::MORPH_CROSS)

    .value("MORPH_ELLIPSE",cv::MorphShapes::MORPH_ELLIPSE)

    ;

    emscripten::enum_<cv::MorphTypes>("cv_MorphTypes")
    
    .value("MORPH_ERODE",cv::MorphTypes::MORPH_ERODE)

    .value("MORPH_DILATE",cv::MorphTypes::MORPH_DILATE)

    .value("MORPH_OPEN",cv::MorphTypes::MORPH_OPEN)

    .value("MORPH_CLOSE",cv::MorphTypes::MORPH_CLOSE)

    .value("MORPH_GRADIENT",cv::MorphTypes::MORPH_GRADIENT)

    .value("MORPH_TOPHAT",cv::MorphTypes::MORPH_TOPHAT)

    .value("MORPH_BLACKHAT",cv::MorphTypes::MORPH_BLACKHAT)

    ;

    emscripten::enum_<cv::PCA::Flags>("cv_PCA_Flags")
    
    .value("DATA_AS_ROW",cv::PCA::Flags::DATA_AS_ROW)

    .value("DATA_AS_COL",cv::PCA::Flags::DATA_AS_COL)

    .value("USE_AVG",cv::PCA::Flags::USE_AVG)

    ;

    emscripten::enum_<cv::RectanglesIntersectTypes>("cv_RectanglesIntersectTypes")
    
    .value("INTERSECT_NONE",cv::RectanglesIntersectTypes::INTERSECT_NONE)

    .value("INTERSECT_PARTIAL",cv::RectanglesIntersectTypes::INTERSECT_PARTIAL)

    .value("INTERSECT_FULL",cv::RectanglesIntersectTypes::INTERSECT_FULL)

    ;

    emscripten::enum_<cv::ReduceTypes>("cv_ReduceTypes")
    
    .value("REDUCE_SUM",cv::ReduceTypes::REDUCE_SUM)

    .value("REDUCE_AVG",cv::ReduceTypes::REDUCE_AVG)

    .value("REDUCE_MAX",cv::ReduceTypes::REDUCE_MAX)

    .value("REDUCE_MIN",cv::ReduceTypes::REDUCE_MIN)

    ;

    emscripten::enum_<cv::RetrievalModes>("cv_RetrievalModes")
    
    .value("RETR_EXTERNAL",cv::RetrievalModes::RETR_EXTERNAL)

    .value("RETR_LIST",cv::RetrievalModes::RETR_LIST)

    .value("RETR_CCOMP",cv::RetrievalModes::RETR_CCOMP)

    .value("RETR_TREE",cv::RetrievalModes::RETR_TREE)

    .value("RETR_FLOODFILL",cv::RetrievalModes::RETR_FLOODFILL)

    ;

    emscripten::enum_<cv::SVD::Flags>("cv_SVD_Flags")
    
    .value("MODIFY_A",cv::SVD::Flags::MODIFY_A)

    .value("NO_UV",cv::SVD::Flags::NO_UV)

    .value("FULL_UV",cv::SVD::Flags::FULL_UV)

    ;

    emscripten::enum_<cv::SortFlags>("cv_SortFlags")
    
    .value("SORT_EVERY_ROW",cv::SortFlags::SORT_EVERY_ROW)

    .value("SORT_EVERY_COLUMN",cv::SortFlags::SORT_EVERY_COLUMN)

    .value("SORT_ASCENDING",cv::SortFlags::SORT_ASCENDING)

    .value("SORT_DESCENDING",cv::SortFlags::SORT_DESCENDING)

    ;

    emscripten::enum_<cv::TemplateMatchModes>("cv_TemplateMatchModes")
    
    .value("TM_SQDIFF",cv::TemplateMatchModes::TM_SQDIFF)

    .value("TM_SQDIFF_NORMED",cv::TemplateMatchModes::TM_SQDIFF_NORMED)

    .value("TM_CCORR",cv::TemplateMatchModes::TM_CCORR)

    .value("TM_CCORR_NORMED",cv::TemplateMatchModes::TM_CCORR_NORMED)

    .value("TM_CCOEFF",cv::TemplateMatchModes::TM_CCOEFF)

    .value("TM_CCOEFF_NORMED",cv::TemplateMatchModes::TM_CCOEFF_NORMED)

    ;

    emscripten::enum_<cv::ThresholdTypes>("cv_ThresholdTypes")
    
    .value("THRESH_BINARY",cv::ThresholdTypes::THRESH_BINARY)

    .value("THRESH_BINARY_INV",cv::ThresholdTypes::THRESH_BINARY_INV)

    .value("THRESH_TRUNC",cv::ThresholdTypes::THRESH_TRUNC)

    .value("THRESH_TOZERO",cv::ThresholdTypes::THRESH_TOZERO)

    .value("THRESH_TOZERO_INV",cv::ThresholdTypes::THRESH_TOZERO_INV)

    .value("THRESH_MASK",cv::ThresholdTypes::THRESH_MASK)

    .value("THRESH_OTSU",cv::ThresholdTypes::THRESH_OTSU)

    .value("THRESH_TRIANGLE",cv::ThresholdTypes::THRESH_TRIANGLE)

    ;

    emscripten::enum_<cv::UMatUsageFlags>("cv_UMatUsageFlags")
    
    .value("USAGE_DEFAULT",cv::UMatUsageFlags::USAGE_DEFAULT)

    .value("USAGE_ALLOCATE_HOST_MEMORY",cv::UMatUsageFlags::USAGE_ALLOCATE_HOST_MEMORY)

    .value("USAGE_ALLOCATE_DEVICE_MEMORY",cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY)

    .value("USAGE_ALLOCATE_SHARED_MEMORY",cv::UMatUsageFlags::USAGE_ALLOCATE_SHARED_MEMORY)

    .value("__UMAT_USAGE_FLAGS_32BIT",cv::UMatUsageFlags::__UMAT_USAGE_FLAGS_32BIT)

    ;

    emscripten::enum_<cv::UndistortTypes>("cv_UndistortTypes")
    
    .value("PROJ_SPHERICAL_ORTHO",cv::UndistortTypes::PROJ_SPHERICAL_ORTHO)

    .value("PROJ_SPHERICAL_EQRECT",cv::UndistortTypes::PROJ_SPHERICAL_EQRECT)

    ;

    emscripten::enum_<cv::ml::ANN_MLP::TrainFlags>("cv_ml_ANN_MLP_TrainFlags")
    
    .value("UPDATE_WEIGHTS",cv::ml::ANN_MLP::TrainFlags::UPDATE_WEIGHTS)

    .value("NO_INPUT_SCALE",cv::ml::ANN_MLP::TrainFlags::NO_INPUT_SCALE)

    .value("NO_OUTPUT_SCALE",cv::ml::ANN_MLP::TrainFlags::NO_OUTPUT_SCALE)

    ;

    emscripten::enum_<cv::ml::ANN_MLP::TrainingMethods>("cv_ml_ANN_MLP_TrainingMethods")
    
    .value("BACKPROP",cv::ml::ANN_MLP::TrainingMethods::BACKPROP)

    .value("RPROP",cv::ml::ANN_MLP::TrainingMethods::RPROP)

    ;

    emscripten::enum_<cv::ml::DTrees::Flags>("cv_ml_DTrees_Flags")
    
    .value("PREDICT_AUTO",cv::ml::DTrees::Flags::PREDICT_AUTO)

    .value("PREDICT_SUM",cv::ml::DTrees::Flags::PREDICT_SUM)

    .value("PREDICT_MAX_VOTE",cv::ml::DTrees::Flags::PREDICT_MAX_VOTE)

    .value("PREDICT_MASK",cv::ml::DTrees::Flags::PREDICT_MASK)

    ;

    emscripten::enum_<cv::ml::EM::Types>("cv_ml_EM_Types")
    
    .value("COV_MAT_SPHERICAL",cv::ml::EM::Types::COV_MAT_SPHERICAL)

    .value("COV_MAT_DIAGONAL",cv::ml::EM::Types::COV_MAT_DIAGONAL)

    .value("COV_MAT_GENERIC",cv::ml::EM::Types::COV_MAT_GENERIC)

    .value("COV_MAT_DEFAULT",cv::ml::EM::Types::COV_MAT_DEFAULT)

    ;

    emscripten::enum_<cv::ml::ErrorTypes>("cv_ml_ErrorTypes")
    
    .value("TEST_ERROR",cv::ml::ErrorTypes::TEST_ERROR)

    .value("TRAIN_ERROR",cv::ml::ErrorTypes::TRAIN_ERROR)

    ;

    emscripten::enum_<cv::ml::KNearest::Types>("cv_ml_KNearest_Types")
    
    .value("BRUTE_FORCE",cv::ml::KNearest::Types::BRUTE_FORCE)

    .value("KDTREE",cv::ml::KNearest::Types::KDTREE)

    ;

    emscripten::enum_<cv::ml::LogisticRegression::Methods>("cv_ml_LogisticRegression_Methods")
    
    .value("BATCH",cv::ml::LogisticRegression::Methods::BATCH)

    .value("MINI_BATCH",cv::ml::LogisticRegression::Methods::MINI_BATCH)

    ;

    emscripten::enum_<cv::ml::SVM::KernelTypes>("cv_ml_SVM_KernelTypes")
    
    .value("CUSTOM",cv::ml::SVM::KernelTypes::CUSTOM)

    .value("LINEAR",cv::ml::SVM::KernelTypes::LINEAR)

    .value("POLY",cv::ml::SVM::KernelTypes::POLY)

    .value("RBF",cv::ml::SVM::KernelTypes::RBF)

    .value("SIGMOID",cv::ml::SVM::KernelTypes::SIGMOID)

    .value("CHI2",cv::ml::SVM::KernelTypes::CHI2)

    .value("INTER",cv::ml::SVM::KernelTypes::INTER)

    ;

    emscripten::enum_<cv::ml::SVM::ParamTypes>("cv_ml_SVM_ParamTypes")
    
    .value("C",cv::ml::SVM::ParamTypes::C)

    .value("GAMMA",cv::ml::SVM::ParamTypes::GAMMA)

    .value("P",cv::ml::SVM::ParamTypes::P)

    .value("NU",cv::ml::SVM::ParamTypes::NU)

    .value("COEF",cv::ml::SVM::ParamTypes::COEF)

    .value("DEGREE",cv::ml::SVM::ParamTypes::DEGREE)

    ;

    emscripten::enum_<cv::ml::SVM::Types>("cv_ml_SVM_Types")
    
    .value("C_SVC",cv::ml::SVM::Types::C_SVC)

    .value("NU_SVC",cv::ml::SVM::Types::NU_SVC)

    .value("ONE_CLASS",cv::ml::SVM::Types::ONE_CLASS)

    .value("EPS_SVR",cv::ml::SVM::Types::EPS_SVR)

    .value("NU_SVR",cv::ml::SVM::Types::NU_SVR)

    ;

    emscripten::enum_<cv::ml::SampleTypes>("cv_ml_SampleTypes")
    
    .value("ROW_SAMPLE",cv::ml::SampleTypes::ROW_SAMPLE)

    .value("COL_SAMPLE",cv::ml::SampleTypes::COL_SAMPLE)

    ;

    emscripten::enum_<cv::ml::StatModel::Flags>("cv_ml_StatModel_Flags")
    
    .value("UPDATE_MODEL",cv::ml::StatModel::Flags::UPDATE_MODEL)

    .value("RAW_OUTPUT",cv::ml::StatModel::Flags::RAW_OUTPUT)

    .value("COMPRESSED_INPUT",cv::ml::StatModel::Flags::COMPRESSED_INPUT)

    .value("PREPROCESSED_INPUT",cv::ml::StatModel::Flags::PREPROCESSED_INPUT)

    ;

    emscripten::enum_<cv::ml::VariableTypes>("cv_ml_VariableTypes")
    
    .value("VAR_NUMERICAL",cv::ml::VariableTypes::VAR_NUMERICAL)

    .value("VAR_ORDERED",cv::ml::VariableTypes::VAR_ORDERED)

    .value("VAR_CATEGORICAL",cv::ml::VariableTypes::VAR_CATEGORICAL)

    ;

    emscripten::enum_<cv::ocl::OclVectorStrategy>("cv_ocl_OclVectorStrategy")
    
    .value("OCL_VECTOR_OWN",cv::ocl::OclVectorStrategy::OCL_VECTOR_OWN)

    .value("OCL_VECTOR_MAX",cv::ocl::OclVectorStrategy::OCL_VECTOR_MAX)

    .value("OCL_VECTOR_DEFAULT",cv::ocl::OclVectorStrategy::OCL_VECTOR_DEFAULT)

    ;

    }

        EMSCRIPTEN_BINDINGS(StringBinding) {
            emscripten::class_<cv::String>("String")
                .function("size", select_overload<size_t()const>(&cv::String::size))
                .function("length", select_overload<size_t()const>(&cv::String::length))
                .function("empty", select_overload<bool()const>(&cv::String::empty))
                .function("toLowerCase",select_overload<String()const>(&cv::String::toLowerCase))
                .function("compareString",select_overload<int(const String&)const>(&cv::String::compare))
                .function("compare",select_overload<int(const char* )const>(&cv::String::compare),allow_raw_pointers() )
                .constructor< const std::string& >()
            ;
        }
        