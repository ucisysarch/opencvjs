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

namespace Wrappers {
    
    void Canny_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,double arg3,int arg4,bool arg5){
        return cv::Canny(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void GaussianBlur_wrapper(const cv::Mat& arg0,cv::Mat& arg1,Size arg2,double arg3,double arg4,int arg5){
        return cv::GaussianBlur(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void HoughCircles_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,double arg3,double arg4,double arg5,double arg6,int arg7,int arg8){
        return cv::HoughCircles(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
    }


    void HoughLines_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,double arg3,int arg4,double arg5,double arg6,double arg7,double arg8){
        return cv::HoughLines(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
    }


    void HoughLinesP_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,double arg3,int arg4,double arg5,double arg6){
        return cv::HoughLinesP(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void HuMoments_wrapper(const Moments& arg0,cv::Mat& arg1){
        return cv::HuMoments(arg0,arg1) ;
    }


    void LUT_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2){
        return cv::LUT(arg0,arg1,arg2) ;
    }


    void Laplacian_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,double arg4,double arg5,int arg6){
        return cv::Laplacian(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    double Mahalanobis_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2){
        return cv::Mahalanobis(arg0,arg1,arg2) ;
    }


    void PCABackProject_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2,cv::Mat& arg3){
        return cv::PCABackProject(arg0,arg1,arg2,arg3) ;
    }


    void PCACompute_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,int arg3){
        return cv::PCACompute(arg0,arg1,arg2,arg3) ;
    }


    void PCACompute_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,double arg3){
        return cv::PCACompute(arg0,arg1,arg2,arg3) ;
    }


    void PCAProject_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2,cv::Mat& arg3){
        return cv::PCAProject(arg0,arg1,arg2,arg3) ;
    }


    double PSNR_wrapper(const cv::Mat& arg0,const cv::Mat& arg1){
        return cv::PSNR(arg0,arg1) ;
    }


    void SVBackSubst_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2,const cv::Mat& arg3,cv::Mat& arg4){
        return cv::SVBackSubst(arg0,arg1,arg2,arg3,arg4) ;
    }


    void SVDecomp_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,cv::Mat& arg3,int arg4){
        return cv::SVDecomp(arg0,arg1,arg2,arg3,arg4) ;
    }


    void Scharr_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,int arg4,double arg5,double arg6,int arg7){
        return cv::Scharr(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7) ;
    }


    void Sobel_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,int arg4,int arg5,double arg6,double arg7,int arg8){
        return cv::Sobel(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
    }


    void absdiff_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2){
        return cv::absdiff(arg0,arg1,arg2) ;
    }


    void accumulate_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2){
        return cv::accumulate(arg0,arg1,arg2) ;
    }


    void accumulateProduct_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,const cv::Mat& arg3){
        return cv::accumulateProduct(arg0,arg1,arg2,arg3) ;
    }


    void accumulateSquare_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2){
        return cv::accumulateSquare(arg0,arg1,arg2) ;
    }


    void accumulateWeighted_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,const cv::Mat& arg3){
        return cv::accumulateWeighted(arg0,arg1,arg2,arg3) ;
    }


    void adaptiveThreshold_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,int arg3,int arg4,int arg5,double arg6){
        return cv::adaptiveThreshold(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void add_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,const cv::Mat& arg3,int arg4){
        return cv::add(arg0,arg1,arg2,arg3,arg4) ;
    }


    void addWeighted_wrapper(const cv::Mat& arg0,double arg1,const cv::Mat& arg2,double arg3,double arg4,cv::Mat& arg5,int arg6){
        return cv::addWeighted(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void applyColorMap_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::applyColorMap(arg0,arg1,arg2) ;
    }


    void approxPolyDP_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,bool arg3){
        return cv::approxPolyDP(arg0,arg1,arg2,arg3) ;
    }


    double arcLength_wrapper(const cv::Mat& arg0,bool arg1){
        return cv::arcLength(arg0,arg1) ;
    }


    void arrowedLine_wrapper(cv::Mat& arg0,Point arg1,Point arg2,const Scalar& arg3,int arg4,int arg5,int arg6,double arg7){
        return cv::arrowedLine(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7) ;
    }


    void batchDistance_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,int arg3,cv::Mat& arg4,int arg5,int arg6,const cv::Mat& arg7,int arg8,bool arg9){
        return cv::batchDistance(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9) ;
    }


    void bilateralFilter_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,double arg3,double arg4,int arg5){
        return cv::bilateralFilter(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void bitwise_and_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,const cv::Mat& arg3){
        return cv::bitwise_and(arg0,arg1,arg2,arg3) ;
    }


    void bitwise_not_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2){
        return cv::bitwise_not(arg0,arg1,arg2) ;
    }


    void bitwise_or_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,const cv::Mat& arg3){
        return cv::bitwise_or(arg0,arg1,arg2,arg3) ;
    }


    void bitwise_xor_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,const cv::Mat& arg3){
        return cv::bitwise_xor(arg0,arg1,arg2,arg3) ;
    }


    void blur_wrapper(const cv::Mat& arg0,cv::Mat& arg1,Size arg2,Point arg3,int arg4){
        return cv::blur(arg0,arg1,arg2,arg3,arg4) ;
    }


    int borderInterpolate_wrapper(int arg0,int arg1,int arg2){
        return cv::borderInterpolate(arg0,arg1,arg2) ;
    }


    Rect boundingRect_wrapper(const cv::Mat& arg0){
        return cv::boundingRect(arg0) ;
    }


    void boxFilter_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,Size arg3,Point arg4,bool arg5,int arg6){
        return cv::boxFilter(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void boxPoints_wrapper(RotatedRect arg0,cv::Mat& arg1){
        return cv::boxPoints(arg0,arg1) ;
    }


    void calcBackProject_wrapper(const std::vector<cv::Mat>& arg0,const std::vector<int>& arg1,const cv::Mat& arg2,cv::Mat& arg3,const std::vector<float>& arg4,double arg5){
        return cv::calcBackProject(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void calcCovarMatrix_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,int arg3,int arg4){
        return cv::calcCovarMatrix(arg0,arg1,arg2,arg3,arg4) ;
    }


    void calcHist_wrapper(const std::vector<cv::Mat>& arg0,const std::vector<int>& arg1,const cv::Mat& arg2,cv::Mat& arg3,const std::vector<int>& arg4,const std::vector<float>& arg5,bool arg6){
        return cv::calcHist(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void cartToPolar_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,cv::Mat& arg3,bool arg4){
        return cv::cartToPolar(arg0,arg1,arg2,arg3,arg4) ;
    }


    bool checkRange_wrapper(const cv::Mat& arg0,bool arg1, Point* arg2,double arg3,double arg4){
        return cv::checkRange(arg0,arg1,arg2,arg3,arg4) ;
    }


    void circle_wrapper(cv::Mat& arg0,Point arg1,int arg2,const Scalar& arg3,int arg4,int arg5,int arg6){
        return cv::circle(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    bool clipLine_wrapper(Rect arg0,  Point& arg1,  Point& arg2){
        return cv::clipLine(arg0,arg1,arg2) ;
    }


    void compare_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,int arg3){
        return cv::compare(arg0,arg1,arg2,arg3) ;
    }


    double compareHist_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,int arg2){
        return cv::compareHist(arg0,arg1,arg2) ;
    }


    void completeSymm_wrapper(cv::Mat& arg0,bool arg1){
        return cv::completeSymm(arg0,arg1) ;
    }


    int connectedComponents_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3){
        return cv::connectedComponents(arg0,arg1,arg2,arg3) ;
    }


    int connectedComponentsWithStats_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,cv::Mat& arg3,int arg4,int arg5){
        return cv::connectedComponentsWithStats(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    double contourArea_wrapper(const cv::Mat& arg0,bool arg1){
        return cv::contourArea(arg0,arg1) ;
    }


    void convertMaps_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,cv::Mat& arg3,int arg4,bool arg5){
        return cv::convertMaps(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void convertScaleAbs_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,double arg3){
        return cv::convertScaleAbs(arg0,arg1,arg2,arg3) ;
    }


    void convexHull_wrapper(const cv::Mat& arg0,cv::Mat& arg1,bool arg2,bool arg3){
        return cv::convexHull(arg0,arg1,arg2,arg3) ;
    }


    void convexityDefects_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2){
        return cv::convexityDefects(arg0,arg1,arg2) ;
    }


    void copyMakeBorder_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,int arg4,int arg5,int arg6,const Scalar& arg7){
        return cv::copyMakeBorder(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7) ;
    }


    void cornerEigenValsAndVecs_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,int arg4){
        return cv::cornerEigenValsAndVecs(arg0,arg1,arg2,arg3,arg4) ;
    }


    void cornerHarris_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,double arg4,int arg5){
        return cv::cornerHarris(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void cornerMinEigenVal_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,int arg4){
        return cv::cornerMinEigenVal(arg0,arg1,arg2,arg3,arg4) ;
    }


    void cornerSubPix_wrapper(const cv::Mat& arg0,cv::Mat& arg1,Size arg2,Size arg3,TermCriteria arg4){
        return cv::cornerSubPix(arg0,arg1,arg2,arg3,arg4) ;
    }


    int countNonZero_wrapper(const cv::Mat& arg0){
        return cv::countNonZero(arg0) ;
    }


    Ptr<CLAHE> createCLAHE_wrapper(double arg0,Size arg1){
        return cv::createCLAHE(arg0,arg1) ;
    }


    void createHanningWindow_wrapper(cv::Mat& arg0,Size arg1,int arg2){
        return cv::createHanningWindow(arg0,arg1,arg2) ;
    }


    Ptr<LineSegmentDetector> createLineSegmentDetector_wrapper(int arg0,double arg1,double arg2,double arg3,double arg4,double arg5,double arg6,int arg7){
        return cv::createLineSegmentDetector(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7) ;
    }


    void cvtColor_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3){
        return cv::cvtColor(arg0,arg1,arg2,arg3) ;
    }


    void dct_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::dct(arg0,arg1,arg2) ;
    }


    void demosaicing_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3){
        return cv::demosaicing(arg0,arg1,arg2,arg3) ;
    }


    void destroyAllWindows_wrapper(){
        return cv::destroyAllWindows() ;
    }


    void destroyWindow_wrapper(const String& arg0){
        return cv::destroyWindow(arg0) ;
    }


    double determinant_wrapper(const cv::Mat& arg0){
        return cv::determinant(arg0) ;
    }


    void dft_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3){
        return cv::dft(arg0,arg1,arg2,arg3) ;
    }


    void dilate_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2,Point arg3,int arg4,int arg5,const Scalar& arg6){
        return cv::dilate(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void distanceTransform_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,int arg4){
        return cv::distanceTransform(arg0,arg1,arg2,arg3,arg4) ;
    }


    void distanceTransformWithLabels_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,int arg3,int arg4,int arg5){
        return cv::distanceTransform(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void divide_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,double arg3,int arg4){
        return cv::divide(arg0,arg1,arg2,arg3,arg4) ;
    }


    void divide_wrapper(double arg0,const cv::Mat& arg1,cv::Mat& arg2,int arg3){
        return cv::divide(arg0,arg1,arg2,arg3) ;
    }


    void drawContours_wrapper(cv::Mat& arg0,const std::vector<cv::Mat>& arg1,int arg2,const Scalar& arg3,int arg4,int arg5,const cv::Mat& arg6,int arg7,Point arg8){
        return cv::drawContours(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
    }


    bool eigen_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2){
        return cv::eigen(arg0,arg1,arg2) ;
    }


    void ellipse_wrapper(cv::Mat& arg0,Point arg1,Size arg2,double arg3,double arg4,double arg5,const Scalar& arg6,int arg7,int arg8,int arg9){
        return cv::ellipse(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9) ;
    }


    void ellipse_wrapper(cv::Mat& arg0,const RotatedRect& arg1,const Scalar& arg2,int arg3,int arg4){
        return cv::ellipse(arg0,arg1,arg2,arg3,arg4) ;
    }


    void ellipse2Poly_wrapper(Point arg0,Size arg1,int arg2,int arg3,int arg4,int arg5, std::vector<Point>& arg6){
        return cv::ellipse2Poly(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void equalizeHist_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::equalizeHist(arg0,arg1) ;
    }


    void erode_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2,Point arg3,int arg4,int arg5,const Scalar& arg6){
        return cv::erode(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void exp_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::exp(arg0,arg1) ;
    }


    void extractChannel_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::extractChannel(arg0,arg1,arg2) ;
    }


    void fillConvexPoly_wrapper(cv::Mat& arg0,const cv::Mat& arg1,const Scalar& arg2,int arg3,int arg4){
        return cv::fillConvexPoly(arg0,arg1,arg2,arg3,arg4) ;
    }


    void fillPoly_wrapper(cv::Mat& arg0,const std::vector<cv::Mat>& arg1,const Scalar& arg2,int arg3,int arg4,Point arg5){
        return cv::fillPoly(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void filter2D_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,const cv::Mat& arg3,Point arg4,double arg5,int arg6){
        return cv::filter2D(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void findContours_wrapper(cv::Mat& arg0,std::vector<cv::Mat>& arg1,cv::Mat& arg2,int arg3,int arg4,Point arg5){
        return cv::findContours(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void findNonZero_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::findNonZero(arg0,arg1) ;
    }


    RotatedRect fitEllipse_wrapper(const cv::Mat& arg0){
        return cv::fitEllipse(arg0) ;
    }


    void fitLine_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,double arg3,double arg4,double arg5){
        return cv::fitLine(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void flip_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::flip(arg0,arg1,arg2) ;
    }


    int floodFill_wrapper(cv::Mat& arg0,cv::Mat& arg1,Point arg2,Scalar arg3, Rect* arg4,Scalar arg5,Scalar arg6,int arg7){
        return cv::floodFill(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7) ;
    }


    void gemm_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,double arg2,const cv::Mat& arg3,double arg4,cv::Mat& arg5,int arg6){
        return cv::gemm(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    Mat getAffineTransform_wrapper(const cv::Mat& arg0,const cv::Mat& arg1){
        return cv::getAffineTransform(arg0,arg1) ;
    }


    Mat getDefaultNewCameraMatrix_wrapper(const cv::Mat& arg0,Size arg1,bool arg2){
        return cv::getDefaultNewCameraMatrix(arg0,arg1,arg2) ;
    }


    void getDerivKernels_wrapper(cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,int arg4,bool arg5,int arg6){
        return cv::getDerivKernels(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    Mat getGaborKernel_wrapper(Size arg0,double arg1,double arg2,double arg3,double arg4,double arg5,int arg6){
        return cv::getGaborKernel(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    Mat getGaussianKernel_wrapper(int arg0,double arg1,int arg2){
        return cv::getGaussianKernel(arg0,arg1,arg2) ;
    }


    int getOptimalDFTSize_wrapper(int arg0){
        return cv::getOptimalDFTSize(arg0) ;
    }


    Mat getPerspectiveTransform_wrapper(const cv::Mat& arg0,const cv::Mat& arg1){
        return cv::getPerspectiveTransform(arg0,arg1) ;
    }


    void getRectSubPix_wrapper(const cv::Mat& arg0,Size arg1,Point2f arg2,cv::Mat& arg3,int arg4){
        return cv::getRectSubPix(arg0,arg1,arg2,arg3,arg4) ;
    }


    Mat getRotationMatrix2D_wrapper(Point2f arg0,double arg1,double arg2){
        return cv::getRotationMatrix2D(arg0,arg1,arg2) ;
    }


    Mat getStructuringElement_wrapper(int arg0,Size arg1,Point arg2){
        return cv::getStructuringElement(arg0,arg1,arg2) ;
    }


    Size getTextSize_wrapper(const String& arg0,int arg1,double arg2,int arg3, int* arg4){
        return cv::getTextSize(arg0,arg1,arg2,arg3,arg4) ;
    }


    int getTrackbarPos_wrapper(const String& arg0,const String& arg1){
        return cv::getTrackbarPos(arg0,arg1) ;
    }


    double getWindowProperty_wrapper(const String& arg0,int arg1){
        return cv::getWindowProperty(arg0,arg1) ;
    }


    void goodFeaturesToTrack_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,double arg3,double arg4,const cv::Mat& arg5,int arg6,bool arg7,double arg8){
        return cv::goodFeaturesToTrack(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
    }


    void grabCut_wrapper(const cv::Mat& arg0,cv::Mat& arg1,Rect arg2,cv::Mat& arg3,cv::Mat& arg4,int arg5,int arg6){
        return cv::grabCut(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void hconcat_wrapper(const std::vector<cv::Mat>& arg0,cv::Mat& arg1){
        return cv::hconcat(arg0,arg1) ;
    }


    void idct_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::idct(arg0,arg1,arg2) ;
    }


    void idft_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3){
        return cv::idft(arg0,arg1,arg2,arg3) ;
    }


    Mat imdecode_wrapper(const cv::Mat& arg0,int arg1){
        return cv::imdecode(arg0,arg1) ;
    }


    bool imencode_wrapper(const String& arg0,const cv::Mat& arg1, std::vector<uchar>& arg2,const std::vector<int>& arg3){
        return cv::imencode(arg0,arg1,arg2,arg3) ;
    }


    Mat imread_wrapper(const String& arg0,int arg1){
        return cv::imread(arg0,arg1) ;
    }


    bool imreadmulti_wrapper(const String& arg0,std::vector<Mat>& arg1,int arg2){
        return cv::imreadmulti(arg0,arg1,arg2) ;
    }


    void imshow_wrapper(const String& arg0,const cv::Mat& arg1){
        return cv::imshow(arg0,arg1) ;
    }


    bool imwrite_wrapper(const String& arg0,const cv::Mat& arg1,const std::vector<int>& arg2){
        return cv::imwrite(arg0,arg1,arg2) ;
    }


    void inRange_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2,cv::Mat& arg3){
        return cv::inRange(arg0,arg1,arg2,arg3) ;
    }


    void initUndistortRectifyMap_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2,const cv::Mat& arg3,Size arg4,int arg5,cv::Mat& arg6,cv::Mat& arg7){
        return cv::initUndistortRectifyMap(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7) ;
    }


    float initWideAngleProjMap_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,Size arg2,int arg3,int arg4,cv::Mat& arg5,cv::Mat& arg6,int arg7,double arg8){
        return cv::initWideAngleProjMap(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
    }


    void insertChannel_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::insertChannel(arg0,arg1,arg2) ;
    }


    void integral_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::integral(arg0,arg1,arg2) ;
    }


    void integral2_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,int arg3,int arg4){
        return cv::integral(arg0,arg1,arg2,arg3,arg4) ;
    }


    void integral3_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,cv::Mat& arg3,int arg4,int arg5){
        return cv::integral(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    float intersectConvexConvex_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,bool arg3){
        return cv::intersectConvexConvex(arg0,arg1,arg2,arg3) ;
    }


    double invert_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::invert(arg0,arg1,arg2) ;
    }


    void invertAffineTransform_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::invertAffineTransform(arg0,arg1) ;
    }


    bool isContourConvex_wrapper(const cv::Mat& arg0){
        return cv::isContourConvex(arg0) ;
    }


    double kmeans_wrapper(const cv::Mat& arg0,int arg1,cv::Mat& arg2,TermCriteria arg3,int arg4,int arg5,cv::Mat& arg6){
        return cv::kmeans(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void line_wrapper(cv::Mat& arg0,Point arg1,Point arg2,const Scalar& arg3,int arg4,int arg5,int arg6){
        return cv::line(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void linearPolar_wrapper(const cv::Mat& arg0,cv::Mat& arg1,Point2f arg2,double arg3,int arg4){
        return cv::linearPolar(arg0,arg1,arg2,arg3,arg4) ;
    }


    void log_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::log(arg0,arg1) ;
    }


    void logPolar_wrapper(const cv::Mat& arg0,cv::Mat& arg1,Point2f arg2,double arg3,int arg4){
        return cv::logPolar(arg0,arg1,arg2,arg3,arg4) ;
    }


    void magnitude_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2){
        return cv::magnitude(arg0,arg1,arg2) ;
    }


    double matchShapes_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,int arg2,double arg3){
        return cv::matchShapes(arg0,arg1,arg2,arg3) ;
    }


    void matchTemplate_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,int arg3,const cv::Mat& arg4){
        return cv::matchTemplate(arg0,arg1,arg2,arg3,arg4) ;
    }


    void max_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2){
        return cv::max(arg0,arg1,arg2) ;
    }


    Scalar mean_wrapper(const cv::Mat& arg0,const cv::Mat& arg1){
        return cv::mean(arg0,arg1) ;
    }


    void meanStdDev_wrapper(const cv::Mat& arg0,cv::Mat& arg1,cv::Mat& arg2,const cv::Mat& arg3){
        return cv::meanStdDev(arg0,arg1,arg2,arg3) ;
    }


    void medianBlur_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::medianBlur(arg0,arg1,arg2) ;
    }


    void merge_wrapper(const std::vector<cv::Mat>& arg0,cv::Mat& arg1){
        return cv::merge(arg0,arg1) ;
    }


    void min_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2){
        return cv::min(arg0,arg1,arg2) ;
    }


    RotatedRect minAreaRect_wrapper(const cv::Mat& arg0){
        return cv::minAreaRect(arg0) ;
    }


    double minEnclosingTriangle_wrapper(const cv::Mat& arg0, OutputArray arg1){
        return cv::minEnclosingTriangle(arg0,arg1) ;
    }


    void minMaxLoc_wrapper(const cv::Mat& arg0, double* arg1, double* arg2, Point* arg3, Point* arg4,const cv::Mat& arg5){
        return cv::minMaxLoc(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void mixChannels_wrapper(const std::vector<cv::Mat>& arg0,InputOutputArrayOfArrays arg1,const std::vector<int>& arg2){
        return cv::mixChannels(arg0,arg1,arg2) ;
    }


    Moments moments_wrapper(const cv::Mat& arg0,bool arg1){
        return cv::moments(arg0,arg1) ;
    }


    void morphologyEx_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,const cv::Mat& arg3,Point arg4,int arg5,int arg6,const Scalar& arg7){
        return cv::morphologyEx(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7) ;
    }


    void moveWindow_wrapper(const String& arg0,int arg1,int arg2){
        return cv::moveWindow(arg0,arg1,arg2) ;
    }


    void mulSpectrums_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,int arg3,bool arg4){
        return cv::mulSpectrums(arg0,arg1,arg2,arg3,arg4) ;
    }


    void mulTransposed_wrapper(const cv::Mat& arg0,cv::Mat& arg1,bool arg2,const cv::Mat& arg3,double arg4,int arg5){
        return cv::mulTransposed(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void multiply_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,double arg3,int arg4){
        return cv::multiply(arg0,arg1,arg2,arg3,arg4) ;
    }


    void namedWindow_wrapper(const String& arg0,int arg1){
        return cv::namedWindow(arg0,arg1) ;
    }


    double norm_wrapper(const cv::Mat& arg0,int arg1,const cv::Mat& arg2){
        return cv::norm(arg0,arg1,arg2) ;
    }


    double norm_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,int arg2,const cv::Mat& arg3){
        return cv::norm(arg0,arg1,arg2,arg3) ;
    }


    void normalize_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,double arg3,int arg4,int arg5,const cv::Mat& arg6){
        return cv::normalize(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void patchNaNs_wrapper(cv::Mat& arg0,double arg1){
        return cv::patchNaNs(arg0,arg1) ;
    }


    void perspectiveTransform_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2){
        return cv::perspectiveTransform(arg0,arg1,arg2) ;
    }


    void phase_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,bool arg3){
        return cv::phase(arg0,arg1,arg2,arg3) ;
    }


    Point2d phaseCorrelate_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2, double* arg3){
        return cv::phaseCorrelate(arg0,arg1,arg2,arg3) ;
    }


    double pointPolygonTest_wrapper(const cv::Mat& arg0,Point2f arg1,bool arg2){
        return cv::pointPolygonTest(arg0,arg1,arg2) ;
    }


    void polarToCart_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,cv::Mat& arg3,bool arg4){
        return cv::polarToCart(arg0,arg1,arg2,arg3,arg4) ;
    }


    void polylines_wrapper(cv::Mat& arg0,const std::vector<cv::Mat>& arg1,bool arg2,const Scalar& arg3,int arg4,int arg5,int arg6){
        return cv::polylines(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void pow_wrapper(const cv::Mat& arg0,double arg1,cv::Mat& arg2){
        return cv::pow(arg0,arg1,arg2) ;
    }


    void preCornerDetect_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3){
        return cv::preCornerDetect(arg0,arg1,arg2,arg3) ;
    }


    void putText_wrapper(cv::Mat& arg0,const String& arg1,Point arg2,int arg3,double arg4,Scalar arg5,int arg6,int arg7,bool arg8){
        return cv::putText(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
    }


    void pyrDown_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const Size& arg2,int arg3){
        return cv::pyrDown(arg0,arg1,arg2,arg3) ;
    }


    void pyrMeanShiftFiltering_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,double arg3,int arg4,TermCriteria arg5){
        return cv::pyrMeanShiftFiltering(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void pyrUp_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const Size& arg2,int arg3){
        return cv::pyrUp(arg0,arg1,arg2,arg3) ;
    }


    void randShuffle_wrapper(cv::Mat& arg0,double arg1,RNG* arg2){
        return cv::randShuffle(arg0,arg1,arg2) ;
    }


    void randn_wrapper(cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2){
        return cv::randn(arg0,arg1,arg2) ;
    }


    void randu_wrapper(cv::Mat& arg0,const cv::Mat& arg1,const cv::Mat& arg2){
        return cv::randu(arg0,arg1,arg2) ;
    }


    void rectangle_wrapper(cv::Mat& arg0,Point arg1,Point arg2,const Scalar& arg3,int arg4,int arg5,int arg6){
        return cv::rectangle(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void reduce_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,int arg3,int arg4){
        return cv::reduce(arg0,arg1,arg2,arg3,arg4) ;
    }


    void remap_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2,const cv::Mat& arg3,int arg4,int arg5,const Scalar& arg6){
        return cv::remap(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void repeat_wrapper(const cv::Mat& arg0,int arg1,int arg2,cv::Mat& arg3){
        return cv::repeat(arg0,arg1,arg2,arg3) ;
    }


    void resize_wrapper(const cv::Mat& arg0,cv::Mat& arg1,Size arg2,double arg3,double arg4,int arg5){
        return cv::resize(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void resizeWindow_wrapper(const String& arg0,int arg1,int arg2){
        return cv::resizeWindow(arg0,arg1,arg2) ;
    }


    int rotatedRectangleIntersection_wrapper(const RotatedRect& arg0,const RotatedRect& arg1,cv::Mat& arg2){
        return cv::rotatedRectangleIntersection(arg0,arg1,arg2) ;
    }


    void scaleAdd_wrapper(const cv::Mat& arg0,double arg1,const cv::Mat& arg2,cv::Mat& arg3){
        return cv::scaleAdd(arg0,arg1,arg2,arg3) ;
    }


    void sepFilter2D_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,const cv::Mat& arg3,const cv::Mat& arg4,Point arg5,double arg6,int arg7){
        return cv::sepFilter2D(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7) ;
    }


    void setIdentity_wrapper(cv::Mat& arg0,const Scalar& arg1){
        return cv::setIdentity(arg0,arg1) ;
    }


    void setTrackbarMax_wrapper(const String& arg0,const String& arg1,int arg2){
        return cv::setTrackbarMax(arg0,arg1,arg2) ;
    }


    void setTrackbarPos_wrapper(const String& arg0,const String& arg1,int arg2){
        return cv::setTrackbarPos(arg0,arg1,arg2) ;
    }


    void setWindowProperty_wrapper(const String& arg0,int arg1,double arg2){
        return cv::setWindowProperty(arg0,arg1,arg2) ;
    }


    void setWindowTitle_wrapper(const String& arg0,const String& arg1){
        return cv::setWindowTitle(arg0,arg1) ;
    }


    bool solve_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,int arg3){
        return cv::solve(arg0,arg1,arg2,arg3) ;
    }


    int solveCubic_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::solveCubic(arg0,arg1) ;
    }


    double solvePoly_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::solvePoly(arg0,arg1,arg2) ;
    }


    void sort_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::sort(arg0,arg1,arg2) ;
    }


    void sortIdx_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2){
        return cv::sortIdx(arg0,arg1,arg2) ;
    }


    void split_wrapper(const cv::Mat& arg0,std::vector<cv::Mat>& arg1){
        return cv::split(arg0,arg1) ;
    }


    void sqrBoxFilter_wrapper(const cv::Mat& arg0,cv::Mat& arg1,int arg2,Size arg3,Point arg4,bool arg5,int arg6){
        return cv::sqrBoxFilter(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void sqrt_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::sqrt(arg0,arg1) ;
    }


    int startWindowThread_wrapper(){
        return cv::startWindowThread() ;
    }


    void subtract_wrapper(const cv::Mat& arg0,const cv::Mat& arg1,cv::Mat& arg2,const cv::Mat& arg3,int arg4){
        return cv::subtract(arg0,arg1,arg2,arg3,arg4) ;
    }


    Scalar sumElems_wrapper(const cv::Mat& arg0){
        return cv::sum(arg0) ;
    }


    double threshold_wrapper(const cv::Mat& arg0,cv::Mat& arg1,double arg2,double arg3,int arg4){
        return cv::threshold(arg0,arg1,arg2,arg3,arg4) ;
    }


    Scalar trace_wrapper(const cv::Mat& arg0){
        return cv::trace(arg0) ;
    }


    void transform_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2){
        return cv::transform(arg0,arg1,arg2) ;
    }


    void transpose_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::transpose(arg0,arg1) ;
    }


    void undistort_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2,const cv::Mat& arg3,const cv::Mat& arg4){
        return cv::undistort(arg0,arg1,arg2,arg3,arg4) ;
    }


    void undistortPoints_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2,const cv::Mat& arg3,const cv::Mat& arg4,const cv::Mat& arg5){
        return cv::undistortPoints(arg0,arg1,arg2,arg3,arg4,arg5) ;
    }


    void vconcat_wrapper(const std::vector<cv::Mat>& arg0,cv::Mat& arg1){
        return cv::vconcat(arg0,arg1) ;
    }


    int waitKey_wrapper(int arg0){
        return cv::waitKey(arg0) ;
    }


    void warpAffine_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2,Size arg3,int arg4,int arg5,const Scalar& arg6){
        return cv::warpAffine(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void warpPerspective_wrapper(const cv::Mat& arg0,cv::Mat& arg1,const cv::Mat& arg2,Size arg3,int arg4,int arg5,const Scalar& arg6){
        return cv::warpPerspective(arg0,arg1,arg2,arg3,arg4,arg5,arg6) ;
    }


    void watershed_wrapper(const cv::Mat& arg0,cv::Mat& arg1){
        return cv::watershed(arg0,arg1) ;
    }


    void finish_wrapper(){
        return cv::ocl::finish() ;
    }


    bool haveAmdBlas_wrapper(){
        return cv::ocl::haveAmdBlas() ;
    }


    bool haveAmdFft_wrapper(){
        return cv::ocl::haveAmdFft() ;
    }


    bool haveOpenCL_wrapper(){
        return cv::ocl::haveOpenCL() ;
    }


    void setUseOpenCL_wrapper(bool arg0){
        return cv::ocl::setUseOpenCL(arg0) ;
    }


    bool useOpenCL_wrapper(){
        return cv::ocl::useOpenCL() ;
    }

}

    EMSCRIPTEN_BINDINGS(testBinding) {
        
    emscripten::class_<cv::Algorithm>("Algorithm")
    
    ;

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

    emscripten::class_<cv::ml::StatModel>("ml_StatModel")
    
    ;

    function("Canny", select_overload<void(const cv::Mat&,cv::Mat&,double,double,int,bool)>(&Wrappers::Canny_wrapper));

    function("GaussianBlur", select_overload<void(const cv::Mat&,cv::Mat&,Size,double,double,int)>(&Wrappers::GaussianBlur_wrapper));

    function("HoughCircles", select_overload<void(const cv::Mat&,cv::Mat&,int,double,double,double,double,int,int)>(&Wrappers::HoughCircles_wrapper));

    function("HoughLines", select_overload<void(const cv::Mat&,cv::Mat&,double,double,int,double,double,double,double)>(&Wrappers::HoughLines_wrapper));

    function("HoughLinesP", select_overload<void(const cv::Mat&,cv::Mat&,double,double,int,double,double)>(&Wrappers::HoughLinesP_wrapper));

    function("HuMoments", select_overload<void(const Moments&,cv::Mat&)>(&Wrappers::HuMoments_wrapper));

    function("LUT", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::LUT_wrapper));

    function("Laplacian", select_overload<void(const cv::Mat&,cv::Mat&,int,int,double,double,int)>(&Wrappers::Laplacian_wrapper));

    function("Mahalanobis", select_overload<double(const cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::Mahalanobis_wrapper));

    function("PCABackProject", select_overload<void(const cv::Mat&,const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::PCABackProject_wrapper));

    function("PCACompute", select_overload<void(const cv::Mat&,cv::Mat&,cv::Mat&,int)>(&Wrappers::PCACompute_wrapper));

    function("PCACompute", select_overload<void(const cv::Mat&,cv::Mat&,cv::Mat&,double)>(&Wrappers::PCACompute_wrapper));

    function("PCAProject", select_overload<void(const cv::Mat&,const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::PCAProject_wrapper));

    function("PSNR", select_overload<double(const cv::Mat&,const cv::Mat&)>(&Wrappers::PSNR_wrapper));

    function("SVBackSubst", select_overload<void(const cv::Mat&,const cv::Mat&,const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::SVBackSubst_wrapper));

    function("SVDecomp", select_overload<void(const cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,int)>(&Wrappers::SVDecomp_wrapper));

    function("Scharr", select_overload<void(const cv::Mat&,cv::Mat&,int,int,int,double,double,int)>(&Wrappers::Scharr_wrapper));

    function("Sobel", select_overload<void(const cv::Mat&,cv::Mat&,int,int,int,int,double,double,int)>(&Wrappers::Sobel_wrapper));

    function("absdiff", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::absdiff_wrapper));

    function("accumulate", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::accumulate_wrapper));

    function("accumulateProduct", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::accumulateProduct_wrapper));

    function("accumulateSquare", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::accumulateSquare_wrapper));

    function("accumulateWeighted", select_overload<void(const cv::Mat&,cv::Mat&,double,const cv::Mat&)>(&Wrappers::accumulateWeighted_wrapper));

    function("adaptiveThreshold", select_overload<void(const cv::Mat&,cv::Mat&,double,int,int,int,double)>(&Wrappers::adaptiveThreshold_wrapper));

    function("add", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,const cv::Mat&,int)>(&Wrappers::add_wrapper));

    function("addWeighted", select_overload<void(const cv::Mat&,double,const cv::Mat&,double,double,cv::Mat&,int)>(&Wrappers::addWeighted_wrapper));

    function("applyColorMap", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::applyColorMap_wrapper));

    function("approxPolyDP", select_overload<void(const cv::Mat&,cv::Mat&,double,bool)>(&Wrappers::approxPolyDP_wrapper));

    function("arcLength", select_overload<double(const cv::Mat&,bool)>(&Wrappers::arcLength_wrapper));

    function("arrowedLine", select_overload<void(cv::Mat&,Point,Point,const Scalar&,int,int,int,double)>(&Wrappers::arrowedLine_wrapper));

    function("batchDistance", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,int,cv::Mat&,int,int,const cv::Mat&,int,bool)>(&Wrappers::batchDistance_wrapper));

    function("bilateralFilter", select_overload<void(const cv::Mat&,cv::Mat&,int,double,double,int)>(&Wrappers::bilateralFilter_wrapper));

    function("bitwise_and", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::bitwise_and_wrapper));

    function("bitwise_not", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::bitwise_not_wrapper));

    function("bitwise_or", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::bitwise_or_wrapper));

    function("bitwise_xor", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::bitwise_xor_wrapper));

    function("blur", select_overload<void(const cv::Mat&,cv::Mat&,Size,Point,int)>(&Wrappers::blur_wrapper));

    function("borderInterpolate", select_overload<int(int,int,int)>(&Wrappers::borderInterpolate_wrapper));

    function("boundingRect", select_overload<Rect(const cv::Mat&)>(&Wrappers::boundingRect_wrapper));

    function("boxFilter", select_overload<void(const cv::Mat&,cv::Mat&,int,Size,Point,bool,int)>(&Wrappers::boxFilter_wrapper));

    function("boxPoints", select_overload<void(RotatedRect,cv::Mat&)>(&Wrappers::boxPoints_wrapper));

    function("calcBackProject", select_overload<void(const std::vector<cv::Mat>&,const std::vector<int>&,const cv::Mat&,cv::Mat&,const std::vector<float>&,double)>(&Wrappers::calcBackProject_wrapper));

    function("calcCovarMatrix", select_overload<void(const cv::Mat&,cv::Mat&,cv::Mat&,int,int)>(&Wrappers::calcCovarMatrix_wrapper));

    function("calcHist", select_overload<void(const std::vector<cv::Mat>&,const std::vector<int>&,const cv::Mat&,cv::Mat&,const std::vector<int>&,const std::vector<float>&,bool)>(&Wrappers::calcHist_wrapper));

    function("cartToPolar", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,cv::Mat&,bool)>(&Wrappers::cartToPolar_wrapper));

    function("checkRange", select_overload<bool(const cv::Mat&,bool, Point*,double,double)>(&Wrappers::checkRange_wrapper), allow_raw_pointers());

    function("circle", select_overload<void(cv::Mat&,Point,int,const Scalar&,int,int,int)>(&Wrappers::circle_wrapper));

    function("clipLine", select_overload<bool(Rect,  Point&,  Point&)>(&Wrappers::clipLine_wrapper));

    function("compare", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,int)>(&Wrappers::compare_wrapper));

    function("compareHist", select_overload<double(const cv::Mat&,const cv::Mat&,int)>(&Wrappers::compareHist_wrapper));

    function("completeSymm", select_overload<void(cv::Mat&,bool)>(&Wrappers::completeSymm_wrapper));

    function("connectedComponents", select_overload<int(const cv::Mat&,cv::Mat&,int,int)>(&Wrappers::connectedComponents_wrapper));

    function("connectedComponentsWithStats", select_overload<int(const cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,int,int)>(&Wrappers::connectedComponentsWithStats_wrapper));

    function("contourArea", select_overload<double(const cv::Mat&,bool)>(&Wrappers::contourArea_wrapper));

    function("convertMaps", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,cv::Mat&,int,bool)>(&Wrappers::convertMaps_wrapper));

    function("convertScaleAbs", select_overload<void(const cv::Mat&,cv::Mat&,double,double)>(&Wrappers::convertScaleAbs_wrapper));

    function("convexHull", select_overload<void(const cv::Mat&,cv::Mat&,bool,bool)>(&Wrappers::convexHull_wrapper));

    function("convexityDefects", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::convexityDefects_wrapper));

    function("copyMakeBorder", select_overload<void(const cv::Mat&,cv::Mat&,int,int,int,int,int,const Scalar&)>(&Wrappers::copyMakeBorder_wrapper));

    function("cornerEigenValsAndVecs", select_overload<void(const cv::Mat&,cv::Mat&,int,int,int)>(&Wrappers::cornerEigenValsAndVecs_wrapper));

    function("cornerHarris", select_overload<void(const cv::Mat&,cv::Mat&,int,int,double,int)>(&Wrappers::cornerHarris_wrapper));

    function("cornerMinEigenVal", select_overload<void(const cv::Mat&,cv::Mat&,int,int,int)>(&Wrappers::cornerMinEigenVal_wrapper));

    function("cornerSubPix", select_overload<void(const cv::Mat&,cv::Mat&,Size,Size,TermCriteria)>(&Wrappers::cornerSubPix_wrapper));

    function("countNonZero", select_overload<int(const cv::Mat&)>(&Wrappers::countNonZero_wrapper));

    function("createCLAHE", select_overload<Ptr<CLAHE>(double,Size)>(&Wrappers::createCLAHE_wrapper));

    function("createHanningWindow", select_overload<void(cv::Mat&,Size,int)>(&Wrappers::createHanningWindow_wrapper));

    function("createLineSegmentDetector", select_overload<Ptr<LineSegmentDetector>(int,double,double,double,double,double,double,int)>(&Wrappers::createLineSegmentDetector_wrapper));

    function("cvtColor", select_overload<void(const cv::Mat&,cv::Mat&,int,int)>(&Wrappers::cvtColor_wrapper));

    function("dct", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::dct_wrapper));

    function("demosaicing", select_overload<void(const cv::Mat&,cv::Mat&,int,int)>(&Wrappers::demosaicing_wrapper));

    function("destroyAllWindows", select_overload<void()>(&Wrappers::destroyAllWindows_wrapper));

    function("destroyWindow", select_overload<void(const String&)>(&Wrappers::destroyWindow_wrapper));

    function("determinant", select_overload<double(const cv::Mat&)>(&Wrappers::determinant_wrapper));

    function("dft", select_overload<void(const cv::Mat&,cv::Mat&,int,int)>(&Wrappers::dft_wrapper));

    function("dilate", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&,Point,int,int,const Scalar&)>(&Wrappers::dilate_wrapper));

    function("distanceTransform", select_overload<void(const cv::Mat&,cv::Mat&,int,int,int)>(&Wrappers::distanceTransform_wrapper));

    function("distanceTransformWithLabels", select_overload<void(const cv::Mat&,cv::Mat&,cv::Mat&,int,int,int)>(&Wrappers::distanceTransformWithLabels_wrapper));

    function("divide", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,double,int)>(&Wrappers::divide_wrapper));

    function("divide", select_overload<void(double,const cv::Mat&,cv::Mat&,int)>(&Wrappers::divide_wrapper));

    function("drawContours", select_overload<void(cv::Mat&,const std::vector<cv::Mat>&,int,const Scalar&,int,int,const cv::Mat&,int,Point)>(&Wrappers::drawContours_wrapper));

    function("eigen", select_overload<bool(const cv::Mat&,cv::Mat&,cv::Mat&)>(&Wrappers::eigen_wrapper));

    function("ellipse", select_overload<void(cv::Mat&,Point,Size,double,double,double,const Scalar&,int,int,int)>(&Wrappers::ellipse_wrapper));

    function("ellipse", select_overload<void(cv::Mat&,const RotatedRect&,const Scalar&,int,int)>(&Wrappers::ellipse_wrapper));

    function("ellipse2Poly", select_overload<void(Point,Size,int,int,int,int, std::vector<Point>&)>(&Wrappers::ellipse2Poly_wrapper));

    function("equalizeHist", select_overload<void(const cv::Mat&,cv::Mat&)>(&Wrappers::equalizeHist_wrapper));

    function("erode", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&,Point,int,int,const Scalar&)>(&Wrappers::erode_wrapper));

    function("exp", select_overload<void(const cv::Mat&,cv::Mat&)>(&Wrappers::exp_wrapper));

    function("extractChannel", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::extractChannel_wrapper));

    function("fillConvexPoly", select_overload<void(cv::Mat&,const cv::Mat&,const Scalar&,int,int)>(&Wrappers::fillConvexPoly_wrapper));

    function("fillPoly", select_overload<void(cv::Mat&,const std::vector<cv::Mat>&,const Scalar&,int,int,Point)>(&Wrappers::fillPoly_wrapper));

    function("filter2D", select_overload<void(const cv::Mat&,cv::Mat&,int,const cv::Mat&,Point,double,int)>(&Wrappers::filter2D_wrapper));

    function("findContours", select_overload<void(cv::Mat&,std::vector<cv::Mat>&,cv::Mat&,int,int,Point)>(&Wrappers::findContours_wrapper));

    function("findNonZero", select_overload<void(const cv::Mat&,cv::Mat&)>(&Wrappers::findNonZero_wrapper));

    function("fitEllipse", select_overload<RotatedRect(const cv::Mat&)>(&Wrappers::fitEllipse_wrapper));

    function("fitLine", select_overload<void(const cv::Mat&,cv::Mat&,int,double,double,double)>(&Wrappers::fitLine_wrapper));

    function("flip", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::flip_wrapper));

    function("floodFill", select_overload<int(cv::Mat&,cv::Mat&,Point,Scalar, Rect*,Scalar,Scalar,int)>(&Wrappers::floodFill_wrapper), allow_raw_pointers());

    function("gemm", select_overload<void(const cv::Mat&,const cv::Mat&,double,const cv::Mat&,double,cv::Mat&,int)>(&Wrappers::gemm_wrapper));

    function("getAffineTransform", select_overload<Mat(const cv::Mat&,const cv::Mat&)>(&Wrappers::getAffineTransform_wrapper));

    function("getDefaultNewCameraMatrix", select_overload<Mat(const cv::Mat&,Size,bool)>(&Wrappers::getDefaultNewCameraMatrix_wrapper));

    function("getDerivKernels", select_overload<void(cv::Mat&,cv::Mat&,int,int,int,bool,int)>(&Wrappers::getDerivKernels_wrapper));

    function("getGaborKernel", select_overload<Mat(Size,double,double,double,double,double,int)>(&Wrappers::getGaborKernel_wrapper));

    function("getGaussianKernel", select_overload<Mat(int,double,int)>(&Wrappers::getGaussianKernel_wrapper));

    function("getOptimalDFTSize", select_overload<int(int)>(&Wrappers::getOptimalDFTSize_wrapper));

    function("getPerspectiveTransform", select_overload<Mat(const cv::Mat&,const cv::Mat&)>(&Wrappers::getPerspectiveTransform_wrapper));

    function("getRectSubPix", select_overload<void(const cv::Mat&,Size,Point2f,cv::Mat&,int)>(&Wrappers::getRectSubPix_wrapper));

    function("getRotationMatrix2D", select_overload<Mat(Point2f,double,double)>(&Wrappers::getRotationMatrix2D_wrapper));

    function("getStructuringElement", select_overload<Mat(int,Size,Point)>(&Wrappers::getStructuringElement_wrapper));

    function("getTextSize", select_overload<Size(const String&,int,double,int, int*)>(&Wrappers::getTextSize_wrapper), allow_raw_pointers());

    function("getTrackbarPos", select_overload<int(const String&,const String&)>(&Wrappers::getTrackbarPos_wrapper));

    function("getWindowProperty", select_overload<double(const String&,int)>(&Wrappers::getWindowProperty_wrapper));

    function("goodFeaturesToTrack", select_overload<void(const cv::Mat&,cv::Mat&,int,double,double,const cv::Mat&,int,bool,double)>(&Wrappers::goodFeaturesToTrack_wrapper));

    function("grabCut", select_overload<void(const cv::Mat&,cv::Mat&,Rect,cv::Mat&,cv::Mat&,int,int)>(&Wrappers::grabCut_wrapper));

    function("hconcat", select_overload<void(const std::vector<cv::Mat>&,cv::Mat&)>(&Wrappers::hconcat_wrapper));

    function("idct", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::idct_wrapper));

    function("idft", select_overload<void(const cv::Mat&,cv::Mat&,int,int)>(&Wrappers::idft_wrapper));

    function("imdecode", select_overload<Mat(const cv::Mat&,int)>(&Wrappers::imdecode_wrapper));

    function("imencode", select_overload<bool(const String&,const cv::Mat&, std::vector<uchar>&,const std::vector<int>&)>(&Wrappers::imencode_wrapper));

    function("imread", select_overload<Mat(const String&,int)>(&Wrappers::imread_wrapper));

    function("imreadmulti", select_overload<bool(const String&,std::vector<Mat>&,int)>(&Wrappers::imreadmulti_wrapper));

    function("imshow", select_overload<void(const String&,const cv::Mat&)>(&Wrappers::imshow_wrapper));

    function("imwrite", select_overload<bool(const String&,const cv::Mat&,const std::vector<int>&)>(&Wrappers::imwrite_wrapper));

    function("inRange", select_overload<void(const cv::Mat&,const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::inRange_wrapper));

    function("initUndistortRectifyMap", select_overload<void(const cv::Mat&,const cv::Mat&,const cv::Mat&,const cv::Mat&,Size,int,cv::Mat&,cv::Mat&)>(&Wrappers::initUndistortRectifyMap_wrapper));

    function("initWideAngleProjMap", select_overload<float(const cv::Mat&,const cv::Mat&,Size,int,int,cv::Mat&,cv::Mat&,int,double)>(&Wrappers::initWideAngleProjMap_wrapper));

    function("insertChannel", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::insertChannel_wrapper));

    function("integral", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::integral_wrapper));

    function("integral2", select_overload<void(const cv::Mat&,cv::Mat&,cv::Mat&,int,int)>(&Wrappers::integral2_wrapper));

    function("integral3", select_overload<void(const cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,int,int)>(&Wrappers::integral3_wrapper));

    function("intersectConvexConvex", select_overload<float(const cv::Mat&,const cv::Mat&,cv::Mat&,bool)>(&Wrappers::intersectConvexConvex_wrapper));

    function("invert", select_overload<double(const cv::Mat&,cv::Mat&,int)>(&Wrappers::invert_wrapper));

    function("invertAffineTransform", select_overload<void(const cv::Mat&,cv::Mat&)>(&Wrappers::invertAffineTransform_wrapper));

    function("isContourConvex", select_overload<bool(const cv::Mat&)>(&Wrappers::isContourConvex_wrapper));

    function("kmeans", select_overload<double(const cv::Mat&,int,cv::Mat&,TermCriteria,int,int,cv::Mat&)>(&Wrappers::kmeans_wrapper));

    function("line", select_overload<void(cv::Mat&,Point,Point,const Scalar&,int,int,int)>(&Wrappers::line_wrapper));

    function("linearPolar", select_overload<void(const cv::Mat&,cv::Mat&,Point2f,double,int)>(&Wrappers::linearPolar_wrapper));

    function("log", select_overload<void(const cv::Mat&,cv::Mat&)>(&Wrappers::log_wrapper));

    function("logPolar", select_overload<void(const cv::Mat&,cv::Mat&,Point2f,double,int)>(&Wrappers::logPolar_wrapper));

    function("magnitude", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::magnitude_wrapper));

    function("matchShapes", select_overload<double(const cv::Mat&,const cv::Mat&,int,double)>(&Wrappers::matchShapes_wrapper));

    function("matchTemplate", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,int,const cv::Mat&)>(&Wrappers::matchTemplate_wrapper));

    function("max", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::max_wrapper));

    function("mean", select_overload<Scalar(const cv::Mat&,const cv::Mat&)>(&Wrappers::mean_wrapper));

    function("meanStdDev", select_overload<void(const cv::Mat&,cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::meanStdDev_wrapper));

    function("medianBlur", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::medianBlur_wrapper));

    function("merge", select_overload<void(const std::vector<cv::Mat>&,cv::Mat&)>(&Wrappers::merge_wrapper));

    function("min", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&)>(&Wrappers::min_wrapper));

    function("minAreaRect", select_overload<RotatedRect(const cv::Mat&)>(&Wrappers::minAreaRect_wrapper));

    function("minEnclosingTriangle", select_overload<double(const cv::Mat&, OutputArray)>(&Wrappers::minEnclosingTriangle_wrapper));

    function("minMaxLoc", select_overload<void(const cv::Mat&, double*, double*, Point*, Point*,const cv::Mat&)>(&Wrappers::minMaxLoc_wrapper), allow_raw_pointers());

    function("mixChannels", select_overload<void(const std::vector<cv::Mat>&,InputOutputArrayOfArrays,const std::vector<int>&)>(&Wrappers::mixChannels_wrapper));

    function("moments", select_overload<Moments(const cv::Mat&,bool)>(&Wrappers::moments_wrapper));

    function("morphologyEx", select_overload<void(const cv::Mat&,cv::Mat&,int,const cv::Mat&,Point,int,int,const Scalar&)>(&Wrappers::morphologyEx_wrapper));

    function("moveWindow", select_overload<void(const String&,int,int)>(&Wrappers::moveWindow_wrapper));

    function("mulSpectrums", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,int,bool)>(&Wrappers::mulSpectrums_wrapper));

    function("mulTransposed", select_overload<void(const cv::Mat&,cv::Mat&,bool,const cv::Mat&,double,int)>(&Wrappers::mulTransposed_wrapper));

    function("multiply", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,double,int)>(&Wrappers::multiply_wrapper));

    function("namedWindow", select_overload<void(const String&,int)>(&Wrappers::namedWindow_wrapper));

    function("norm", select_overload<double(const cv::Mat&,int,const cv::Mat&)>(&Wrappers::norm_wrapper));

    function("norm", select_overload<double(const cv::Mat&,const cv::Mat&,int,const cv::Mat&)>(&Wrappers::norm_wrapper));

    function("normalize", select_overload<void(const cv::Mat&,cv::Mat&,double,double,int,int,const cv::Mat&)>(&Wrappers::normalize_wrapper));

    function("patchNaNs", select_overload<void(cv::Mat&,double)>(&Wrappers::patchNaNs_wrapper));

    function("perspectiveTransform", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::perspectiveTransform_wrapper));

    function("phase", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,bool)>(&Wrappers::phase_wrapper));

    function("phaseCorrelate", select_overload<Point2d(const cv::Mat&,const cv::Mat&,const cv::Mat&, double*)>(&Wrappers::phaseCorrelate_wrapper), allow_raw_pointers());

    function("pointPolygonTest", select_overload<double(const cv::Mat&,Point2f,bool)>(&Wrappers::pointPolygonTest_wrapper));

    function("polarToCart", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,cv::Mat&,bool)>(&Wrappers::polarToCart_wrapper));

    function("polylines", select_overload<void(cv::Mat&,const std::vector<cv::Mat>&,bool,const Scalar&,int,int,int)>(&Wrappers::polylines_wrapper));

    function("pow", select_overload<void(const cv::Mat&,double,cv::Mat&)>(&Wrappers::pow_wrapper));

    function("preCornerDetect", select_overload<void(const cv::Mat&,cv::Mat&,int,int)>(&Wrappers::preCornerDetect_wrapper));

    function("putText", select_overload<void(cv::Mat&,const String&,Point,int,double,Scalar,int,int,bool)>(&Wrappers::putText_wrapper));

    function("pyrDown", select_overload<void(const cv::Mat&,cv::Mat&,const Size&,int)>(&Wrappers::pyrDown_wrapper));

    function("pyrMeanShiftFiltering", select_overload<void(const cv::Mat&,cv::Mat&,double,double,int,TermCriteria)>(&Wrappers::pyrMeanShiftFiltering_wrapper));

    function("pyrUp", select_overload<void(const cv::Mat&,cv::Mat&,const Size&,int)>(&Wrappers::pyrUp_wrapper));

    function("randShuffle", select_overload<void(cv::Mat&,double,RNG*)>(&Wrappers::randShuffle_wrapper), allow_raw_pointers());

    function("randn", select_overload<void(cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::randn_wrapper));

    function("randu", select_overload<void(cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::randu_wrapper));

    function("rectangle", select_overload<void(cv::Mat&,Point,Point,const Scalar&,int,int,int)>(&Wrappers::rectangle_wrapper));

    function("reduce", select_overload<void(const cv::Mat&,cv::Mat&,int,int,int)>(&Wrappers::reduce_wrapper));

    function("remap", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&,const cv::Mat&,int,int,const Scalar&)>(&Wrappers::remap_wrapper));

    function("repeat", select_overload<void(const cv::Mat&,int,int,cv::Mat&)>(&Wrappers::repeat_wrapper));

    function("resize", select_overload<void(const cv::Mat&,cv::Mat&,Size,double,double,int)>(&Wrappers::resize_wrapper));

    function("resizeWindow", select_overload<void(const String&,int,int)>(&Wrappers::resizeWindow_wrapper));

    function("rotatedRectangleIntersection", select_overload<int(const RotatedRect&,const RotatedRect&,cv::Mat&)>(&Wrappers::rotatedRectangleIntersection_wrapper));

    function("scaleAdd", select_overload<void(const cv::Mat&,double,const cv::Mat&,cv::Mat&)>(&Wrappers::scaleAdd_wrapper));

    function("sepFilter2D", select_overload<void(const cv::Mat&,cv::Mat&,int,const cv::Mat&,const cv::Mat&,Point,double,int)>(&Wrappers::sepFilter2D_wrapper));

    function("setIdentity", select_overload<void(cv::Mat&,const Scalar&)>(&Wrappers::setIdentity_wrapper));

    function("setTrackbarMax", select_overload<void(const String&,const String&,int)>(&Wrappers::setTrackbarMax_wrapper));

    function("setTrackbarPos", select_overload<void(const String&,const String&,int)>(&Wrappers::setTrackbarPos_wrapper));

    function("setWindowProperty", select_overload<void(const String&,int,double)>(&Wrappers::setWindowProperty_wrapper));

    function("setWindowTitle", select_overload<void(const String&,const String&)>(&Wrappers::setWindowTitle_wrapper));

    function("solve", select_overload<bool(const cv::Mat&,const cv::Mat&,cv::Mat&,int)>(&Wrappers::solve_wrapper));

    function("solveCubic", select_overload<int(const cv::Mat&,cv::Mat&)>(&Wrappers::solveCubic_wrapper));

    function("solvePoly", select_overload<double(const cv::Mat&,cv::Mat&,int)>(&Wrappers::solvePoly_wrapper));

    function("sort", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::sort_wrapper));

    function("sortIdx", select_overload<void(const cv::Mat&,cv::Mat&,int)>(&Wrappers::sortIdx_wrapper));

    function("split", select_overload<void(const cv::Mat&,std::vector<cv::Mat>&)>(&Wrappers::split_wrapper));

    function("sqrBoxFilter", select_overload<void(const cv::Mat&,cv::Mat&,int,Size,Point,bool,int)>(&Wrappers::sqrBoxFilter_wrapper));

    function("sqrt", select_overload<void(const cv::Mat&,cv::Mat&)>(&Wrappers::sqrt_wrapper));

    function("startWindowThread", select_overload<int()>(&Wrappers::startWindowThread_wrapper));

    function("subtract", select_overload<void(const cv::Mat&,const cv::Mat&,cv::Mat&,const cv::Mat&,int)>(&Wrappers::subtract_wrapper));

    function("sumElems", select_overload<Scalar(const cv::Mat&)>(&Wrappers::sumElems_wrapper));

    function("threshold", select_overload<double(const cv::Mat&,cv::Mat&,double,double,int)>(&Wrappers::threshold_wrapper));

    function("trace", select_overload<Scalar(const cv::Mat&)>(&Wrappers::trace_wrapper));

    function("transform", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&)>(&Wrappers::transform_wrapper));

    function("transpose", select_overload<void(const cv::Mat&,cv::Mat&)>(&Wrappers::transpose_wrapper));

    function("undistort", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::undistort_wrapper));

    function("undistortPoints", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&,const cv::Mat&,const cv::Mat&,const cv::Mat&)>(&Wrappers::undistortPoints_wrapper));

    function("vconcat", select_overload<void(const std::vector<cv::Mat>&,cv::Mat&)>(&Wrappers::vconcat_wrapper));

    function("waitKey", select_overload<int(int)>(&Wrappers::waitKey_wrapper));

    function("warpAffine", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&,Size,int,int,const Scalar&)>(&Wrappers::warpAffine_wrapper));

    function("warpPerspective", select_overload<void(const cv::Mat&,cv::Mat&,const cv::Mat&,Size,int,int,const Scalar&)>(&Wrappers::warpPerspective_wrapper));

    function("watershed", select_overload<void(const cv::Mat&,cv::Mat&)>(&Wrappers::watershed_wrapper));

    function("finish", select_overload<void()>(&Wrappers::finish_wrapper));

    function("haveAmdBlas", select_overload<bool()>(&Wrappers::haveAmdBlas_wrapper));

    function("haveAmdFft", select_overload<bool()>(&Wrappers::haveAmdFft_wrapper));

    function("haveOpenCL", select_overload<bool()>(&Wrappers::haveOpenCL_wrapper));

    function("setUseOpenCL", select_overload<void(bool)>(&Wrappers::setUseOpenCL_wrapper));

    function("useOpenCL", select_overload<bool()>(&Wrappers::useOpenCL_wrapper));

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

namespace Utils{
    cv::Mat imreadwrapper(const std::string& file , int mode){
        cv::String str(file.data() ) ;
        return cv::imread(  str , mode ) ;
    }

    //std::string data( const cv::Mat& mat ) {
    //    std::string ret ( (char*) mat.data , mat.total()  * mat.elemSize() ) ;r
    //    retun ret ;
    //}

    void u8data(const cv::Mat& mat, emscripten::val onComplete) {
          onComplete(emscripten::memory_view<uint8_t>(mat.total() * mat.elemSize(), (unsigned char*) mat.data ));
    }

    void u16data(const cv::Mat& mat, emscripten::val onComplete) {
          onComplete(emscripten::memory_view<uint16_t>(mat.total() * mat.elemSize() / 2 , (unsigned short*) mat.data ));
    }




}

EMSCRIPTEN_BINDINGS(Utils) {

    register_vector<int>("IntegerVector");
    register_vector<char>("CharVector");
    register_vector<unsigned>("VectorUnsigned");
    register_vector<unsigned char>("VectorUnsignedChar");
    register_vector<std::string>("StringVector");
    register_vector<emscripten::val>("EmValVector");
    register_vector<float>("FloatVector");
    register_vector<std::vector<int>>("IntegerVectorVector");
    register_vector<std::vector<Point>>("PointVectorVector");
    register_vector<cv::Vec4i>("Vec4iVector");
    register_vector<cv::Mat>("MatVector");

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
        .function("data", &Utils::u8data)
        .function("u16data", &Utils::u16data)
        .function("at_uchar" , select_overload<unsigned char& (int,int,int)>(&cv::Mat::at<unsigned char>) )
        .function("at_ushort" , select_overload<unsigned short& (int,int,int)>(&cv::Mat::at<unsigned short>) )

    ;


    emscripten::class_<cv::Vec<int,4>>("Vec4i")
        .constructor<>()
        .constructor<int,int,int,int>()
    ;

    emscripten::class_<cv::String>("String")
        .function("size", select_overload<size_t()const>(&cv::String::size))
        .function("length", select_overload<size_t()const>(&cv::String::length))
        .function("empty", select_overload<bool()const>(&cv::String::empty))
        .function("toLowerCase",select_overload<String()const>(&cv::String::toLowerCase))
        .function("compareString",select_overload<int(const String&)const>(&cv::String::compare))
        .function("compare",select_overload<int(const char* )const>(&cv::String::compare),allow_raw_pointers() )
        .constructor< const std::string& >()
        .constructor<>()
    ;

    emscripten::class_<cv::Size_<int>> ("Size")
        .constructor<> ()
        .constructor<int, int> ()
        .constructor<const cv::Size_<int>& > ()
        .property("width" , &cv::Size_<int>::width )
        .property("height" , &cv::Size_<int>::height )
    ;

    emscripten::class_<cv::Point_<int>> ("Point")
        .constructor<> ()
        .constructor<int, int> ()
        .constructor<const cv::Point_<int>& > ()
        .property("x" , &cv::Point_<int>::x )
        .property("y" , &cv::Point_<int>::y )
    ;

    emscripten::class_<cv::Rect_<int>> ("Rect")
        .constructor<> ()
        .constructor<const cv::Point_<int>& , const cv::Size_<int>&  > ()
        .constructor<int, int,int, int> ()
        .constructor<const cv::Rect_<int>& > ()
        .property("x" , &cv::Rect_<int>::x )
        .property("y" , &cv::Rect_<int>::y )
        .property("width" , &cv::Rect_<int>::width )
        .property("height" , &cv::Rect_<int>::height )
    ;

    emscripten::class_<cv::Scalar_<double>> ("Scalar")
        .constructor<> ()
        .constructor<double> ()
        .constructor<double,double> ()
        .constructor<double,double,double> ()
        .constructor<double,double,double,double> ()
        .class_function("all" , &cv::Scalar_<double>::all )
        .function("isReal", select_overload<bool()const>(&cv::Scalar_<double>::isReal))
    ;


}
