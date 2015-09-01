#include <jni.h>
#include <iostream>
#include <string>
#include <exception>


#include <iomanip>
#include <locale>
#include <sstream>
#include <string> // this should be already included in <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "svm.h"
#include "DetectionParams.h"
#include "ObjDetector.h"
#include "wicab_utils.h"

using namespace std;
using namespace cv;

Mat rgb, gray;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)

ObjDetector detector;
int created = 0;
double fps = 0;

string num2str(float num);
void rgb2argb(unsigned char *bgra, int *argb, int sz);
void gray2argb(unsigned char *gray, int *argb, int sz);
extern "C" {

JNIEXPORT int JNICALL Java_org_ski_wicablib_Detector_setClassifier(JNIEnv *env,
		jobject thisObj, jstring yamlConfigFile, jstring classifiersFolder) {
	const char *config = env->GetStringUTFChars(yamlConfigFile, NULL);
	const char *folder = env->GetStringUTFChars(classifiersFolder, NULL);

	try{
		detector.init(config,folder);
		created = 1;
	}
	catch (std::exception& e){
		std::cout << e.what() << '\n';
		created = 0;
	}
	env->ReleaseStringUTFChars(yamlConfigFile, config);  // release resources
	env->ReleaseStringUTFChars(classifiersFolder, folder);  // release resources
	return created;
}

JNIEXPORT int JNICALL Java_org_ski_wicablib_Detector_detect(JNIEnv *env,
		jobject thisObj,jlong addrYuv,jintArray argb) {
	if (created==0) return -1;
	Mat &yuv = *(Mat *) addrYuv;
	int w = yuv.cols, h = yuv.rows*2/3;
	rgb.create(h,w, CV_8UC(3));
	gray.create(h,w, CV_8UC1);
	cvtColor(yuv,rgb,CV_YUV2RGB_NV21);
	cvtColor(yuv,gray,CV_YUV2GRAY_NV21);

	cv::vector<ObjDetector::DetectionInfo> res = detector.detect(rgb, fps);
//	//plotting ROIs and confidence values
	for (int i = 0; i < res.size(); i++) {
		rectangle(detector.currFrame, res[i].roi, cv::Scalar(255, 0, 255), 2);
		putText(detector.currFrame, "p=" + num2str(res[i].confidence), res[i].roi.br(),
				CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
	}
	putText(detector.currFrame, "fps = " + num2str(fps), Point(200,300),
			CV_FONT_HERSHEY_PLAIN, 3.0, cv::Scalar(0, 255, 0));

	jint *i_argb = env->GetIntArrayElements(argb, 0);
	rgb2argb(detector.currFrame.data,i_argb, w*h);
    env->ReleaseIntArrayElements(argb, i_argb, 0);
	return res.size();

}

}
void rgb2argb(unsigned char *bgra, int *argb, int sz){
	for (int i=0,j=0; i<sz; i++,j+=3){
		argb[i] = 0xff000000|(bgra[j]&0xff)<<16|(bgra[j+1]&0xff)<<8|(bgra[j+2]&0xff);
	}
}
void gray2argb(unsigned char *gray, int *argb, int sz){

	for (int i=0; i<sz; i++){
		argb[i] = 0xff000000|(gray[i]&0xff)<<16|(gray[i]&0xff)<<8|(gray[i]&0xff);
	}
}

string num2str(float num){
	ostringstream Convert;
	Convert << fixed << setprecision(1) << num; // Use some manipulators
	return Convert.str(); // Give the result to the string
}

