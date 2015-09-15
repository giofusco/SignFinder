/*
Copyright 2015 The Smith-Kettlewell Eye Research Institute
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef OBJ_DETECTOR_H
#define OBJ_DETECTOR_H

#include <time.h>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/core/core.hpp>
#include "DetectionParams.h"

/** \class ObjDetector
*  \brief Two-stages object detector.
*  \details This class defines a two stage classifier for object detection. 
*	The first stage consists in a multiscale Adaboost Cascade + LBP descriptor to generate candidate ROIs.
*	The second stage is an SVM trained using HOG desciptor. It confirms or rejects candidate ROIs detected in the first stage.
*/

class ObjDetector
{
public:
    struct DetectionInfo	
    {
        cv::Rect roi;
        double confidence;
    }; ///< Structure containing detection information.
	   ///< It contains the ROI in the image where the detection occurred and the corresponging confidence value (SVM likelihood) assigned by the second stage classifier

    ObjDetector();	///< basic constructor. The parameters are not initialized.
    ObjDetector(const std::string& yamlConfigFile, const std::string& classifiersFolder=std::string()) throw(std::runtime_error); ///< constructor. The parameters are inizialized using the file passed in input and the specified classifiers folder.
	~ObjDetector();
    std::vector<DetectionInfo> detect(cv::Mat& frame, bool doTrack=true) throw (std::runtime_error);	///< performs object detection on the frame in input.
	std::vector<DetectionInfo> detect(cv::Mat& frame, double& FPS, bool doTrack=true) throw (std::runtime_error);	///< performs object detection on the frame in input and returns the frame rate.
    void init(const std::string& yamlConfigFile, const std::string& classifiersFolder=std::string()) throw (std::runtime_error); ///< initializes the parameters using the file in input.
	
    inline std::vector<cv::Rect> getStage1Rois() const { return rois_;}
    std::vector<DetectionInfo> getStage2Rois() const;
	void dumpStage1(std::string prefix); ///< saves ROIs coming from the first stage to disk
	void dumpStage2(std::string prefix); ///< saves ROIs coming from the second stage to disk
	//void dumpStage2(); ///< saves verified ROIs to disk
	cv::Mat currFrame; ///< last processed frame

private:

	ObjDetector(const ObjDetector& that) = delete; //disable copy constructor

	void init() throw (std::runtime_error);	///< initializes the classifiers

//	DetectionParams params_;			//< parameters of the detector
//	cv::CascadeClassifier cascade_;		//< cascade classifier
//	cv::HOGDescriptor hog_;				//< hog feature extractor
//	struct svm_model* model_;			//< svm classifier
	bool init_;

    class CascadeDetector;  //< first stage detector, LBP + Adaboost cascade
    class SVMClassifier;      //< second stage detector, HoG + SVM
    std::unique_ptr<CascadeDetector> pCascadeDetector;  //< ptr to first stage detector
    std::unique_ptr<SVMClassifier> pSVMClassifier;      //< ptr to second stage detector
    
    DetectionParams params_;
    //parameters
    //const float svmThreshold_;
    //const int maxAgePreConfirmation_;
    //const int maxAgePostConfirmation_;
    //const int nConfirm_;
    
    struct TrackingInfo
    {
        cv::Rect roi;
        double confidence;
        int age;
        int nTimesSeen;
    };
    
    cv::Mat prevFrame_;
    
    std::vector<cv::Rect> rois_;        //< first stage outputs
    std::vector<TrackingInfo> secondStageOutputs_;  //< second stage outputs, objects that are potentially being tracked
    //std::vector<ObjDetector::DetectionInfo> result_;
    //std::vector<ObjDetector::DetectionInfo> allResult_; ///< used for debug/ROC experiments, will be removed in final release

    time_t start_;
	int counter_;
};

#endif