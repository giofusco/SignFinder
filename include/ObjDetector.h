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

//#include <time.h>
#include <limits.h>
#include "DetectionParams.h"
#include "svm.h"

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

	ObjDetector(std::string resourceLocation) throw (std::runtime_error);	///< constructor. The parameters are inizialized using the file passed in input.
	~ObjDetector();
	std::vector<DetectionInfo> detect(cv::Mat& frame);	///< performs object detection on the frame in input.
	
    //void dumpStage1(std::string prefix) const; ///< saves ROIs coming from the first stage to disk
	//void dumpStage2(); ///< saves verified ROIs to disk
    //cv::Mat currFrame; ///< last processed frame

    bool isInit() const {return init_; }
    
#ifndef NDEBUG
    inline const std::vector<cv::Rect>& getFirstStageResults() const
    {
        return rois_;
    }
#endif
private:
    ObjDetector(const ObjDetector&) = delete;
    ObjDetector& operator=(const ObjDetector&) = delete;
	std::vector<ObjDetector::DetectionInfo> verifyROIs(cv::Mat& frame, std::vector<cv::Rect>& rois) const throw (std::runtime_error); ///< Filters candidate ROIs using SVM

	const DetectionParams params_;			///< parameters of the detector
	mutable cv::CascadeClassifier cascade_;		///< cascade classifier
	const cv::HOGDescriptor hog_;				///< hog feature extractor
	const svm_model* model_;			///< svm classifier
	bool init_;

	std::vector<cv::Rect> rois_;
    //std::vector<ObjDetector::DetectionInfo> result_;

    mutable int counter_;
};

#endif