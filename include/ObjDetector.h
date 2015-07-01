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
    struct Result
    {
        cv::Rect roi;
        double confidence;
    };

    ObjDetector();	///< basic constructor. The parameters are not initialized.
	ObjDetector(std::string yamlConfigFile) throw (std::runtime_error);	///< constructor. The parameters are inizialized using the file passed in input.
	~ObjDetector();
	std::vector<Result> detect(cv::Mat& frame);	///< performs object detection on the frame in input.
	inline void init(std::string yamlConfigFile) 
		throw (std::runtime_error) /// initializes the parameters using the file in input. 
		{ params_.loadFromFile(yamlConfigFile); }; 
	
	cv::Mat currFrame; ///< last processed frame

private:
	void init() throw (std::runtime_error);	///< initializes the classifiers
	std::vector<ObjDetector::Result> verifyROIs(cv::Mat& frame, std::vector<cv::Rect>& rois); ///< Filters candidate ROIs using SVM

	DetectionParams params_;			///< parameters of the detector
	cv::CascadeClassifier cascade_;		///< cascade classifier
	cv::HOGDescriptor hog_;				///< hog feature extractor
	struct svm_model* model_;			///< svm classifier
};

#endif