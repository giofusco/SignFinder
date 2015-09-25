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

/** @class ObjDetector
*   @brief Two-stages object detector.
*   @details This class defines a two stage classifier for object detection.
*	The first stage consists in a multiscale Adaboost Cascade + LBP descriptor to generate candidate ROIs.
*	The second stage is an SVM trained using HOG desciptor. It confirms or rejects candidate ROIs detected in the first stage.
*/
class ObjDetector
{
public:
    /// Structure containing detection information.
    struct DetectionInfo	
    {
        cv::Rect roi;           ///< detection location
        double confidence;      ///< detection confidence as estimated by the SVM
    };

    /// Default constructor. The parameters are not initialized.
    ObjDetector();
    
    /// constructor. The parameters are inizialized using the file passed in input and the specified classifiers folder.
    /// @param[in] yamlConfigFile config file used to load parameters from
    /// @param[in] classifiersFolder location of the classifier files (if not available in yamlConfigFile. if classifierFolder is empty, then the locations must be provided in the yamlConfigFile
    /// @throw runtime_error if there is any problem reading either the config file or the classifier files.
    ObjDetector(const std::string& yamlConfigFile, const std::string& classifiersFolder=std::string()) throw(std::runtime_error);
    
    /// Dtor
	~ObjDetector();
    
    /// detects SIGNs using the detectors initialized via the vonfiguraion file
    /// @param[in] frame input image
    /// @param[in] doTrack whether to use tracking (default mode). When no tracking is used each frame is considered individually. This is generally less efficient, and may lead to more false alarms.
    /// @return a set of detections
    /// TODO: even if doTrack is false, we can use tracking to see which detections may correspond to which previous detections
    /// @throw std::runtime_error if the classifiers have not been successfully initialized
    std::vector<DetectionInfo> detect(cv::Mat& frame, bool doTrack=true) throw (std::runtime_error);
    
    /// detects SIGNs using the detectors initialized via the vonfiguraion file
    /// @param[in] frame input image
    /// @param[out[ FPS upon return containst the long-term average frames/second used for profiling
    /// @param[in] doTrack whether to use tracking (default mode). When no tracking is used each frame is considered individually. This is generally less efficient, and may lead to more false alarms.
    /// @return a set of detections
    /// TODO: even if doTrack is false, we can use tracking to see which detections may correspond to which previous detections
    /// @throw std::runtime_error if the classifiers have not been successfully initialized
	std::vector<DetectionInfo> detect(cv::Mat& frame, double& FPS, bool doTrack=true) throw (std::runtime_error);
    
    /// initializes the parameters using the file in input.
    /// @param[in] yamlConfigFile config file used to load parameters from
    /// @param[in] classifiersFolder location of the classifier files (if not available in yamlConfigFile. if classifierFolder is empty, then the locations must be provided in the yamlConfigFile
    /// @throw runtime_error if there is any problem reading either the config file or the classifier files.
    void init(const std::string& yamlConfigFile, const std::string& classifiersFolder=std::string()) throw (std::runtime_error);
	
    /// For debugging only
    /// @return the outputs of the first stage (cascade) classifier
    inline std::vector<cv::Rect> getStage1Rois() const { return rois_;}
    
    /// For debugging only
    /// @return the outputs of the second stage (svm) classifier
    std::vector<DetectionInfo> getStage2Rois() const;
    
    /// saves ROIs coming from the first stage to disk
    /// @param[in] prefix prefix of the file names to use when saving first stage results.
	void dumpStage1(std::string prefix);
    
    /// saves ROIs coming from the second stage to disk
    /// @param[in] prefix prefix of the file names to use hen saving second stage results.G
    void dumpStage2(std::string prefix);
	
    cv::Mat currFrame; ///< last processed frame

private:

	ObjDetector(const ObjDetector& that) = delete; //disable copy constructor

	void init() throw (std::runtime_error);	///< initializes the classifiers

	bool init_;

    class CascadeDetector;  //< first stage detector, LBP + Adaboost cascade
    class SVMClassifier;      //< second stage detector, HoG + SVM
    std::unique_ptr<CascadeDetector> pCascadeDetector;  //< ptr to first stage detector
    std::unique_ptr<SVMClassifier> pSVMClassifier;      //< ptr to second stage detector
    
    DetectionParams params_;
    
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

    time_t start_;
	int counter_;
};

#endif