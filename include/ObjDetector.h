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

#include <opencv2/core/core.hpp>
#include <memory>
#include <string>

/** \class ObjDetector
*  \brief Two-stages object detector.
*  \details This class defines a two stage classifier for object detection. 
*	The first stage consists in a multiscale Adaboost Cascade + LBP descriptor to generate candidate ROIs.
*	The second stage is an SVM trained using HOG desciptor. It confirms or rejects candidate ROIs detected in the first stage.
*/

class ObjDetector
{
public:
    /// Structure containing detection information.
    /// It contains the ROI in the image where the detection occurred and the corresponging confidence value (SVM likelihood) assigned by the second stage classifier
    struct DetectionInfo
    {
        cv::Rect roi;
        double confidence;
    };

    /// Parameters for the object detector
    struct Parameters
    {
        std::string cascadeFileName;	///< Adaboost cascade classifier filename
        std::string svmModelFileName;	///< SVM model filename
        
        cv::Size hogWinSize;		///< windows size for HOG descriptor
        cv::Size cascadeMinWin;		///< min window size for multi-scale detection
        cv::Size cascadeMaxWin;		///< max window size for multi-scale detection
        
        float cascadeScaleFactor;	///< multiscale detection scaling factor
        float SVMThreshold;			///< theshold for rejection of candidate ROIs
        
        Parameters();
    };

    typedef std::unique_ptr<ObjDetector> Ptr;
    static Ptr Create(const Parameters& params) throw (std::runtime_error);
    
    /// Dtor
	~ObjDetector();
    
    /*!
     * Use 2-Stage object detector on the input frame.
     * @param[in] frame
     * @return a vector of DetectionInfo containing information about the verified detections
     */
	std::vector<DetectionInfo> detect(const cv::Mat& frame);
	
#ifndef NDEBUG
    inline const std::vector<cv::Rect>& getFirstStageResults() const
    {
        return rois_;
    }
#endif
private:
    /// Default Ctor
    ObjDetector() noexcept;

    /// Delete copy ctor and assignment operator
    ObjDetector(const ObjDetector&) = delete;
    ObjDetector& operator=(const ObjDetector&) = delete;
    
    class CascadeDetector;  //< first stage detector, LBP + Adaboost cascade
    class SVMDetector;      //< second stage detector, HoG + SVM
    std::unique_ptr<CascadeDetector> pCascadeDetector;  //< ptr to first stage detector
    std::unique_ptr<SVMDetector> pSVMDetector;          //< ptr to second stage detector
    
    std::vector<cv::Rect> rois_;        //< rois detected by the first stage
};

#endif