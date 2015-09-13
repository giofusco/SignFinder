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
  
Author: Giovanni Fusco - giofusco@ski.org

*/

#include "ObjDetector.h"
#include "svm.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// Ctor with default parameters
ObjDetector::Parameters::Parameters():
cascadeScaleFactor(1.1f),
SVMThreshold(.5f)
{}

/// @class cascade detector using lbp features. used as first stage detector
class ObjDetector::CascadeDetector
{
public:
    
    /// Ctor
    /// @param[in] cascadeFileName name of file to load the cascade from
    /// @param[in] minWinSize minimum size of the scanning window
    /// @param[in] maxWinSize maximum size of the scanning window
    /// @param[in] scaleFactor scale factor to use for multi-scale detection
    /// @throw std::runtime_error if unable to allocate memory of read the cascade file
    CascadeDetector(const std::string& cascadeFileName, const cv::Size& minWinSize, const cv::Size& maxWinSize, float scaleFactor) throw (std::runtime_error):
    minSz_(minWinSize),
    maxSz_(maxWinSize),
    scaleFactor_(scaleFactor),
    cascade_(cascadeFileName)
    {
        // check that cascade detector was loaded successfully
        if ( cascade_.empty() )
        {
            throw std::runtime_error("CascadeDetector :: Unable to load cascade detector from file " + cascadeFileName);
        }
    }
    
    /// Dtor
    ~CascadeDetector() = default;

    /*!
     * First stage of cascade classifier
     * @param[in] frame frame to process
     * @return a vector of detections candidates
     */
    std::vector<cv::Rect> detect(const cv::Mat& frame) const
    {
        std::vector<cv::Rect> rois;
        cascade_.detectMultiScale( frame, rois, scaleFactor_, 0, 0, minSz_, maxSz_ );
        
        groupRectangles(rois, 1);
        return rois;
    }
private:
    const cv::Size minSz_;  //< min win size
    const cv::Size maxSz_;  //< max win size
    const float scaleFactor_;     //< scale factor
    mutable cv::CascadeClassifier cascade_;		///< cascade classifier, mutable since cv::CascadeClassifier::detectMultiScale is not const, but does not change internal state of ObjDetector
};  // ObjDetector::CascadeDetector


/// @class svm detector using HoG features, used as second stage classifier
class ObjDetector::SVMDetector
{
public:
    /// Ctor
    /// @param[in] svmModelFileName name of file to load the svm model from
    /// @param[in] hogWinSize size of the window to calculate HoG
    /// @param[in] svmThreshold detection threshold
    /// @throw std::runtime_error if unable to allocate memory of read the cascade file
    SVMDetector(const std::string& svmModelFileName, const cv::Size& hogWinSize, float svmThreshold) throw (std::runtime_error):
    hogWinSz_(hogWinSize),
    svmThreshold_(svmThreshold),
    pModel_(svm_load_model(svmModelFileName.c_str())),
    hog_(hogWinSize,            //winSize
         cv::Size(16,16),       //blockSize
         cv::Size(4,4),         //blockStride
         cv::Size(8,8),         //cellSize
         9,                     //nbins
         1,                     //derivAperture
         -1,                    //winSigma,
         cv::HOGDescriptor::L2Hys,  //histogramNormType,
         .2,                    //L2HysThreshold,
         true,                  //gammaCorrection
         1                      //nLevels
    )
    {
        //check that svm model was loaded successfully
        if ( !pModel_ )
        {
            throw std::runtime_error("SVMDetector :: Unable to load svm model from file " + svmModelFileName);
        }
    }
    
    /// Dtor
    ~SVMDetector() = default;
    
    /*!
     * Verifies the ROIs detected in the first stage using SVM + HOG
     * @param[in] frame frame to process
     * @param[in] rois vector containing the candidate ROIs
     * @return a vector of detections that passed the verification
     * @throw runtime error if unable to allocate memory for this stage
     */
    std::vector<ObjDetector::DetectionInfo> detect(const cv::Mat& frame, const std::vector<cv::Rect>& rois) const
    {
        std::vector<ObjDetector::DetectionInfo> result;
        double prob_est[2];
        
        //TODO: The index is set from scratch for each patch. If descriptor sizes are the same for each window, the next two can be optimized by setting it once as member variables
        std::vector<float> desc;
        std::vector<svm_node> x;
        
        for (const auto& roi : rois)
        {
            //use svm to classify the rois
            cv::Mat patch(frame, roi);
            
            cv::Mat res(1, 1, CV_32FC1);
            cv::resize(patch, patch, hogWinSz_);
            hog_.compute(patch, desc);
            x.resize(desc.size() + 1);
            
            for (int d = 0; d < desc.size(); d++){
                x[d].index = d + 1;  // Index starts from 1; Pre-computed kernel starts from 0
                x[d].value = desc[d];
            }
            
            x[desc.size()].index = -1;
            patch.release();
            res.release();
            desc.clear();
            
            svm_predict_probability(pModel_.get(), x.data(), prob_est);
            
            if ( (prob_est[0] > svmThreshold_) )
                result.push_back({ roi, prob_est[0] });
        }
        return result;
    }
private:
    const cv::Size hogWinSz_;                   //< min win size
    const float svmThreshold_;                  //< scale factor
    const std::unique_ptr<svm_model> pModel_;   //< svm model
    const cv::HOGDescriptor hog_;				//< hog feature extractor
};  // ObjDetector::SVMDetector


ObjDetector::Ptr ObjDetector::Create(const Parameters& params) throw (std::runtime_error)
{
    Ptr handle(new ObjDetector());
    if (!handle)
    {
        throw std::runtime_error("ObjDetector :: Unable to allocate memory for object detector.");
    }
    
    try
    {
        // create cascade detector
        handle->pCascadeDetector = std::unique_ptr<CascadeDetector>( new CascadeDetector(params.cascadeFileName, params.cascadeMinWin, params.cascadeMaxWin, params.cascadeScaleFactor) );
        
        // create svm
        handle->pSVMDetector = std::unique_ptr<SVMDetector>( new SVMDetector(params.svmModelFileName, params.hogWinSize, params.SVMThreshold) );
        
    }
    catch(std::exception& err)
    {
        throw std::runtime_error(std::string("ObjDetector :: ") + err.what());
    }
    
    assert( handle->pCascadeDetector && handle->pSVMDetector );     //If no exceptions here, must have valid detectors initialized

    return handle;
}

ObjDetector::ObjDetector() noexcept:
pCascadeDetector(nullptr),
pSVMDetector(nullptr)
{}

ObjDetector::~ObjDetector() = default;

/*!
* Use 2-Stages object detector on the input frame.
* @param[in] frame
* \return a vector of DetectionInfo containing information about the detections 
*/
std::vector<ObjDetector::DetectionInfo> ObjDetector::detect(const cv::Mat& frame)
{
    assert( pCascadeDetector && pSVMDetector );

    //Run cascade detector
    rois_ = pCascadeDetector->detect(frame);
    //groupRectangles(rois_, 1);
    
    //Verify rois_ using HoG-SVM.
    return pSVMDetector->detect(frame, rois_);
}

