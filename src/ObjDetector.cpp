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


ObjDetector::ObjDetector(std::string resourceLocation) throw(std::runtime_error):
params_(resourceLocation),
cascade_(params_.cascadeFileName),
hog_(params_.hogWinSize,    //winSize
     cv::Size(16,16),       //blockSize
     cv::Size(4,4),         //blockSize
     cv::Size(8,8),         //cellSize
     9,                     //nbins
     1,                     //derivAperture
     -1,                    //winSigma,
     cv::HOGDescriptor::L2Hys,  //hostogramNormType,
     .2,                    //L2HysThreshold,
     true,                  //gammaCorrection
     1                      //nLevels
     ),
model_(svm_load_model(params_.svmModelFileName.c_str())),
init_(false),
counter_(0)
{
    //check for problems
    
    if (!params_.isInit())
        throw std::runtime_error("OBJDETECTOR ERROR :: Could not initialize parameters.");
    if (cascade_.empty())
        throw(std::runtime_error("OBJDETECTOR ERROR :: Cannot load cascade classifier." + params_.cascadeFileName));
    
    if (model_ == nullptr)
    {
        throw(std::runtime_error("OBJDETECTOR ERROR :: Cannot load SVM classifier.\n"));
    }
    init_ = true;
}

//void setClassifiersFolder(std::string folder);

ObjDetector::~ObjDetector()
{
    if (model_)
        delete model_;
}

/*!
* Use 2-Stages object detector on the input frame.
* @param[in] frame
* \return a vector of DetectionInfo containing information about the detections 
*/
std::vector<ObjDetector::DetectionInfo> ObjDetector::detect(cv::Mat& frame){

    if (!isInit())
    {
        throw std::runtime_error("OBJDETECTOR ERROR :: Could not initialize object detector");
    }
    
    std::vector<DetectionInfo> result;
    
    if (params_.scalingFactor != 1 && params_.scalingFactor > 0)
        resize(frame, frame, cv::Size(frame.size().width * params_.scalingFactor, frame.size().height* params_.scalingFactor));

    if (params_.flip)
        flip(frame, frame, 0);

    if (params_.transpose)      
        frame = frame.t();
    
    //frame.copyTo(currFrame);
    //cropping
    cv::Mat cropped;
    frame(cv::Rect(0, 0, frame.size().width * params_.croppingFactors[0], frame.size().height*params_.croppingFactors[1])).copyTo(cropped);

    rois_.clear();
    cascade_.detectMultiScale(cropped, rois_, params_.cascadeScaleFactor, 0, 0, params_.cascadeMinWin, params_.cascadeMaxWin);
    groupRectangles(rois_, 1);

    if (params_.showIntermediate){
        cv::imshow("Cropped Input", cropped);
        cv::Mat tmp;
        cropped.copyTo(tmp); // temporary copy to avoid changing pixels in the original image
        for (const auto& r : rois_){
            cv::rectangle(tmp, r, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("Stage 1", tmp);
    }

    ++counter_;
    return verifyROIs(cropped, rois_);
}

/*!
* Verifies the ROIs detected in the first stage using SVM + HOG
* @param[in] frame frame to process
* @param[in] rois vector containing the candidate ROIs
* @return a vector of detections that passed the verification
* @throw runtime error if unable to allocate memory for this stage
*/
std::vector<ObjDetector::DetectionInfo> ObjDetector::verifyROIs(cv::Mat& frame, std::vector<cv::Rect>& rois) const throw (std::runtime_error) {

	std::vector<ObjDetector::DetectionInfo> result;
	double prob_est[2];
	std::vector<float> desc;

    std::vector<svm_node> x;

	for (int r = 0; r < rois.size(); r++){
		//use svm to classify the rois
		cv::Mat patch(frame, rois[r]);

		cv::Mat res(1, 1, CV_32FC1);
		resize(patch, patch, params_.hogWinSize);
		hog_.compute(patch, desc);
        x.resize(desc.size() + 1);

        //TODO: This is done for each patch. If descriptor sizes are the same, this can be optimized by moving out of the loop.
		for (int d = 0; d < desc.size(); d++){
			x[d].index = d + 1;  // Index starts from 1; Pre-computed kernel starts from 0
			x[d].value = desc[d];
		}

		x[desc.size()].index = -1;
		patch.release();
		res.release();
		desc.clear();

		svm_predict_probability(model_, x.data(), prob_est);

		if ((prob_est[0] > params_.SVMThreshold))
			result.push_back({ rois[r], prob_est[0] });
	}

	return result;
}
