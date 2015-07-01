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


Author:    Giovanni Fusco

*/

#include "ObjDetector.h"


ObjDetector::ObjDetector()
{
	params_ = DetectionParams();
}


ObjDetector::ObjDetector(std::string yamlConfigFile) throw(std::runtime_error) : ObjDetector(){
	params_.loadFromFile(yamlConfigFile);
	init();
}


ObjDetector::~ObjDetector()
{

}

/*!
* initializes the classifiers and the HOG feature extractor.
* \exception std::runtime_error error loading one of the classfiers
*/
void ObjDetector::init()throw(std::runtime_error){
	if (!cascade_.load(params_.cascadeFile))
		throw(std::runtime_error("OBJDETECTOR ERROR :: Cannot load cascade classifier.\n"));

	//setting up HOG descriptor
	hog_.winSize = params_.hogWinSize;
	//default OpenCV HOG params 
	hog_.blockStride = cv::Size(4, 4);
	hog_.cellSize = cv::Size(8, 8);
	hog_.nlevels = 1;

	//setting up SVM
	model_ = svm_load_model(params_.svmModelFile.c_str());
	if (model_ == NULL)
		throw(std::runtime_error("OBJDETECTOR ERROR :: Cannot load SVM classifier.\n"));
}

/*!
* Use 2-Stages object detector on the input frame.
* @param[in] frame
* \return a vector of detection ROIs 
*/
std::vector<cv::Rect> ObjDetector::detect(cv::Mat& frame){

	if (params_.isInit()){
		if (params_.scalingFactor != 1 && params_.scalingFactor > 0)
			resize(frame, frame, cv::Size(frame.size().width * params_.scalingFactor, frame.size().height* params_.scalingFactor));

		if (params_.flip)
			flip(frame, frame, 0);

		if (params_.transpose){
			frame = frame.t();
			frame.copyTo(currFrame);
			//cropping
			cv::Mat cropped;
			frame(cv::Rect(0, 0, frame.size().width * params_.croppingFactors[0], frame.size().width*params_.croppingFactors[1])).copyTo(cropped);

			if (params_.showIntermediate)
				cv::imshow("Cropped Input", cropped);

			std::vector<cv::Rect> rois, filteredRois;
			cascade_.detectMultiScale(cropped, rois, params_.cascadeScaleFactor, 0, 0, params_.cascadeMinWin, params_.cascadeMaxWin);
			groupRectangles(rois, 1);

			filteredRois = verifyROIs(cropped, rois);
			return filteredRois;
		}
	}
}

/*!
* Verifies the ROIs detected in the first stage using SVM + HOG
* @param[in] frame frame to process
* @param[in] rois vector containing the candidate ROIs
* \return a vector of ROIs that passed the verification
*/
std::vector<cv::Rect> ObjDetector::verifyROIs(cv::Mat& frame, std::vector<cv::Rect>& rois){

	std::vector<cv::Rect> outRois;
	double prob_est[2];
	std::vector<float> desc;
	for (int r = 0; r < rois.size(); r++){
		//use svm to classify the rois
		cv::Mat patch(frame, rois[r]);

		cv::Mat res(1, 1, CV_32FC1);
		resize(patch, patch, params_.hogWinSize);
		hog_.compute(patch, desc);
		svm_node *x;
		x = (struct svm_node *)malloc((desc.size() + 1)*sizeof(struct svm_node));

		for (int d = 0; d<desc.size(); d++){
			x[d].index = d + 1;  // Index starts from 1; Pre-computed kernel starts from 0
			x[d].value = desc[d];
		}

		x[desc.size()].index = -1;
		patch.release();
		res.release();
		desc.clear();

		int predict_label = svm_predict_probability(model_, x, prob_est);
		if (prob_est[0] > params_.SVMThreshold)
			outRois.push_back(rois[r]);
		delete(x);
	}

	return outRois;
}