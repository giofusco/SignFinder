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
    ObjDetector();	///< basic constructor. The parameters are not initialized.
	ObjDetector(std::string yamlConfigFile) throw (std::runtime_error);	///< constructor. The parameters are inizialized using the file passed in input.
	~ObjDetector();
	std::vector<cv::Rect> detect(cv::Mat& frame);	///< performs object detection on the frame in input. 
	inline void init(std::string yamlConfigFile) 
		throw (std::runtime_error) /// initializes the parameters using the file in input. 
		{ params_.loadFromFile(yamlConfigFile); }; 
	
	cv::Mat currFrame; ///< last processed frame

private:
	void init() throw (std::runtime_error);	///< initializes the classifiers
	std::vector<cv::Rect> verifyROIs(cv::Mat& frame, std::vector<cv::Rect>& rois); ///< Filters candidate ROIs using SVM

	DetectionParams params_;			///< parameters of the detector
	cv::CascadeClassifier cascade_;		///< cascade classifier
	cv::HOGDescriptor hog_;				///< hog feature extractor
	struct svm_model* model_;			///< svm classifier
};

#endif