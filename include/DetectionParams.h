#ifndef DETECTION_PARAMS_H
#define DETECTION_PARAMS_H
#include <exception>
#include <iostream>
#include <opencv2/opencv.hpp>


/** \class DetectionParams
*  \brief container of the parameters used by ObjDetector.
*  \details This class parses the YAML configuration file and 
*	initializes the parameters  used by the object detector.
*/

class DetectionParams
{
public:
	DetectionParams();		
	DetectionParams(std::string yamlConfigFile) throw (std::runtime_error); 
	~DetectionParams();

	void loadFromFile(std::string yamlConfigFile) throw(std::runtime_error);
	inline bool isInit() { return init_; }

	std::string configFileName;	///< full path to the configuration file
	std::string cascadeFile;	///< full path to the cascade classifier
	std::string svmModelFile;	///< full path to the SVM model

	cv::Size hogWinSize;		///< windows size for HOG descriptor
	cv::Size cascadeMinWin;		///< min window size for multi-scale detection
	cv::Size cascadeMaxWin;		///< max window size for multi-scale detection 
	
	float cascadeMaxWinFactor;	
	float croppingFactors[2];	///< (0: width, 1: height)
	float scalingFactor;		///< image rescaling factor (0., +inf)
	float cascadeScaleFactor;	///< multiscale detection scaling factor
	float SVMThreshold;			///< theshold for rejection of candidate ROIs
	
	bool flip;					///< flip input image (used for landscape videos)
	bool transpose;				///< transpose input image (used for landscape videos)
	bool showIntermediate;		///< show debugging info
	
private:
	bool init_;					///< specifies if the parameters have been initialized or not
};



#endif