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

Author: Giovanni Fusco - giofusco@ski.org & Ender Tekin

*/

#include "DetectionParams.h"

/*!
* Basic constructor. The state will not be valid,
* unless loadFromFile is successively called.
*/
DetectionParams::DetectionParams():
init_(false)
{
}


/*!
* Constructor with initialization
* @param[in] yamlConfigFile the full path to the configuration file to parse
* \exception std::runtime_error error accessing or parsing the configuration file
*/
DetectionParams::DetectionParams(const std::string& yamlConfigFile, const std::string& classifiersFolder) throw(std::runtime_error) :
DetectionParams()
{
	loadFromFile(yamlConfigFile, classifiersFolder);
}


/*!
* Destructor
*/
DetectionParams::~DetectionParams() = default;


/*!
* Initialize the parameters using a configuration file
* @param[in] yamlConfigFile the full path to the configuration file to parse
* @param[in] classifiersFolder the folder containing the classifiers to load
*/
void DetectionParams::loadFromFile(const std::string& yamlConfigFile, const std::string& classFolder) throw(std::runtime_error){

	cv::FileNode n;
	cv::FileNodeIterator it;

	try{
		cv::FileStorage fs(yamlConfigFile, cv::FileStorage::READ);
		configFileName = yamlConfigFile;

		if (fs.isOpened()){

			//if the folder containing the classifiers is not specified read it from the config file
			if (classFolder.empty()){
				this->classifiersFolder = (std::string)fs["ClassifiersFolder"];
				if (classifiersFolder.empty())
					throw (std::runtime_error("CONFIG PARSER ERROR :: Classifiers Folder not specified. \n"));
			}
			else classifiersFolder = classFolder;

			//replace separators with the right ones
			fixPathString(classifiersFolder);
			
			cascadeFile = (std::string)fs["CascadeFile"];
			if (cascadeFile.empty())
				throw (std::runtime_error("CONFIG PARSER ERROR :: Cascade Classifier not specified. \n"));
			cascadeFile = classifiersFolder + cascadeFile; //full filename

			svmModelFile = (std::string)fs["SVMFile"];
			if (svmModelFile.empty())
				throw (std::runtime_error("CONFIG PARSER ERROR :: SVM Classifier not specified. \n"));
			svmModelFile = classifiersFolder + svmModelFile; //full filename

			n = fs["minWinSize"];
			if (!n.empty()){
				for (it = n.begin(); it != n.end(); ++it){
					cv::FileNode tmp = *it;
					if (tmp.name() == "width")
						cascadeMinWin.width = (int)tmp[0];
					else if (tmp.name() == "height")
						cascadeMinWin.height = (int)(int)tmp[0];
					else{
						std::clog << "Config File :: Unexpected Parameter: '" << tmp.name() << "' Ignoring.\n";
					}
				}
			}
			else
				throw(std::runtime_error("CONFIG PARSER ERROR :: Cascade Minimum Window Size not specified.\n"));


			n = fs["maxWinSizeFactor"];
			if (!n.empty())
				cascadeMaxWinFactor = (float)n[0];
			else{
				std::cerr << "Config File :: maxWinSizeFactor not found. Using default value.\n";
				cascadeMaxWinFactor = 8.;
			}
			cascadeMaxWin = cv::Size(cascadeMinWin.width*cascadeMaxWinFactor, cascadeMinWin.height*cascadeMaxWinFactor);

			n = fs["HOG_winSize"];
			if (!n.empty()){
				for (it = n.begin(); it != n.end(); ++it){
					cv::FileNode tmp = *it;
					if (tmp.name() == "width")
						hogWinSize.width = (int)tmp[0];
					else if (tmp.name() == "height")
						hogWinSize.height = (int)(int)tmp[0];
					else{
						std::clog << "Config File :: Unexpected Parameter: '" << tmp.name() << "' Ignoring.\n";
					}
				}
			}
			else
				throw(std::runtime_error("CONFIG PARSER ERROR :: HOG Window Size not specified.\n"));

			n = fs["CroppingFactors"];
			if (!n.empty()){
				for (it = n.begin(); it != n.end(); ++it){
					cv::FileNode tmp = *it;
					if (tmp.name() == "width")
						croppingFactors[0] = (float)tmp[0];
					else if (tmp.name() == "height")
						croppingFactors[1] = (float)tmp[0];
					else{
						std::clog << "Config File :: Unexpected Parameter: '" << tmp.name() << "' Ignoring.\n";
					}
				}
			}
			else{
				croppingFactors[0] = 1.;
				croppingFactors[1] = 1.;
			}

			n = fs["ScaleFactor"];
			if (!n.empty())
				scalingFactor = (float)n[0];
			else
				scalingFactor = 1.;

			n = fs["Flip"];
			if (!n.empty())
				flip = (int)n[0];
			else
				flip = false;

			n = fs["Transpose"];
			if (!n.empty())
				transpose = (int)n[0];
			else
				transpose = false;

			n = fs["ShowIntermediate"];
			if (!n.empty())
				showIntermediate = (int)n[0];
			else
				showIntermediate = false;

			n = fs["CascadeScaleFactor"];
			if (!n.empty())
				cascadeScaleFactor = (float)n[0];
			else
				cascadeScaleFactor = 1.1;

			n = fs["SVMThreshold"];
			if (!n.empty())
				SVMThreshold = (float)n[0];
			else
				SVMThreshold = .5;

            n = fs["maxAgePreConfirmation"];
            maxAgePreConfirmation = (n.empty() ? 5 : (int) n);
            
            n = fs["maxAgePostConfirmation"];
            maxAgePostConfirmation = (n.empty() ? 15 : (int) n);
            
            n = fs["nHangOverFrames"];
            nHangOverFrames = (n.empty() ? 3 : (int) n);


			init_ = true;
		}
		else
			throw(std::runtime_error("CONFIG PARSER ERROR :: Couldn't load configuration file: " + yamlConfigFile + "\n"));
	}

	catch (std::exception& e)
	{
		throw(e);
	}

}

void DetectionParams::fixPathString(std::string& instring){

	char  sep = separator();

#ifdef _WIN32
	std::replace(instring.begin(), instring.end(), '/', '\\'); // replace all '/' to '\'
#else
	std::replace(instring.begin(), instring.end(), '\\', '/'); // replace all '\' to '/'
#endif
	if (instring.back() != sep)
		instring.push_back(sep);

}

void DetectionParams::setClassifiersFolder(std::string folder){ ///< sets the folder containing the classifiers (useful to change path at runtime)

}