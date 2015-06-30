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


/*! \mainpage WICAB - SignFinder
*/


#include "ObjDetector.h"
#include <exception>

int main(int argc, char* argv[]){

	if (argc < 3)
		std::cout << "Not enough input parameters. USAGE: signFinder ParametersFile videoFile.\n";
	else{

		try{
			ObjDetector detector = ObjDetector(argv[1]);
			std::string videoname = std::string(argv[2]);
			cv::VideoCapture vc;
			if (vc.open(videoname)){
				cv::Mat frame;
				std::vector<cv::Rect> rois;
				int keypress;
				std::vector<cv::Rect>::iterator it;
				while (vc.read(frame)){
					rois = detector.detect(frame);
					
					//plotting ROIs
					for (it = rois.begin(); it != rois.end(); ++it)
						rectangle(detector.currFrame, *it, cv::Scalar(0, 0, 255), 2);
					cv::imshow("Detection", detector.currFrame);

					keypress = cv::waitKey(1);

					if (keypress == 27) //exit on escape
						break;
				}
			}
			else
				throw(std::runtime_error("SIGNFINDER ERROR :: Unable to load video file.\n"));
		}

		catch (std::exception& e){
			std::cout << e.what() << '\n';
		}
	}
}


