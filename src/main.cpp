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
#include <exception>

int main(int argc, char* argv[]){

	if (argc < 3)
		std::cout << "Not enough input parameters. USAGE: signFinder ParametersFile videoFile.\n";
	else{

		try{
			ObjDetector detector(argv[1]);
			std::string videoname(argv[2]);
			std::string outputFileName;
			cv::VideoCapture vc(videoname);

			if (vc.isOpened()){
				cv::Mat frame;
				int keypress;
				int frameno = 0;
				double fps = 0;
				while (vc.read(frame)){

					++frameno;

					auto result = detector.detect(frame, fps);
					putText(detector.currFrame, "FPS: " + std::to_string(fps), cv::Point(100, detector.currFrame.size().height - 100), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));

					//plotting ROIs and confidence values
					for (const auto& res : result)
					{
						cv::rectangle(detector.currFrame, res.roi, cv::Scalar(0, 0, 255), 2);
						//write confidence and size 
						putText(detector.currFrame, "p=" + std::to_string(res.confidence), res.roi.br(), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
						putText(detector.currFrame, std::to_string(res.roi.width) + "x" + std::to_string(res.roi.height), res.roi.tl(), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
					}
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


