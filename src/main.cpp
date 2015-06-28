//// main.cpp : Defines the entry point for the console application.


/*! \mainpage WICAB - SignFinder
*
* \section intro_sec Introduction
*
* This is the introduction.
*
* \section install_sec Installation
*
* \subsection step1 Step 1: OpenCV
* Install and compile [OpenCV 2.4.11 (download here)]( http://opencv.org/downloads.html). <br>
* [Installation instruction here] (http://docs.opencv.org/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html). <br>
* After the installation, make sure that the directory containing the binary libraries is included in the system path. <br>
*
** Warning: This software has been tested using [OpenCV 2.4.11]( http://opencv.org/downloads.html). Using other versions may result in incompatibilities or it may affect performance.

* etc...
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


