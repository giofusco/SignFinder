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



std::map<std::string, std::string> parseOptions(int argc, char* argv[]){
	std::map<std::string, std::string> options;
	int o = 1;
	std::string opt;
	while (o < argc){
		opt = argv[o];

		if (opt == "-res"){
			options.insert(std::make_pair("res", argv[o + 1]));
		}
		else if (opt == "-video"){
			options.insert(std::make_pair("video", argv[o + 1]));
		}
		else if (opt == "-camid"){
			options.insert(std::make_pair("camid", argv[o + 1]));
		}
		else if (opt == "-dump"){
			options.insert(std::make_pair("dump", argv[o + 1]));
		}
		else if (opt == "-save")
			options.insert(std::make_pair("save", "1"));
		o++;

	}
	return options;
}





int main(int argc, char* argv[]){

	if (argc < 3)
    {
		std::cout << "Not enough input parameters. USAGE: SignFinder -res resourceLocation  ( -video videoFile | -camid webcamID ) [-dump prefix_dumped_patches] [-save] \n";
        return EXIT_FAILURE;
    }

    std::map<std::string, std::string> options = parseOptions(argc, argv);

    try{

        //**** example of usage with hardcoded path
        //std::string basedir = "C:\\dev\\workspace\\WICAB_SingFinding\\Deliverable\\SignFinder\\build\\bin\\res";
        // ObjDetector detector(argv[1], basedir);
        //****

        std::cerr << options["res"] << "\n";
        ObjDetector detector(options["res"]);

        std::string videoname;
        cv::VideoCapture vc;
        if (options.count("camid") > 0){
            vc = cv::VideoCapture(std::stoi(options["camid"]));
            vc.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
            vc.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
        }

        else if (options.count("video") > 0)
            vc = cv::VideoCapture(options["video"]);

        else{
            throw(std::runtime_error("SIGNFINDER ERROR :: No video source has been specified.\n"));
        }

        bool dumpPatches = false;
        bool saveFrames = false;
        std::string patchPrefix;
        if (options.count("dump") > 0){
            dumpPatches = true;
            patchPrefix = options["dump"];
        }

        if (options.count("save") > 0)
            saveFrames = true;

        if (vc.isOpened()){
            cv::Mat frame;
            int keypress;
            int frameno = 0;
            time_t start, end;
            while (vc.read(frame)){
                //measure delta_T
                time(&start);
                auto result = detector.detect(frame);
                time(&end);
                double fps = 1.0 / difftime(end, start);
                ++frameno;

                if (dumpPatches)
                    detector.dumpStage1(patchPrefix);
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
                if (saveFrames)
                    cv::imwrite(std::string("frame_" + std::to_string(frameno) + ".png"), detector.currFrame);
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
#ifndef NDEBUG_
        throw;  //for memory dump in debug mode.
#endif
    }
    return EXIT_SUCCESS;
    
}

