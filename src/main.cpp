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


#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ObjDetector.h"
#include "version.h"

namespace
{
    /// Parameters and command line arguments
    struct Options
    {
        std::string configFile;         //< The configuration file in YAML format
        std::string input;              // input file stream to process
        std::string output;             // name of output file if one is given
        std::string patchPrefix;        // if non-empty, dump patches to disk with this prefix
		std::string roisFile;			// if non-empty, saves detection ROIs to the specified file
		std::string label;				// label for the ROIs
        int maxDim;                     // maximum dimension of the image in pixels
        bool isFlipped;					//< flip input image if true (used for landscape videos)
        bool isTransposed;				//< transpose input image if true (used for landscape videos)
        bool doShowIntermediate;        //< show debugging info
        bool doSaveFrames;              //< if true, save frames to disk
        bool doTrack;                   //< whether the detector should us tracking.
    };
    
    /// Prints basic usage to terminal
    inline void printUsage()
    {
        std::cerr << "USAGE: SignFinder -c configfile [-p prefix] [-m maxdim] [-r roisFilename] [-s] [-d] [-f] [-t] [-n] [-o output] -i input" << std::endl;
    }
    
    /// Parses command line options
    /// @param[in] argc number of command line arguments (including program name)
    /// @param[in] argv list of arguments
    /// @return an options structure populated by parsing the command line
    /// @throw runtime_error if there was a problem parsing the command line arguments,=
    Options parseOptions(int argc, char* argv[]) throw(std::runtime_error)
    {
        const char* keys =
        {
            "{ h | help            | false       | print this message                                            }"
            "{ v | version         | false       | version info                                                  }"
            "{ i | input           |             | input. Either a file name, or a digit indicating webcam id    }"
            "{ c | configFile      |             | location of config file                                       }"
            "{ p | patchPrefix     |             | prefix for dumping detected patches to disk. If none, nothign is dumped}"
            "{ s | saveFrames      | false       | whether to save frames                                        }"
            "{ d | debug           | false       | whether to show intermediate detection stage results          }"
            "{ f | flip            | false       | whether to flip the input image                               }"
            "{ t | transpose       | false       | whether to transpose the input image                          }"
            "{ n | notrack         | false       | whether to turn off tracking                                  }"
            "{ m | maxdim          | 640         | maximum dimension of the image to use while processing.       }"
            "{ o | output          |             | if a name is specified, the detection results are saved to a video file given here.}"
			"{ r | roisFile        |             | saves detected rois to a text file given here.                }"
			"{ l | label           |             | specify label for the ROIs.                                   }"		
        };
        cv::CommandLineParser parser(argc, argv, keys);
        if ( (1 == argc) || (parser.get<bool>("h")) )
        {
            printUsage();
            parser.printParams();
            std::cout << "SignFinder v" << SIGNFINDER_VERSION << std::endl;
            exit(EXIT_SUCCESS);
        }
        if (parser.get<bool>("v"))
        {
            std::cout << "SignFinder v" << SIGNFINDER_VERSION << std::endl;
            exit(EXIT_SUCCESS);
        }
        
        Options opts;
        
        opts.input = parser.get<std::string>("i");
            std::cerr<<opts.input<< std::endl;
        if (opts.input.empty())
        {
            printUsage();
            throw std::runtime_error("Parser Error :: No input source specified");
        }
        opts.configFile = parser.get<std::string>("c");
        if (opts.configFile.empty())
        {
            printUsage();
            throw std::runtime_error("Parser Error :: No configuration file specified.");
        }
        
        opts.output = parser.get<std::string>("output");
        opts.patchPrefix = parser.get<std::string>("p");
		opts.roisFile = parser.get<std::string>("r");
		opts.label = parser.get<std::string>("l");
        opts.maxDim = parser.get<int>("m");
        opts.doSaveFrames = parser.get<bool>("s");
        opts.doShowIntermediate = parser.get<bool>("d");
        opts.isTransposed = parser.get<bool>("t");
        opts.isFlipped = parser.get<bool>("f");
        opts.doTrack = !parser.get<bool>("n");
        
#ifndef NDEBUG
        //list arguments and parameters
        std::clog << "Program parameters and arguments from the configuration file:" << std::endl;
        std::clog << "\tInput: " << opts.input << std::endl;
        std::clog << "\tConfig file: " << opts.configFile << std::endl;
        if ( !opts.output.empty() )
        {
            std::clog << "\tOutput: " << opts.output << std::endl;
        }
        
        std::clog << std::boolalpha;
        std::clog << "Input file options:" << std::endl;
        std::clog << "\tisFlipped: " << opts.isFlipped << std::endl;
        std::clog << "\tisTransposed: " << opts.isTransposed << std::endl;
        std::clog << "\tmaxDim: " << opts.maxDim << std::endl;
        
        std::clog << "Debug options: " << std::endl;
        std::clog << "\tpatchPrefix: " << opts.patchPrefix << std::endl;
        std::clog << "\tdoShowIntermediate: " << opts.doShowIntermediate << std::endl;
        std::clog << "\tdoSaveFrames: " << opts.doSaveFrames << std::endl;
        std::clog << "\tnoTrack: " << !opts.doTrack << std::endl;
#endif
        return opts;
    }
    
    static const cv::Scalar COLOR_RED {0, 0, 255};
    static const cv::Scalar COLOR_GREEN {0, 255, 0};
    static const cv::Scalar COLOR_BLUE {255, 0, 0};
    static const cv::Scalar COLOR_YELLOW {0, 255, 255};
    
    static const cv::Scalar COLOR_CASCADE_DETECTION = COLOR_RED;
    static const cv::Scalar COLOR_CANDIDATE = COLOR_YELLOW;
    static const cv::Scalar COLOR_VERIFIED_SIGN = COLOR_GREEN;
    
    /// Converts a size structure to a string
    /// @param[in] sz size
    /// @return stringified representation of sz
    template<typename T>
    inline std::string to_string(const cv::Size_<T>& sz)
    {
        return std::to_string(sz.width) + "x" + std::to_string(sz.height);
    }
}   //::<anon>

/// Main entry point
/// @param[in] argc number of command line arguments (including program name)
/// @param[in] argv list of arguments
/// @return EXIT_SUCCESS if the program completes successfully, EXIT_FAILURE if an error occurs.
int main(int argc, char* argv[])
{
    try
    {
        auto options = parseOptions(argc, argv);
        
        ObjDetector detector(options.configFile);
        
        std::string videoname;
        cv::VideoCapture vc;
        if (options.input.size() == 1) //camera
        {
            const int camIndex = (char) options.input.front() - '0';
            std::cerr << camIndex << std::endl;
            if ( (camIndex < 0) || (camIndex > 9) )
            {
                throw std::runtime_error("Parser Error :: Webcam index must be between 0..9");
            }
            vc = cv::VideoCapture(camIndex);
            if (!vc.isOpened())
            {
                throw std::runtime_error(std::string("Unable to open webcam ") + options.input);
            }
            //vc.set(CV_CAP_PROP_FRAME_HEIGHT, options.size.height);
            vc.set(CV_CAP_PROP_FRAME_WIDTH, options.maxDim);
        }
        else
        {
            vc = cv::VideoCapture(options.input);
            if (!vc.isOpened())
            {
                throw std::runtime_error(std::string("Unable to open video file ") + options.input);
            }
        }
    #ifndef NDEBUG
        cv::Size openedStreamSize( vc.get(CV_CAP_PROP_FRAME_WIDTH), vc.get(CV_CAP_PROP_FRAME_HEIGHT) );
        std::clog << "Opened stream size: " << openedStreamSize << std::endl;
    #endif
        
        assert(vc.isOpened());
        cv::Mat frame;
        int keypress;
        int frameno = 0;
        double fps = 0;
        cv::VideoWriter vw;

		std::ofstream roisFile;
		if (!options.roisFile.empty()){
			roisFile.open(options.roisFile);
			if (!roisFile.is_open())
				throw std::runtime_error("Unable to open ROIs file.");
			roisFile << options.input << "\n";
			roisFile << options.label << "\n";
		}
		

        while ( vc.read(frame) )
        {
            ++frameno;
			
            const float scaleFactor = (float) options.maxDim / (float) std::max( frame.cols, frame.rows );
            cv::resize(frame, frame, cv::Size(), scaleFactor, scaleFactor);

			if ((frameno == 1) && roisFile.is_open())
				roisFile << frame.size().height << " " << frame.size().width << "\n";

            if ( !vw.isOpened() && !options.output.empty() )
            {
                std::cerr << "Saving output frames to video: " << options.output << std::endl;
                vw.open(options.output, CV_FOURCC('M', 'P', 'E', 'G'), 30, frame.size());
            }

            if (options.isFlipped)
            {
                cv::flip(frame, frame, 0);
            }
            if (options.isTransposed)
            {
                frame = frame.t();
            }

            // Run detector
            auto result = detector.detect(frame, fps, options.doTrack);
            if ( !options.patchPrefix.empty() )
            {
                detector.dumpStage2(options.patchPrefix);
            }
            putText(detector.currFrame, "FPS: " + std::to_string(fps), cv::Point(100, detector.currFrame.size().height - 100), CV_FONT_HERSHEY_PLAIN, 1.0, COLOR_BLUE);
            if (options.doShowIntermediate)
            {
                //plot amd write confidence and size of stage 1
                auto rois1 = detector.getStage1Rois();
                for (const auto& res : rois1)
                {
                    cv::rectangle(detector.currFrame, res, COLOR_CASCADE_DETECTION, 1);
                    putText(detector.currFrame, to_string(res.size()), res.tl(), CV_FONT_HERSHEY_PLAIN, 1.0, COLOR_CASCADE_DETECTION);
                }

                //plot amd write confidence and size of stage 2
                auto rois2 = detector.getStage2Rois();
                for (const auto& res : rois2)
                {
                    cv::rectangle(detector.currFrame, res.roi, COLOR_CANDIDATE, 1);
                    putText(detector.currFrame, to_string(res.roi.size()), res.roi.tl(), CV_FONT_HERSHEY_PLAIN, 1.0, COLOR_CASCADE_DETECTION);
                }
            }
            //plotting detecions and confidence values
            for (const auto& res : result)
            {
                cv::rectangle(detector.currFrame, res.roi, COLOR_VERIFIED_SIGN, 2);
                putText(detector.currFrame, "p=" + std::to_string(res.confidence), res.roi.br(), CV_FONT_HERSHEY_PLAIN, 1.0, COLOR_VERIFIED_SIGN);
                putText(detector.currFrame, to_string(res.roi.size()), res.roi.tl(), CV_FONT_HERSHEY_PLAIN, 1.0, COLOR_VERIFIED_SIGN);
				if (!options.roisFile.empty()){
					roisFile << frameno << " " << res.roi.tl().x << " " << res.roi.tl().y << " " << res.roi.br().x << " " << res.roi.br().y  
						<< " " << res.confidence << " " << options.label << "\n";
				}
            }
            cv::imshow("Detection", detector.currFrame);
            if (options.doSaveFrames)
            {
                cv::imwrite(std::string("frame_" + std::to_string(frameno) + ".png"), detector.currFrame);
            }
            keypress = cv::waitKey(1);
            
            if (keypress == 27) //exit on escape
                break;
        }
        return EXIT_SUCCESS;
    }
    catch (std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        return EXIT_FAILURE;
    }
}

