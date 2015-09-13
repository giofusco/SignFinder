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
#include <chrono>
#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
    /// Parameters and command line arguments
    struct Options
    {
        // Detector parameters
        ObjDetector::Parameters detectorParameters;
        // Input parameters
        std::string input;              ///< input file stream to process
        std::string patchPrefix;        ///< if non-empty, dump patches to disk with this prefix
        cv::Size size;                  ///< image size to use while processing. If aspect ratio differs, then the image is first cropped so the aspect ratio is not changed.
        bool isFlipped;					///< flip input image if true (used for landscape videos)
        bool isTransposed;				///< transpose input image if true (used for landscape videos)
        bool doShowIntermediate;        ///< show debugging info
        bool doSaveFrames;              ///< if true, save frames to disk
    };
    
    void loadDetectorParametersFromFile(const std::string& configFile, ObjDetector::Parameters& params)
    {
        if (configFile.empty())
        {
            throw std::runtime_error("Parser Error :: No config file specified.");
        }
        
        cv::FileStorage fs(configFile, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            throw std::runtime_error("Parser Error :: Couldn't load configuration file: " + configFile + "\n") ;

        }
        
        cv::FileNode n, n2;
        
        //replace separators with the right ones
        //fixPathString(classifiersFolder);
        
        // Required parameters
        n = fs["minWinSize"];
        if ( n.empty() )
        {
            throw std::runtime_error("Parser Error :: Cascade Minimum Window Size not specified.\n");
        }
        n2 = n["width"];
        if ( n2.empty() )
        {
            throw std::runtime_error("Parser Error :: Cascade Minimum Window Size width not specified.\n");
        }
        params.cascadeMinWin.width = (int) n2;
        n2 = n["height"];
        if ( n2.empty() )
        {
            throw std::runtime_error("Parser Error :: Cascade Minimum Window Size height not specified.\n");
        }
        params.cascadeMinWin.height = (int) n2;
        
        n = fs["HOG_winSize"];
        if ( n.empty() )
        {
            throw(std::runtime_error("Parser Error :: HOG Window Size not specified.\n"));
        }
        n2 = n["width"];
        if ( n2.empty() )
        {
            throw std::runtime_error("Parser Error :: HOG Window Size width not specified.\n");
        }
        params.hogWinSize.width = (int) n2;
        n2 = n["height"];
        if ( n2.empty() )
        {
            throw std::runtime_error("Parser Error :: HOG Window Size height not specified.\n");
        }
        params.hogWinSize.height = (int) n2;

        // Optional parameters
        n = fs["maxWinSizeFactor"];
        const float cascadeMaxWinFactor = (n.empty() ? 8.f : (float) n[0]);
        params.cascadeMaxWin.width = params.cascadeMinWin.width * cascadeMaxWinFactor;
        params.cascadeMaxWin.height = params.cascadeMinWin.height * cascadeMaxWinFactor;

        n = fs["CascadeScaleFactor"];
        if ( !n.empty() )
        {
            params.cascadeScaleFactor = (float) n;
        }
        
        n = fs["SVMThreshold"];
        if ( !n.empty() )
        {
            params.SVMThreshold = (float) n;
        }
    }
    
    Options parseOptions(int argc, char* argv[])
    {
        const char* keys =
        {
            "{h | help            | false       | print this message                                            }"
            "{v | version         | false       | version info                                                  }"
            "{1 |                 |             | input. Either a file name, or a digit indicating webcam id    }"
            "{r | resources       |             | location of resource files                                    }"
            "{p | patchePrefix    |             | prefix for dumping detected patches to disk. If none, nothign is dumped}"
            "{s | save_frames     | false       | whether to save frames                                        }"
            "{d | debug           | false       | whether to show intermediate detection stage results          }"
            "{f | flip            | false       | whether to flip the input image                               }"
            "{t | transpose       | false       | whether to transpose the input image                          }"
            "{x | size            | 640x480     | size of the image to use while processing.                    }"
        };
        cv::CommandLineParser parser(argc, argv, keys);
        if (parser.get<bool>("h"))
        {
            parser.printParams();
            std::cout << "SignFinder v0.1" << std::endl;
            exit(EXIT_SUCCESS);
        }
        if (parser.get<bool>("v"))
        {
            std::cout << "SignFinder v0.1" << std::endl;
            exit(EXIT_SUCCESS);
        }
        
        Options opts;
        
        //TODO: Check and sanitize command line arguments
        opts.input = parser.get<std::string>("1");
        if (opts.input.empty())
        {
            throw std::runtime_error("Parser Error :: No input source specified");
        }
        const std::string resourceLocation = parser.get<std::string>("r");
        if (resourceLocation.empty())
        {
            throw std::runtime_error("Parser Error :: No resource location specified");
        }
        // Get appropriate file names
        const std::string configFileName = resourceLocation + "/" + "config.yaml";
        opts.detectorParameters.cascadeFileName = resourceLocation + "/" + "cascade.xml";
        opts.detectorParameters.svmModelFileName = resourceLocation + "/" + "model.svm";
        
        opts.patchPrefix = parser.get<std::string>("p");
        opts.doSaveFrames = parser.get<bool>("s");
        opts.doShowIntermediate = parser.get<bool>("d");
        opts.isTransposed = parser.get<bool>("t");
        opts.isFlipped = parser.get<bool>("f");
        
        const std::string szString = parser.get<std::string>("x");
        int n = szString.find('x');
        if (n == std::string::npos)
        {
            throw std::runtime_error("Parser Error :: Unable tp parse size");
        }
        opts.size.width = std::stoi(szString.substr(0, n));
        opts.size.height = std::stoi(szString.substr(n+1));
        
#ifndef NDEBUG
        //list arguments and parameters
        std::cerr << "Program parameters and arguments from the configuration file:" << std::endl;
        std::cerr << "\tInput: " << opts.input << std::endl;
        std::cerr << "\tConfig file: " << configFileName << std::endl;
        std::cerr << "\tCascade file: " << opts.detectorParameters.cascadeFileName << std::endl;
        std::cerr << "\tSVM model file: " << opts.detectorParameters.svmModelFileName << std::endl;

        std::cerr << std::boolalpha;
        std::cerr << "Input file options:" << std::endl;
        std::cerr << "\tisFlipped: " << opts.isFlipped << std::endl;
        std::cerr << "\tisTransposed: " << opts.isTransposed << std::endl;
        std::cerr << "\tsize: " << opts.size << std::endl;
        
        std::cerr << "Debug options: " << std::endl;
        std::cerr << "\tpatchPrefix: " << opts.patchPrefix << std::endl;
        std::cerr << "\tdoShowIntermediate: " << opts.doShowIntermediate << std::endl;
        std::cerr << "\tdoSaveFrames: " << opts.doSaveFrames << std::endl;
#endif
        
        loadDetectorParametersFromFile(configFileName, opts.detectorParameters);
        
#ifndef NDEBUG
        std::cerr << "Cascade detector parameters: " << std::endl;
        std::cerr << "\tminWinSize: " << opts.detectorParameters.cascadeMinWin << std::endl;
        std::cerr << "\tmaxWinSize: " << opts.detectorParameters.cascadeMaxWin << std::endl;
        std::cerr << "\tscaleFactor: " << opts.detectorParameters.cascadeScaleFactor << std::endl;
        
        std::cerr << "SVM parameters: " << std::endl;
        std::cerr << "\tHoGWinSize: " << opts.detectorParameters.hogWinSize << std::endl;
        std::cerr << "\tsvmThreshold: " << opts.detectorParameters.SVMThreshold << std::endl;
#endif
        return opts;
        
    }

#ifndef NDEBUG
    void dumpStage1(const ObjDetector& detector, const cv::Mat& frame, const std::string& prefix) {
        int cnt = 0;
        for (const auto& r : detector.getFirstStageResults()){
            cnt++;
            cv::Mat p = frame(r);
            std::string fname = prefix + "_" + std::to_string(cnt) + ".png";
            cv::imwrite(fname, p);
        }
    }
#endif
    
    inline std::string to_string(const cv::Size& sz)
    {
        return std::to_string(sz.width) + "x" + std::to_string(sz.height);
    }

}

int main(int argc, char* argv[])
{
    auto options = parseOptions(argc, argv);

    auto pDetector = ObjDetector::Create(options.detectorParameters);

    std::string videoname;
    cv::VideoCapture vc;
    if (options.input.size() == 1) //camera
    {
        const int camIndex = options.input.front() - '0';
        if ( (camIndex < 0) || (camIndex > 9) )
        {
            throw std::runtime_error("Parser Error :: Webcam index must be between 0..9");
        }
        vc = cv::VideoCapture(camIndex);
        if (!vc.isOpened())
        {
            throw std::runtime_error(std::string("Unable to open webcam ") + options.input);
        }
        vc.set(CV_CAP_PROP_FRAME_HEIGHT, options.size.height);
        vc.set(CV_CAP_PROP_FRAME_WIDTH, options.size.width);
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
    std::cerr << "Opened stream size: " << openedStreamSize << std::endl;
#endif

    assert(vc.isOpened());

    cv::Mat frame;
    int keypress;
    int frameno = 0;
    while ( vc.read(frame) )
    {
        //preprocess frame (resize + crop)
        if (options.isFlipped)
            cv::flip(frame, frame, 0);
        if (options.isTransposed)
            frame = frame.t();
        
        const float scaleFactor = std::min( (float) options.size.width / (float) frame.cols, (float) options.size.height / (float) frame.rows );
        cv::resize(frame, frame, cv::Size(), scaleFactor, scaleFactor);
        //cv::resize(frame, frame, options.size);

        //measure delta_T
        auto start = std::chrono::system_clock::now();
        auto result = pDetector->detect(frame);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);    //duration in milliseconds
        double fps = 1000.0 / duration.count(); // (1000 ms/s) / (duration ms / frame) = frame/s
        std::cerr << "Frame " << frameno << ": " << "Stage 1 det = " << pDetector->getFirstStageResults().size() << ", Stage 2 det = " << result.size() << std::endl;
        if ( !result.empty() )
        {
            std::cout << '\a';
        }
        ++frameno;

        if (options.doShowIntermediate)
        {
            for (const auto& r : pDetector->getFirstStageResults())
            {
                cv::rectangle(frame, r, cv::Scalar(255, 0, 0), 2);
            }
        }
        if (!options.patchPrefix.empty())
        {
            dumpStage1(*pDetector, frame, options.patchPrefix + "_" + std::to_string(frameno));
        }
        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(100, frame.size().height - 100), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));

        //plotting ROIs and confidence values
        for (const auto& res : result)
        {
            cv::rectangle(frame, res.roi, cv::Scalar(0, 0, 255), 2);
            //write confidence and size 
            cv::putText(frame, "p=" + std::to_string(res.confidence), res.roi.br(), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
            cv::putText(frame, to_string(res.roi.size()), res.roi.tl(), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
        }

        cv::imshow("Detections", frame);

        if (options.doSaveFrames)
            cv::imwrite(std::string("frame_" + std::to_string(frameno) + ".png"), frame);
        keypress = cv::waitKey(1);

        if (keypress == 27) //exit on escape
            break;
    }
    return EXIT_SUCCESS;
    
}

