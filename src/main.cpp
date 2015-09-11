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

namespace
{
    /// Parameters and command line arguments
    struct Options
    {
        // Detector parameters
        ObjDetector::Parameters detectorParameters;
        // Input parameters
        std::string input;              ///< input file stream to process
        bool isFlipped;					///< flip input image if true (used for landscape videos)
        bool isTransposed;				///< transpose input image if true (used for landscape videos)
        bool doShowIntermediate;        ///< show debugging info
        bool doDumpPatches;             ///< if true, dump patches to disk
        bool doSaveFrames;              ///< if true, save frames to disk

        Options():
        isFlipped(false),
        isTransposed(false),
        doShowIntermediate(false),
        doDumpPatches(false),
        doSaveFrames(false)
        {}
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

//        n = fs["ScaleFactor"];
//        const float scalingFactor = (n.empty() ? 1.f : (float) n[0]);

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

//        n = fs["CroppingFactors"];
//        if (!n.empty()){
//            for (it = n.begin(); it != n.end(); ++it){
//                cv::FileNode tmp = *it;
//                if (tmp.name() == "width")
//                    croppingFactors[0] = (float)tmp[0];
//                else if (tmp.name() == "height")
//                    croppingFactors[1] = (float)tmp[0];
//                else{
//                    std::clog << "Config File :: Unexpected Parameter: '" << tmp.name() << "' Ignoring.\n";
//                }
//            }
//        }
//        else{
//            croppingFactors[0] = 1.;
//            croppingFactors[1] = 1.;
//        }
//        
//        n = fs["ScaleFactor"];
//        if (!n.empty())
//            scalingFactor = (float)n[0];
//        else
//            scalingFactor = 1.;
//        
//        n = fs["Flip"];
//        if (!n.empty())
//            flip = (int)n[0];
//        else
//            flip = false;
//        
//        n = fs["Transpose"];
//        if (!n.empty())
//            transpose = (int)n[0];
//        else
//            transpose = false;
//        
//        n = fs["ShowIntermediate"];
//        if (!n.empty())
//            showIntermediate = (int)n[0];
//        else
//            showIntermediate = false;
//        
        
    }
    
    Options parseOptions(int argc, char* argv[])
    {
//        std::map<std::string, std::string> options;
//        int o = 1;
//        std::string opt;
//        while (o < argc){
//            opt = argv[o];
//            
//            if (opt == "-res"){
//                options.insert(std::make_pair("res", argv[o + 1]));
//            }
//            else if (opt == "-video"){
//                options.insert(std::make_pair("video", argv[o + 1]));
//            }
//            else if (opt == "-camid"){
//                options.insert(std::make_pair("camid", argv[o + 1]));
//            }
//            else if (opt == "-dump"){
//                options.insert(std::make_pair("dump", argv[o + 1]));
//            }
//            else if (opt == "-save")
//                options.insert(std::make_pair("save", "1"));
//            o++;
//            
//        }
        
        const char* keys =
        {
            "{h | help            | false       | print this message                                            }"
            "{v | version         | false       | version info                                                  }"
            "{1 |                 |             | input. Either a file name, or a digit indicating webcam id    }"
            "{r | resources       |             | location of resource files                                    }"
            "{d | dump_patches    | false       | whether to dump detected patches to disk                      }"
            "{s | save_frames     | false       | whether to save frames                                        }"
            "{x | show_intermediate| false      | whether to show intermediate detection stage results          }"
            "{f | flip            | false       | whether to flip the input image                               }"
            "{t | transpose       | false       | whether to transpose the input image                          }"
            "{c | crop            | 1,1         | ratio of the imahe to crop (w,h). 1,1 is the whole image.     }"
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
        
        opts.doDumpPatches = parser.get<bool>("d");
        opts.doSaveFrames = parser.get<bool>("s");
        opts.doShowIntermediate = parser.get<bool>("x");
        opts.isTransposed = parser.get<bool>("t");
        opts.isFlipped = parser.get<bool>("f");
        
        const std::string cropping = parser.get<std::string>("c");
#ifndef NDEBUG_
        //list arguments and parameters
        std::cerr << "Program parameters and arguments from the configuration file:" << std::endl;
        std::cerr << "\tInput: " << opts.input << std::endl;
        std::cerr << "\tConfig file: " << configFileName << std::endl;
        std::cerr << "\tCascade file: " << opts.detectorParameters.cascadeFileName << std::endl;
        std::cerr << "\tSVM model file: " << opts.detectorParameters.svmModelFileName << std::endl;

        std::cerr << "Input file options:" << std::endl;
        std::cerr << "\tisFlipped: " << opts.isFlipped << std::endl;
        std::cerr << "\tisTransposed: " << opts.isTransposed << std::endl;
        std::cerr << "\tcrop" << cropping;
        
        std::cerr << "Debug options: " << std::endl;
        std::cerr << "\tdoShowIntermediate: " << opts.doShowIntermediate << std::endl;
        std::cerr << "\tdoDumpPatches: " << opts.doDumpPatches << std::endl;
        std::cerr << "\tdoSaveFrames: " << opts.doSaveFrames << std::endl;
#endif
        
        loadDetectorParametersFromFile(configFileName, opts.detectorParameters);
        
#ifndef NDEBUG_
        std::cerr << "Cascade detector parameters: " << std::endl;
        std::cerr << "\tminWinSize: " << opts.detectorParameters.cascadeMinWin << std::endl;
        std::cerr << "\tmaxWinSize: " << opts.detectorParameters.cascadeMaxWin << std::endl;
        std::cerr << "\tscaleFactor: " << opts.detectorParameters.cascadeScaleFactor;
        
        std::cerr << "SVM parameters: " << std::endl;
        std::cerr << "\tHoGWinSize: " << opts.detectorParameters.hogWinSize << std::endl;
        std::cerr << "\tsvmThreshold: " << opts.detectorParameters.SVMThreshold << std::endl;
#endif
        return opts;
        
    }

#ifndef NDEBUG_
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
    
}




int main(int argc, char* argv[]){

	if (argc < 3)
    {
		std::cout << "Not enough input parameters. USAGE: SignFinder -res resourceLocation  ( -video videoFile | -camid webcamID ) [-dump prefix_dumped_patches] [-save] \n";
        return EXIT_FAILURE;
    }

    auto options = parseOptions(argc, argv);

    try{

        auto pDetector = ObjDetector::Create(options.detectorParameters);

        std::string videoname;
        cv::VideoCapture vc;
        if (options.input.size() == 1) //camera
        {
            const char camIndex = options.input.front();
        }
        
        
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
            while (vc.read(frame)){
                //measure delta_T
                auto start = std::chrono::system_clock::now();
                auto result = pDetector->detect(frame);
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
                double fps = 1000.0 / duration.count();
                ++frameno;

                if (dumpPatches)
                {
                    dumpStage1(*pDetector, frame, patchPrefix + "_" + std::to_string(frameno));
                }
                putText(frame, "FPS: " + std::to_string(fps), cv::Point(100, frame.size().height - 100), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0));

                //plotting ROIs and confidence values
                for (const auto& res : result)
                {
                    cv::rectangle(frame, res.roi, cv::Scalar(0, 0, 255), 2);
                    //write confidence and size 
                    putText(frame, "p=" + std::to_string(res.confidence), res.roi.br(), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
                    putText(frame, std::to_string(res.roi.width) + "x" + std::to_string(res.roi.height), res.roi.tl(), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
                }
                cv::imshow("Detection", frame);
                if (saveFrames)
                    cv::imwrite(std::string("frame_" + std::to_string(frameno) + ".png"), frame);
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

