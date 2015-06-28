cmake_minimum_required(VERSION 2.8) 
PROJECT (SignFinder)

find_package(OpenCV REQUIRED )

set(LIBSVM_INCLUDE_DIR "" CACHE FILEPATH "Path to libsvm includes")

set( NAME_SRC
    src/main.cpp   
    src/DetectionParams.cpp
    src/ObjDetector.cpp
    ${LIBSVM_INCLUDE_DIR}/svm.cpp
)


INCLUDE_DIRECTORIES(${LIBSVM_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( SignFinder ${NAME_SRC} ${NAME_HEADERS} )

TARGET_LINK_LIBRARIES(SignFinder opencv_core opencv_imgproc opencv_video opencv_objdetect opencv_highgui opencv_gpu opencv_ml ${EXTERNAL_LIBS})