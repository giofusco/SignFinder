cmake_minimum_required(VERSION 2.8) 
PROJECT (SignFinder)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

#Find OpenCV
find_package(OpenCV REQUIRED )
if (OpenCV_FOUND)
  message(STATUS "OpenCV Found")
  message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")
  message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

#Find LibSVM
find_package(LIBSVM REQUIRED)
if (LIBSVM_FOUND)
  message(STATUS "LibSVM Found")
  message(STATUS "LibSVM_LIBRARIES = ${LIBSVM_LIBRARIES}")
  message(STATUS "LibSVM_INCLUDE_DIRS = ${LIBSVM_INCLUDE_DIRS}")
  include_directories(${LIBSVM_INCLUDE_DIRS})
endif()

set( NAME_SRC
    src/main.cpp   
    src/DetectionParams.cpp
    src/ObjDetector.cpp
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( ${PROJECT_NAME} ${NAME_SRC} ${NAME_HEADERS} )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)


TARGET_LINK_LIBRARIES(${PROJECT_NAME} opencv_core opencv_imgproc opencv_video opencv_objdetect opencv_highgui opencv_gpu opencv_ml ${LIBSVM_LIBRARIES} ${EXTERNAL_LIBS})