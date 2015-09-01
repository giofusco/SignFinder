cmake_minimum_required(VERSION 2.8) 
PROJECT (SignFinder)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "compile flags" FORCE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

set( NAME_SRC
    src/main.cpp   
    src/DetectionParams.cpp
    src/ObjDetector.cpp
)


#Find OpenCV
find_package(OpenCV REQUIRED )
if (OpenCV_FOUND)
  message(STATUS "OpenCV Found")
  message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")
  message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


#Find LibSVM
set(LIBSVM_LIBRARY "")
find_package(LibSVM)
if (LIBSVM_FOUND)
  message(STATUS "LibSVM Found")
  message(STATUS "LibSVM_LIBRARIES = ${LIBSVM_LIBRARIES}")
  message(STATUS "LibSVM_INCLUDE_DIRS = ${LIBSVM_INCLUDE_DIRS}")
  include_directories(${LIBSVM_INCLUDE_DIRS})
else()
  set(LIBSVM_DIR "" CACHE FILEPATH "Path to libsvm includes")
  include_directories(${LIBSVM_DIR})
  list(APPEND NAME_SRC ${LIBSVM_DIR}/svm.cpp)
endif()

#Find LibSVM
#set(LIBSVM_DIR "" CACHE FILEPATH "Path to libsvm includes")

#Copy resources
set(BIN_NAME ${PROJECT_NAME})
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message(STATUS "Debug mode")
    set(TARGET_NAME ${TARGET_NAME}d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message(STATUS "Release mode")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB NAME_HEADERS "include/*.h" "include/*.hpp")

set(OUTPUT_FOLDER ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_FOLDER})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_FOLDER})
add_executable( ${BIN_NAME} ${NAME_SRC} ${NAME_HEADERS} )
set_property(TARGET ${BIN_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${BIN_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_target_properties( ${BIN_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_FOLDER}
)	
message(STATUS "Executables will be placed in ${OUTPUT_FOLDER}.")
file(INSTALL res DESTINATION ${OUTPUT_FOLDER})

TARGET_LINK_LIBRARIES(${BIN_NAME} opencv_core opencv_imgproc opencv_video opencv_objdetect opencv_highgui opencv_gpu opencv_ml ${LIBSVM_LIBRARY})

