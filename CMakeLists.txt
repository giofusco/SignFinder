cmake_minimum_required(VERSION 2.8) 
PROJECT(SignFinder)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "compile flags" FORCE)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

set( NAME_SRC
    src/main.cpp   
    src/DetectionParams.cpp
    src/ObjDetector.cpp
    src/MedianFlowTracker.hpp
    src/MedianFlowTracker.cpp
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
find_package(LibSVM)
if (LIBSVM_FOUND)
  message(STATUS "LibSVM Found")
  message(STATUS "LibSVM_LIBRARIES = ${LIBSVM_LIBRARIES}")
  message(STATUS "LibSVM_INCLUDE_DIRS = ${LIBSVM_INCLUDE_DIRS}")
  include_directories(${LIBSVM_INCLUDE_DIRS})
else()
  set(LIBSVM_LIBRARY "")
  set(LIBSVM_DIR "" CACHE FILEPATH "Path to libsvm includes")
  include_directories(${LIBSVM_DIR})
  list(APPEND NAME_SRC ${LIBSVM_DIR}/svm.cpp)
endif()

#Copy resources
set(BIN_NAME ${PROJECT_NAME})
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message(STATUS "Debug mode")
    set(TARGET_NAME ${TARGET_NAME}d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message(STATUS "Release mode")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${HEADER_DIR})
file(GLOB NAME_HEADERS "${HEADER_DIR}/*.h" "${HEADER_DIR}/*.hpp")

set(OUTPUT_FOLDER ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_FOLDER})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_FOLDER})
add_executable( ${BIN_NAME} ${NAME_SRC} ${NAME_HEADERS} )
#set_property(TARGET ${BIN_NAME} PROPERTY CXX_STANDARD 11)
#set_property(TARGET ${BIN_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_target_properties( ${BIN_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_FOLDER}
)	
message(STATUS "Executables will be placed in ${OUTPUT_FOLDER}.")
file(INSTALL res DESTINATION ${OUTPUT_FOLDER})

# Process version.h file.
configure_file(
  "${HEADER_DIR}/version.h.in"
  "${PROJECT_BINARY_DIR}/version.h"
  )
include_directories(${PROJECT_BINARY_DIR})


TARGET_LINK_LIBRARIES(${BIN_NAME} opencv_core opencv_imgproc opencv_video opencv_objdetect opencv_highgui opencv_gpu opencv_ml ${LIBSVM_LIBRARY})

# add a target to generate API documentation with Doxygen
# Thanks to https://www.tty1.net/blog/2014/cmake-doxygen_en.html
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})

if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${PROJECT_BINARY_DIR}/Doxyfile)
    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()
