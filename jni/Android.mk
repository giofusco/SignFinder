LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_SDK_JNI = C:\\OpenCV-2.4.10-android-sdk\\sdk\\native\\jni
include $(OPENCV_SDK_JNI)/OpenCV.mk

LOCAL_MODULE    := WicabLib
LOCAL_SRC_FILES := svm.cpp ObjDetector.cpp DetectionParams.cpp wicabLibJni.cpp

LOCAL_C_INCLUDES += $(OPENCV_SDK_JNI)/include

include $(BUILD_SHARED_LIBRARY)
