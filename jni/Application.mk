# use this to select gcc instead of clang
APP_ABI := armeabi-v7a
NDK_TOOLCHAIN_VERSION := 4.9

APP_CPPFLAGS += -std=c++11 -fexceptions  -frtti
APP_STL := gnustl_static
