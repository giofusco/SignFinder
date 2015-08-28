#ifndef __WICAN_UTILS__
#define __WICAN_UTILS__


#include <jni.h>

/**
 *  A utility class for JNI wrapper
 */
struct JavaInfo{
	JNIEnv *env;
	jclass clazz;
	jobject thisObj;

	JavaInfo(){}
	/**
	 * save env, thisObj of Java, and find the corresponding class
	 */
	void init(JNIEnv *env,jobject thisObj){
		this->env = env;
		this->thisObj = thisObj;
		clazz = env->GetObjectClass(thisObj);
	}

	/**
	 * get Java object integer field value
	 */
	int getInt(const char * name){
		jfieldID fidNumber = env->GetFieldID(clazz,name,"I");
		return env->GetIntField(thisObj, fidNumber);
	}
	void setInt(const char * name, int val){
		jfieldID fidNumber = env->GetFieldID(clazz,name,"I");
		env->SetIntField(thisObj, fidNumber, val);
	}

	/**
	 * call member function "displayVideo" of the Java object;
	 */
	void call_displayVideo(){
		jmethodID id = env->GetMethodID(clazz, "displaVideo", "()V");
		env->CallVoidMethod(thisObj, id, 2);
	}
};


#endif
