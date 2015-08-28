package org.ski.wicablib;


import org.opencv.core.Mat;

public class Detector {

	int created = 0;
	Detector(String yamlConfigFile, String classifiersFolder) throws RuntimeException {
			System.loadLibrary("WicabLib");		
		created = setClassifier(yamlConfigFile, classifiersFolder);
		if (created ==0)
			throw new RuntimeException("Detect not created");
	}

	native int setClassifier(String yamlConfigFile, String classifiersFolder);
	native int detect(long addrYuv, int [] argb4Display);
}
