package org.ski.wicablib;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.os.Bundle;
import android.os.Environment;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.hardware.Camera;
import android.util.Log;
import android.view.Menu;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.Window;
import android.view.WindowManager;
import android.widget.LinearLayout;

public class WicabCamAct extends Activity implements SurfaceHolder.Callback{
    private static final String  TAG = "WicabCam:  ";

    static String data_signature = "2015-08-27-1";
    
	int width= 640, height = 480;
	
	Bitmap bmp2Display;
	PreviewNew preview;
	Camera mCamera;
	Mat myuv;
	int x,y,w,h;
	
	Detector detector;
	static String TARGET_BASE_PATH = Environment.getExternalStorageDirectory().getPath() + "/WicabLib/";

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);	
		check4CopyAsset();
		mLoaderCallback = new BaseLoaderCallback(this) {
	        @Override
	        public void onManagerConnected(int status) {
	            switch (status) {
	                case LoaderCallbackInterface.SUCCESS:
	                {
	                    Log.i(TAG, "OpenCV loaded successfully");	
	                    myuv = new Mat(height + height/2, width, CvType.CV_8UC1);
	            		detector = new Detector(TARGET_BASE_PATH+"restroom_sign_men_config.yaml",TARGET_BASE_PATH);
	                } break;
	                default:
	                {
	                    super.onManagerConnected(status);
	                } break;
	            }
	        }
	    };
		
	    preview = new PreviewNew(this);
		bmp2Display = Bitmap.createBitmap(width,height, Bitmap.Config.ARGB_8888);

		LinearLayout layout = new LinearLayout(this);
		layout.addView(preview, width, height);
		
		layout.setGravity(LinearLayout.VERTICAL);
		setContentView(layout);
	}
	
/**
 * if the asset data is new, copy it 
 */
	void check4CopyAsset(){
		String DATA_SIGNATURE = "Data Signature";
		String dataStrName = "Data String";

		SharedPreferences settings = getSharedPreferences(DATA_SIGNATURE, 0);
		String str  = settings.getString(dataStrName, "???");	
		if (str.compareTo(data_signature)!=0){
			
			assetManager = getAssets();
			new File(TARGET_BASE_PATH).mkdirs();			
			new Thread(new Runnable(){
				@Override
				public void run() {
					copyFileOrDir("");					
				}				
			}).start();
			SharedPreferences.Editor editor = settings.edit();
			editor.putString(dataStrName, data_signature);
			editor.commit();
		}		
	}
	
	private BaseLoaderCallback  mLoaderCallback;
	@Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }
		
	int argb[] = new int[width*height];
	Camera.PreviewCallback cbWithBuffer = new Camera.PreviewCallback() {
		public void onPreviewFrame(byte[] _data, Camera _camera) {
			myuv.put(0, 0, _data);
			if (detector!=null)
				detector.detect(myuv.getNativeObjAddr(),argb);
			if (bmp2Display!=null)
				bmp2Display.setPixels(argb, 0, width, 0, 0, width, height);
			
			preview.invalidate();
			_camera.addCallbackBuffer(_data);
		}
	};
	
	public void surfaceCreated(SurfaceHolder holder) {
		mCamera = Camera.open();
		mCamera.addCallbackBuffer(new byte[width * height * 3/2]); // YUV
		mCamera.setPreviewCallbackWithBuffer(cbWithBuffer);
		try {
			mCamera.setPreviewDisplay(holder);
		} catch (IOException exception) {
			mCamera.release();
			mCamera = null;
		}
	}

	public void surfaceDestroyed(SurfaceHolder holder) {
		mCamera.stopPreview();
		mCamera.setPreviewCallbackWithBuffer(null);
		mCamera.release();
		mCamera = null;
	}

	public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
		Camera.Parameters parameters = mCamera.getParameters();
		parameters.setPreviewSize(width, height);
		mCamera.setParameters(parameters);
		mCamera.setPreviewCallbackWithBuffer(cbWithBuffer);
		mCamera.startPreview();
	}
	
	class PreviewNew extends SurfaceView {		
		PreviewNew(Context context) {
			super(context);				
			SurfaceHolder mHolder = getHolder();
			mHolder.setFixedSize(width, height);
			mHolder.addCallback(WicabCamAct.this);
			mHolder.setKeepScreenOn(false);
			setWillNotDraw(false);  // crucial,or onDraw() will not be involked
		}

		@Override
		protected void onDraw(Canvas canvas) {
			super.onDraw(canvas);
			if (bmp2Display!=null){
				canvas.drawBitmap(bmp2Display, 0,0, null);
			}
		}
	}


	static AssetManager assetManager;	
	static private void copyFileOrDir(String path) {
		String assets[] = null;
		try {
			Log.i("tag", "copyFileOrDir() " + path);
			assets = assetManager.list(path);
			if (assets.length == 0) {
				copyFile(path);
			} else {
				String fullPath = TARGET_BASE_PATH + path;
				Log.i("tag", "path=" + fullPath);
				File dir = new File(fullPath);
				if (!dir.exists() && !path.startsWith("images")
						&& !path.startsWith("sounds")
						&& !path.startsWith("webkit"))
					if (!dir.mkdirs())
						Log.i("tag", "could not create dir " + fullPath);
				for (int i = 0; i < assets.length; ++i) {
					String p;
					if (path.equals(""))
						p = "";
					else
						p = path + "/";

					if (!path.startsWith("images")
							&& !path.startsWith("sounds")
							&& !path.startsWith("webkit"))
						copyFileOrDir(p + assets[i]);
				}
			}
		} catch (IOException ex) {
			Log.e("tag", "I/O Exception", ex);
		}
	}

	static private void copyFile(String filename) {
		String newFileName = null;
		try {
			Log.i("tag", "copyFile() " + filename);
			InputStream in = assetManager.open(filename);
			newFileName = TARGET_BASE_PATH;
			if (filename.endsWith(".jpg")) // extension was added to avoid
											// compression on APK file
				newFileName += filename.substring(0, filename.length() - 4);
			else
				newFileName += filename;
			OutputStream out = new FileOutputStream(newFileName);
			byte[] buffer = new byte[1024];
			int read;
			while ((read = in.read(buffer)) != -1) 
				out.write(buffer, 0, read);			
			out.flush();
		} catch (Exception e) {
			Log.e("tag", "Exception in copyFile() of " + newFileName);
			Log.e("tag", "Exception in copyFile() " + e.toString());
		}
	}
}
