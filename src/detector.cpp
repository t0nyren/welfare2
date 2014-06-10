#include "detector.h"
//#include "log.h"
#include <ctime>
#include <sys/time.h>
#include <iostream>
#include <fstream>

Detector::Detector(){
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	ifstream fin;
	fin.open("detector.cfg");
	if (fin.fail()){
		cout<<"Cannot open detector configuration file"<<endl;
		exit(1);
	}
	
	string cascade_szu;
	string cascade_opencv;
	string intradetect;
	string intratrack;
	dtype = UNKNOWN;
	//parse parameters
	char line[1024];
	fin.getline(line,1024);
	string dt;
	while(!fin.eof()){
		char *tok;
		char *p;
		tok = strtok_r(line, " ", &p);
		if (strcmp(tok,"TYPE")==0){
			dt = strtok_r(NULL, " ", &p);
		}		
		if (strcmp(tok,"CASCADE_OPENCV")==0){
			cascade_opencv = strtok_r(NULL, " ", &p);
		}
		else if (strcmp(tok,"CASCADE_SZU")==0){
			cascade_szu = strtok_r(NULL, " ", &p);
		}		
		else if (strcmp(tok,"INTRADETECT")==0){
			intradetect = strtok_r(NULL, " ", &p);
		}
		else if (strcmp(tok,"INTRATRACK")==0){
			intratrack = strtok_r(NULL, " ", &p);
		}		
		fin.getline(line,1024);
	}
	if (strcmp(dt.data(),"SZU")==0){
		faceCascade = LoadMBLBPCascade(cascade_szu.data());
		dtype = DSZU;
	}
	else if (strcmp(dt.data(),"OPENCV")==0){
		HaarCascade = (CvHaarClassifierCascade*)cvLoad(cascade_opencv.data(), 0, 0, 0);
		dtype = DOPENCV;
	}
	else{
		cout<<"Unknown detector cascade type"<<dtype<<endl;
		exit(1);
	}
	if(faceCascade == NULL && HaarCascade == NULL)
    {
        printf("Couldn't load Face detector '%s'\n", cascade_szu.data());
        exit(1);
    }
	
	xxd = new XXDescriptor(4);
	faceLandmark = new FaceAlignment(intradetect.data(), intratrack.data(), xxd);
	if (!faceLandmark->Initialized()) {
		cerr << "FaceAlignment cannot be initialized." << endl;
		exit(1);
	}
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	cout<<"Detection initialized in "<<elapsed<<" seconds"<<endl;
}

Mat Detector::detect(const string imgname, int numLandmarks){
	Mat resized;
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	
	IplImage *frame = cvLoadImage(imgname.data(), 2|4);
	if (frame == NULL)
    {
      fprintf(stderr, "Cannot open image %s.Returning empty Mat...\n", imgname.data());
      return resized;
    }
	
	else if (frame->width < 50 || frame->height < 50)
    {
      fprintf(stderr, "image %s too small.Returning empty Mat...\n", imgname.data());
      cvReleaseImage(&frame);
	  return resized;
    }	
	else if (frame->width > 100000 || frame->height > 100000)
    {
      fprintf(stderr, "image %s too large.Returning empty Mat...\n", imgname.data());
      cvReleaseImage(&frame);
	  return resized;
    }
	
	// convert image to grayscale
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
    cvConvertImage(frame, frame_bw);
	Mat frame_mat(frame, 1);
	// Smallest face size.
    CvSize minFeatureSize = cvSize(100, 100);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;
	
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    // Detect all the faces in the greyscale image.
	if (dtype == DSZU){
		rects = MBLBPDetectMultiScale(frame_bw, faceCascade, storage, 1229, 1, 50, 500);
	}
	else if (dtype == DOPENCV){
		rects = cvHaarDetectObjects(frame_bw, HaarCascade, storage, search_scale_factor, 2, flags, minFeatureSize);
	}
	else{
		cout<<"Unknown detector type: "<<dtype<<endl;
		return resized;
	}
	gettimeofday(&end, NULL);	
    double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	
	//nFaces = rects->total;
	//cout<<"Face detected in "<<elapsed<<" seconds, faces: "<<nFaces<<endl;	
	gettimeofday(&begin, NULL);
	if (nFaces != 1){
		cvReleaseMemStorage(&storage);
		cvReleaseImage(&frame_bw);
		cvReleaseImage(&frame);	
		return resized;
	}
		
	int iface = 0;
	CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
	
	//Face landmark detection
	float score, notFace = 0.5;
	Mat X;
	Rect rect(r->x, r->y, r->width, r->height);
	INTRAFACE::HeadPose hp;

	gettimeofday(&begin, NULL);
	if (faceLandmark->Detect(frame_mat, rect, X, score) == INTRAFACE::IF_OK)
	{
		faceLandmark->EstimateHeadPose(X,hp);
		// only draw valid faces
		if (score >= notFace) {
			for (int i = 0 ; i < X.cols ; i++)
				cv::circle(frame_mat,cv::Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)), 2, cv::Scalar(0,255,0), -1);
		}
		else{
			//cout<<"False positive face"<<endl;
			return resized;
		}
	}
	else
	{
		//cout<<"Landmark detection failed"<<endl;
		return resized;
	}
	gettimeofday(&end, NULL);	
    elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Landmarks detected in "<<elapsed<<" seconds"<<endl;
	return frame_mat;
}

Mat Detector::detect(const string imgname, Mat& landmarks, int* pose, int numLandmarks){
	Mat resized;
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	
	IplImage *frame = cvLoadImage(imgname.data(), 2|4);
	if (frame == NULL)
    {
      fprintf(stderr, "Cannot open image %s.Returning empty Mat...\n", imgname.data());
	  //LOGGER(INFO,"Detector","detect", "Cannot open image");
      return resized;
    }
	
	else if (frame->width < 50 || frame->height < 50)
    {
      fprintf(stderr, "image %s too small.Returning empty Mat...\n", imgname.data());
      cvReleaseImage(&frame);
	  return resized;
    }	
	else if (frame->width > 100000 || frame->height > 100000)
    {
      fprintf(stderr, "image %s too large.Returning empty Mat...\n", imgname.data());
      cvReleaseImage(&frame);
	  return resized;
    }
	
	// convert image to grayscale
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
	cvConvertImage(frame, frame_bw);
	Mat frame_mat(frame, 1);
	// Smallest face size.
    CvSize minFeatureSize = cvSize(100, 100);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;
	
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    // Detect all the faces in the greyscale image.
	if (dtype == DSZU){
		rects = MBLBPDetectMultiScale(frame_bw, faceCascade, storage, 1229, 1, 50, 500);
	}
	else if (dtype == DOPENCV){
		rects = cvHaarDetectObjects(frame_bw, HaarCascade, storage, search_scale_factor, 2, flags, minFeatureSize);
	}
	else{
		cout<<"Unknown detector type: "<<dtype<<endl;
		return resized;
	}
	gettimeofday(&end, NULL);	
    double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Face detected in "<<elapsed<<" seconds"<<endl;	
	gettimeofday(&begin, NULL);
	nFaces = rects->total;

	if (nFaces != 1){
		cvReleaseMemStorage(&storage);
		cvReleaseImage(&frame_bw);
		cvReleaseImage(&frame);	
		return resized;
	}
		
	int iface = 0;
	CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
	
	//Face landmark detection
	float score, notFace = 0.5;
	Mat X;
	Rect rect(r->x, r->y, r->width, r->height);
	INTRAFACE::HeadPose hp;

	gettimeofday(&begin, NULL);
	if (faceLandmark->Detect(frame_mat, rect, X, score) == INTRAFACE::IF_OK)
	{
		faceLandmark->EstimateHeadPose(X,hp);
		// only draw valid faces
		if (score >= notFace) {
			for (int i = 0 ; i < X.cols ; i++)
				cv::circle(frame_mat,cv::Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)), 1, cv::Scalar(0,255,0), -1);
		}
		else{
			//cout<<"False positive face"<<endl;
			return resized;
		}
	}
	else
	{
		//cout<<"Landmark detection failed"<<endl;
		return resized;
	}
	gettimeofday(&end, NULL);	
    elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Landmarks detected in "<<elapsed<<" seconds"<<endl;
	gettimeofday(&begin, NULL);
	//imwrite( "./tmp/face.jpg" , frame_mat );
	//Face alignment
	int normSize = 100;
	int patchSize = 30;
	//calculate bounding box
	double angle = hp.angles[0];
	Mat rotated = rotateImage(frame_mat, -angle);
	
	//rotate landmarks
	float minx = 10000;
	float maxx = 0;
	float miny = 10000;
	float maxy = 0;
	
	for (unsigned int i = 0; i < X.cols; i++){
		//cout<<X.at<float>(0,i)<<" "<<X.at<float>(1,i)<<endl;
		//rotatePoint(frame_mat, angle, X.at<float>(0,i), X.at<float>(1,i), X.at<float>(0,i), X.at<float>(1,i));
		//cout<<X.at<float>(0,i)<<" "<<X.at<float>(1,i)<<endl;
		//cout<<"bbox: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
		//cout<<landmarks[i]<<" "<<landmarks[i+1]<<endl;
		circle(rotated, Point(X.at<float>(0,i), X.at<float>(1,i)), 2, Scalar(255,0,0));
		if (X.at<float>(0,i)  < minx ){
			minx = X.at<float>(0,i) ;
		}
		if (X.at<float>(0,i) > maxx){
			maxx = X.at<float>(0,i) ;
		}
		
		if (X.at<float>(1,i)  < miny){
			miny = X.at<float>(1,i) ;
		}
		
		if (X.at<float>(1,i) > maxy){
			maxy = X.at<float>(1,i);
		}
	}
	//TODO: validate min max

	//extend to rectangle
	float margin = (maxx - minx) - (maxy - miny);
	if (margin > 0){
		maxy += margin/2;
		miny -= margin/2;
	}
	else{
		maxx -= margin/2;
		minx += margin/2;
	}
	//cout<<"bbox: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
	//extend patch region
	int region = (((double)patchSize)/2 + 1)/(normSize - patchSize - 2)*(maxx - minx);
	//cout<<"region: "<<region<<endl;
	minx = minx - region - 1;
	maxx = maxx + region + 1;
	miny = miny - region - 1;
	maxy = maxy + region + 1;
	//cout<<"bounds: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
	if (minx < 0 || miny < 0 || maxx > frame_mat.cols || maxy > frame_mat.rows){
		//cout<<"Bounding box out of bound: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
		return resized;
	}
	//resize
	Mat roi(rotated, Rect(minx,miny,maxx-minx,maxy-miny));
	resize(roi, resized, Size(normSize, normSize));
	for (unsigned int i = 0; i < X.cols; i++){
		X.at<float>(0,i) = (X.at<float>(0,i) - minx)/(maxx-minx)*normSize;
		X.at<float>(1,i) = (X.at<float>(1,i)- miny)/(maxy-miny)*normSize;
		circle(resized, Point(X.at<float>(0,i), X.at<float>(1,i)), 2, Scalar(255,0,0));
	}
	//return resized;
	
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&frame_bw);
	cvReleaseImage(&frame);
	
	gettimeofday(&end, NULL);	
    elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Face aligned in "<<elapsed<<" seconds"<<endl;	
	return resized;
}



Mat Detector::detectNorm(const string filename, const float faceWidth, const float faceHeight, const float patchSize, Mat& landmarks, int numLandmarks, bool mirror, bool showLandmark){
	Mat resized;
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	
	IplImage *frame = cvLoadImage(filename.data(), 2|4);
	if (frame == NULL)
    {
      fprintf(stderr, "Cannot open image %s.Returning empty Mat...\n", filename.data());
      return resized;
    }
	
	else if (frame->width < 50 || frame->height < 50)
    {
      fprintf(stderr, "image %s too small.Returning empty Mat...\n", filename.data());
      cvReleaseImage(&frame);
	  return resized;
    }	
	else if (frame->width > 100000 || frame->height > 100000)
    {
      fprintf(stderr, "image %s too large.Returning empty Mat...\n", filename.data());
      cvReleaseImage(&frame);
	  return resized;
    }
	
	// convert image to grayscale
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
    cvConvertImage(frame, frame_bw);
	Mat frame_mat(frame, 1);
	// Smallest face size.
    CvSize minFeatureSize = cvSize(100, 100);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;
	
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    // Detect all the faces in the greyscale image.
	if (dtype == DSZU){
		rects = MBLBPDetectMultiScale(frame_bw, faceCascade, storage, 1229, 1, 50, 700);
	}
	else if (dtype == DOPENCV){
		rects = cvHaarDetectObjects(frame_bw, HaarCascade, storage, search_scale_factor, 2, flags, minFeatureSize);
	}
	else{
		cout<<"Unknown detector type: "<<dtype<<endl;
		return resized;
	}
	gettimeofday(&end, NULL);	
    double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Face detected in "<<elapsed<<" seconds"<<endl;
	gettimeofday(&begin, NULL);
	nFaces = rects->total;

	bool isdetect = false;
	INTRAFACE::HeadPose hp;
	for (int iface = 0; iface < nFaces; iface++){
		int iface = 0;
		CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
		
		//Face landmark detection
		float score, notFace = 0.5;
		Rect rect(r->x, r->y, r->width, r->height);

		gettimeofday(&begin, NULL);
		if (faceLandmark->Detect(frame_mat, rect, landmarks, score) == INTRAFACE::IF_OK)
		{
			faceLandmark->EstimateHeadPose(landmarks,hp);
			// only draw valid faces
			if (score >= notFace) {
				if (showLandmark){
					for (int i = 0 ; i < landmarks.cols ; i++)
						cv::circle(frame_mat,cv::Point((int)landmarks.at<float>(0,i), (int)landmarks.at<float>(1,i)), 2, cv::Scalar(0,255,0), -1);
				}
				isdetect = true;
				break;
			}
			else{
				//cout<<"False positive face"<<endl;
				continue;
			}
		}
		else
		{
			//cout<<"Landmark detection failed"<<endl;
			continue;
		}
	}
	
	if (!isdetect){
		cvReleaseMemStorage(&storage);
		cvReleaseImage(&frame_bw);
		cvReleaseImage(&frame);
		return resized;	
	}
	
	gettimeofday(&end, NULL);	
    elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Landmarks detected in "<<elapsed<<" seconds"<<endl;
	gettimeofday(&begin, NULL);
	//imwrite( "./tmp/face.jpg" , frame_mat );
	//Face alignment
	int normSize = 100;
	double roll = hp.angles[0];
	double pitch = hp.angles[1];
	Mat rotated = rotateImage(frame_mat, -roll);
	
	//rotate landmarks
	float minx = 10000;
	float maxx = 0;
	float miny = 10000;
	float maxy = 0;
	
	for (unsigned int i = 0; i < landmarks.cols; i++){
		//cout<<X.at<float>(0,i)<<" "<<X.at<float>(1,i)<<endl;
		rotatePoint(frame_mat, roll, (double)landmarks.at<float>(0,i), (double)landmarks.at<float>(1,i), landmarks.at<float>(0,i), landmarks.at<float>(1,i));
		//cout<<X.at<float>(0,i)<<" "<<X.at<float>(1,i)<<endl;
		//cout<<"bbox: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
		//cout<<landmarks[i]<<" "<<landmarks[i+1]<<endl;
		//circle(rotated, Point(landmarks.at<float>(0,i), landmarks.at<float>(1,i)), 2, Scalar(255,0,0));
		if (landmarks.at<float>(0,i)  < minx ){
			minx = landmarks.at<float>(0,i) ;
		}
		if (landmarks.at<float>(0,i) > maxx){
			maxx = landmarks.at<float>(0,i) ;
		}
		
		if (landmarks.at<float>(1,i)  < miny){
			miny = landmarks.at<float>(1,i) ;
		}
		
		if (landmarks.at<float>(1,i) > maxy){
			maxy = landmarks.at<float>(1,i);
		}
	}
	
	//TODO: validate min max

	//extend to rectangle
	float whRatio = (faceWidth - patchSize)/(faceHeight - patchSize);
	float curRatio = (maxx-minx)/(maxy-miny);
	if (curRatio >= whRatio){
		//fit width, extend height
		float region =  (maxx-minx)/whRatio -(maxy-miny);
		miny -= region/2;
		maxy += region/2;
	}
	else{
		//fit height, extend width
		float region = (maxy - miny)*whRatio - (maxx-minx);
		minx -= region/2;
		maxx += region/2;
	}
	//extend scaled patch
	float px = (maxx-minx)/(faceWidth-patchSize)*patchSize/2;
	float py = (maxy-miny)/(faceHeight-patchSize)*patchSize/2;
	maxx += px + 2;
	minx -= px + 2;
	maxy += py + 2;
	miny -= py + 2;
	if (minx < 0 || miny < 0 || maxx > rotated.cols || maxy > rotated.rows){
		//extend img
		RNG rng(12345);
		//Scalar value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
		Scalar value = Scalar( 0, 0, 0);
		copyMakeBorder( rotated, rotated, patchSize, patchSize, patchSize, patchSize, BORDER_CONSTANT, value );
		
		minx += patchSize;
		miny += patchSize;
		maxx += patchSize;
		maxy += patchSize;
		for (unsigned int i = 0; i < landmarks.cols; i++){
			landmarks.at<float>(0,i) = landmarks.at<float>(0,i) + patchSize;
			landmarks.at<float>(1,i) = landmarks.at<float>(1,i) + patchSize;
		}
		if (minx < 0 || miny < 0 || maxx > rotated.cols || maxy > rotated.rows){
			cout<<"Bounding box out of bound: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
			cout<<"num of cols: "<<rotated.cols<<" "<<"num of rows:"<<rotated.rows<<endl;
			return resized;
		}
		//return resized;
	}
	
	//resize
	//cout<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
	Mat roi(rotated, Rect(minx,miny,maxx-minx,maxy-miny));
	resize(roi, resized, Size(faceWidth, faceHeight));
	//cout<<"Num of landmarks: "<<landmarks.cols<<" type: "<<landmarks.type()<<endl;
	for (unsigned int i = 0; i < landmarks.cols; i++){
		landmarks.at<float>(0,i) = (landmarks.at<float>(0,i) - minx)/(maxx-minx)*faceWidth;
		landmarks.at<float>(1,i) = (landmarks.at<float>(1,i)- miny)/(maxy-miny)*faceHeight;
	}

	if (mirror && pitch < 0){
		flip(resized, resized, 1);
		for (int i = 0; i < landmarks.cols; i++){
			landmarks.at<float>(0,i) = resized.cols - landmarks.at<float>(0,i);
		}
	}	
	
	if (numLandmarks == 5){
		Mat newlandmarks(2, 5, CV_64F);
		//left eye
		for (int i = 0; i < 2; i++)
			newlandmarks.at<float>(i,0) = (landmarks.at<float>(i,19) + landmarks.at<float>(i,22))/2;
			
		//right eye
		for (int i = 0; i < 2; i++)
			newlandmarks.at<float>(i,1) = (landmarks.at<float>(i,25) + landmarks.at<float>(i,28))/2;		
		
		//nose
		for (int i = 0; i < 2; i++)
			newlandmarks.at<float>(i,2) = landmarks.at<float>(i,13) ;
			
		//mouth left
		for (int i = 0; i < 2; i++)
			newlandmarks.at<float>(i,3) = landmarks.at<float>(i,31) ;		
		
		//mouth right
		for (int i = 0; i < 2; i++)
			newlandmarks.at<float>(i,4) = landmarks.at<float>(i,37) ;
		//landmarks.resize(5);
		//for (int i = 0; i < 2; i++){
		//	for (int j = 0; j < 5; j++){
		//		landmarks.at<float>(i,j) = newlandmarks.at<float>(i,j);
		//	}
		//}
		landmarks = newlandmarks;
	}
	//eye, eyebrow, eye-band,  nose-ver, mouth, left face, right face, jaw, forehead, nose-hor
	//40x20x2, 40x20x2, 100x40, 40x60, 60x40, 40x60,40x60, 50x20, 100x40, 60x40
	else if (numLandmarks == 36 || numLandmarks == 24){
		Mat newlandmarks(2, 12, CV_64F);
		float x1 = 10000;
		float y1 = 10000;
		float x2 = 0;
		float y2 = 0;
		//left eye
		for (int i = 19; i <= 24; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}
		newlandmarks.at<float>(0,0) = (x1+x2)/2;
		newlandmarks.at<float>(1,0) = (y1+y2)/2;
		
		//right eye
		x1 = 10000;
		y1 = 10000;
		x2 = 0;
		y2 = 0;
		for (int i = 25; i <= 30; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}
		newlandmarks.at<float>(0,1) = (x1+x2)/2;
		newlandmarks.at<float>(1,1) = (y1+y2)/2;

		//left eyebrow
		x1 = 10000;
		y1 = 10000;
		x2 = 0;
		y2 = 0;		
		for (int i = 0; i <= 4; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}
		newlandmarks.at<float>(0,2) = (x1+x2)/2;
		newlandmarks.at<float>(1,2) = (y1+y2)/2;
		
		//right eyebrow
		x1 = 10000;
		y1 = 10000;
		x2 = 0;
		y2 = 0;
		for (int i = 5; i <= 9; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}
		newlandmarks.at<float>(0,3) = (x1+x2)/2;
		newlandmarks.at<float>(1,3) = (y1+y2)/2;		
		
		//eye band
		x1 = 10000;
		y1 = 10000;
		x2 = 0;
		y2 = 0;		
		for (int i = 0; i <= 9; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}
		for (int i = 19; i <= 30; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}
		newlandmarks.at<float>(0,4) = (x1+x2)/2;
		newlandmarks.at<float>(1,4) = (y1+y2)/2;
		
		
		
		//nose vertical
		x1 = 10000;
		y1 = 10000;
		x2 = 0;
		y2 = 0;		
		for (int i = 10; i <= 18; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}		
		newlandmarks.at<float>(0,5) = (x1+x2)/2;
		newlandmarks.at<float>(1,5) = (y1+y2)/2;
		
		//mouth
		x1 = 10000;
		y1 = 10000;
		x2 = 0;
		y2 = 0;		
		for (int i = 31; i <= 42; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}		
		newlandmarks.at<float>(0,6) = (x1+x2)/2;
		newlandmarks.at<float>(1,6) = (y1+y2)/2;

		//left face
		x1 = 10000;
		y1 = 10000;
		x2 = 0;
		y2 = 0;
		for (int i = 0; i <= 4; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}
		for (int i = 10; i <= 14; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}		
		newlandmarks.at<float>(0,7) = (x1+x2)/2;
		newlandmarks.at<float>(1,7) = (y1+y2)/2;		

		//right face
		x1 = 10000;
		y1 = 10000;
		x2 = 0;
		y2 = 0;		
		for (int i = 5; i <= 9; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}
		for (int i = 10; i <= 14; i++){
			if (landmarks.at<float>(0,i) < x1){
				x1 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(0,i) > x2){
				x2 = landmarks.at<float>(0,i);
			}
			if (landmarks.at<float>(1,i) < y1){
				y1 = landmarks.at<float>(1,i);
			}
			if (landmarks.at<float>(1,i) > y2){
				y2 = landmarks.at<float>(1,i);
			}
		}			
		newlandmarks.at<float>(0,8) = (x1+x2)/2;
		newlandmarks.at<float>(1,8) = (y1+y2)/2;		
		
		
		//chin
		newlandmarks.at<float>(0,9) = landmarks.at<float>(0,40);
		newlandmarks.at<float>(1,9) = landmarks.at<float>(1,40) + 10;
		
		//forehead	
		newlandmarks.at<float>(0,10) = (landmarks.at<float>(0,2)+landmarks.at<float>(0,7))/2;
		newlandmarks.at<float>(1,10) = (landmarks.at<float>(1,2)+landmarks.at<float>(1,7))/2 - 10;

		//nose horizontal
		newlandmarks.at<float>(0,11) = landmarks.at<float>(0,16);
		newlandmarks.at<float>(1,11) = landmarks.at<float>(1,16);
		
		landmarks = newlandmarks;		
	}
	

	
	if (showLandmark){
		for (unsigned int i = 0; i < landmarks.cols; i++){
			circle(resized, Point(landmarks.at<float>(0,i), landmarks.at<float>(1,i)), 3, Scalar(255,0,0));
		}
	}
	//clear stuff
	
	gettimeofday(&end, NULL);	
    elapsed = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&frame_bw);
	cvReleaseImage(&frame);
	
	//cout<<"Face aligned in "<<elapsed<<" seconds"<<endl;	
	return resized;
}

Mat Detector::rotateImage(const Mat& source, double angle)
{
    Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    warpAffine(source, dst, rot_mat, source.size());
    return dst;
}

void Detector::rotatePoint(const Mat& source, double angle, const double& x1, const double& y1, float& x, float& y){
	double x0 = source.cols/2.0F;
	double y0 = source.rows/2.0F;
	x = x1 - x0;
	y = y1 - y0;
	angle = angle * PI / 180;
	double tx = x*cos(angle) - y*sin(angle);
	double ty = x*sin(angle) + y*cos(angle);
	x = tx;
	y = ty;
	x += x0;
	y += y0;
}

void Detector::detectNorm(const Mat& capframe, const float width, const float height, const float patchSize, int maxNumFaces, int& numFaces, Mat* ret, Mat* landmarks_origin, Mat* landmarks, bool mirror, int numLandmarks, bool showLandmark){
	//vector<Rect> faces;
	//face_cascade.detectMultiScale(capframe, faces, 1.2, 2, 0, Size(50,50));

	IplImage* frame = cvCloneImage(&(IplImage)capframe);
	if (frame == NULL)
    {
      fprintf(stderr, "capframe is NULL.Returning empty Mat...\n");
      return;
    }
	
	else if (frame->width < 50 || frame->height < 50)
    {
      fprintf(stderr, "capframe too small.Returning empty Mat...\n");
      cvReleaseImage(&frame);
	  return;
    }	
	else if (frame->width > 100000 || frame->height > 100000)
    {
      fprintf(stderr, "capframe too large.Returning empty Mat...\n");
      cvReleaseImage(&frame);
	  return;
    }
	
	// convert image to grayscale
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
	
    cvConvertImage(frame, frame_bw);
	Mat frame_mat(frame, 1);
	// Smallest face size.
    CvSize minFeatureSize = cvSize(100, 100);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;
	
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);
	
    // Detect all the faces in the greyscale image.
	if (dtype == DSZU){
		rects = MBLBPDetectMultiScale(frame_bw, faceCascade, storage, 1229, 1, 50, 500);
	}
	else if (dtype == DOPENCV){
		rects = cvHaarDetectObjects(frame_bw, HaarCascade, storage, search_scale_factor, 2, flags, minFeatureSize);
	}
	else{
		cout<<"Unknown detector type: "<<dtype<<endl;
		return;
	}
	
	nFaces = rects->total;

	int detectcount = 0;
	INTRAFACE::HeadPose hp; //roll, yaw, pitch
	for (int iface = 0; iface < nFaces; iface++){
		//Face landmark detection
		CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
		float score, notFace = 0.5;
		Rect rect(r->x, r->y, r->width, r->height);
		if (faceLandmark->Detect(capframe, rect, landmarks[detectcount], score) == INTRAFACE::IF_OK)
		{
			
			// only draw valid faces
			if (score >= notFace) {
				landmarks_origin[detectcount] = landmarks[detectcount].clone();
				faceLandmark->EstimateHeadPose(landmarks[detectcount],hp);
				double roll = hp.angles[0];
				double pitch = hp.angles[1];
				ret[detectcount] = norm(capframe, roll, pitch, width, height, patchSize, landmarks_origin[detectcount], landmarks[detectcount], mirror, numLandmarks, showLandmark);
				detectcount++;
				if (detectcount > maxNumFaces)
					break;
			}
			else{
				//cout<<"False positive face"<<endl;
				continue;
			}
		}
		else
		{
			//cout<<"Landmark detection failed"<<endl;
			continue;
		}
	}
	numFaces = detectcount;
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&frame_bw);
	cvReleaseImage(&frame);
}

Mat Detector::norm(const Mat& frame_mat, double roll, double pitch, const float faceWidth, const float faceHeight, const float patchSize, Mat& landmarks_orgin, Mat& landmarks, bool mirror, int numLandmarks, bool showLandmark){

	Mat rotated = rotateImage(frame_mat, -roll);
	Mat resized;
	//rotate landmarks
	float minx = 10000;
	float maxx = 0;
	float miny = 10000;
	float maxy = 0;
	
	for (unsigned int i = 0; i < landmarks.cols; i++){
		//cout<<X.at<float>(0,i)<<" "<<X.at<float>(1,i)<<endl;
		rotatePoint(frame_mat, roll, (double)landmarks.at<float>(0,i), (double)landmarks.at<float>(1,i), landmarks.at<float>(0,i), landmarks.at<float>(1,i));
		//cout<<X.at<float>(0,i)<<" "<<X.at<float>(1,i)<<endl;
		//cout<<"bbox: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
		//cout<<landmarks[i]<<" "<<landmarks[i+1]<<endl;
		//circle(rotated, Point(landmarks.at<float>(0,i), landmarks.at<float>(1,i)), 2, Scalar(255,0,0));
		if (landmarks.at<float>(0,i)  < minx ){
			minx = landmarks.at<float>(0,i) ;
		}
		if (landmarks.at<float>(0,i) > maxx){
			maxx = landmarks.at<float>(0,i) ;
		}
		
		if (landmarks.at<float>(1,i)  < miny){
			miny = landmarks.at<float>(1,i) ;
		}
		
		if (landmarks.at<float>(1,i) > maxy){
			maxy = landmarks.at<float>(1,i);
		}
	}

	//TODO: validate min max

	//extend to rectangle
	float whRatio = (faceWidth - patchSize)/(faceHeight - patchSize);
	float curRatio = (maxx-minx)/(maxy-miny);
	if (curRatio >= whRatio){
		//fit width, extend height
		float region =  (maxx-minx)/whRatio -(maxy-miny);
		miny -= region/2;
		maxy += region/2;
	}
	else{
		//fit height, extend width
		float region = (maxy - miny)*whRatio - (maxx-minx);
		minx -= region/2;
		maxx += region/2;
	}
	//extend scaled patch
	float px = (maxx-minx)/(faceWidth-patchSize)*patchSize/2;
	float py = (maxy-miny)/(faceHeight-patchSize)*patchSize/2;
	maxx += px + 2;
	minx -= px + 2;
	maxy += py + 2;
	miny -= py + 2;
	if (minx < 0 || miny < 0 || maxx > rotated.cols || maxy > rotated.rows){
		//cout<<"Bounding box out of bound: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
		//cout<<"num of cols: "<<frame_mat.cols<<" "<<"num of rows:"<<frame_mat.rows<<endl;
		//extend img
		RNG rng(12345);
		Scalar value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
		copyMakeBorder( rotated, rotated, patchSize, patchSize, patchSize, patchSize, BORDER_CONSTANT, value );
		
		minx += patchSize;
		miny += patchSize;
		maxx += patchSize;
		maxy += patchSize;
		for (unsigned int i = 0; i < landmarks.cols; i++){
			landmarks.at<float>(0,i) = landmarks.at<float>(0,i) + patchSize;
			landmarks.at<float>(1,i) = landmarks.at<float>(1,i) + patchSize;
		}
		if (minx < 0 || miny < 0 || maxx > rotated.cols || maxy > rotated.rows){
			cout<<"Bounding box out of bound: "<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
			cout<<"num of cols: "<<rotated.cols<<" "<<"num of rows:"<<rotated.rows<<endl;
			return resized;
		}
		//return resized;
	}
	
	//resize
	//cout<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
	Mat roi(rotated, Rect(minx,miny,maxx-minx,maxy-miny));

	resize(roi, resized, Size(faceWidth, faceHeight));
	//cout<<"Num of landmarks: "<<landmarks.cols<<" type: "<<landmarks.type()<<endl;
	for (unsigned int i = 0; i < landmarks.cols; i++){
		landmarks.at<float>(0,i) = (landmarks.at<float>(0,i) - minx)/(maxx-minx)*faceWidth;
		landmarks.at<float>(1,i) = (landmarks.at<float>(1,i)- miny)/(maxy-miny)*faceHeight;
	}
	
	
	if (numLandmarks == 5){
		Mat newLandmarks(2, 5, CV_64F);
		//left eye
		for (int i = 0; i < 2; i++)
			newLandmarks.at<float>(i,0) = (landmarks.at<float>(i,19) + landmarks.at<float>(i,22))/2;
			
		//right eye
		for (int i = 0; i < 2; i++)
			newLandmarks.at<float>(i,1) = (landmarks.at<float>(i,25) + landmarks.at<float>(i,28))/2;		
		
		//nose
		for (int i = 0; i < 2; i++)
			newLandmarks.at<float>(i,2) = landmarks.at<float>(i,13) ;
			
		//mouth left
		for (int i = 0; i < 2; i++)
			newLandmarks.at<float>(i,3) = landmarks.at<float>(i,31) ;		
		
		//mouth right
		for (int i = 0; i < 2; i++)
			newLandmarks.at<float>(i,4) = landmarks.at<float>(i,37) ;
			
		landmarks = newLandmarks;
	}

	//mirror
	if (mirror && pitch < 0){
		flip(resized, resized, 1);
		for (int i = 0; i < landmarks.cols; i++){
			landmarks.at<float>(0,i) = resized.cols - landmarks.at<float>(0,i);
			//landmarks.at<float>(1,i) = resized.rows - landmarks.at<float>(1,i);
		}
	}

	if (showLandmark){
		for (unsigned int i = 0; i < landmarks.cols; i++){
			circle(resized, Point(landmarks.at<float>(0,i), landmarks.at<float>(1,i)), 3, Scalar(255,0,0));
		}
	}
	return resized;
}
