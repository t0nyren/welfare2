#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "recognizer.h"
#include <vector>
#include <deque>
#include <ctime>
#include <fstream>
using namespace std;

#define modelpath "/mnt/B/datasets/experiment/model"
#define csvpath "/mnt/B/datasets/experiment/people.csv"
#define testfile "test.mp4"

#define REALTIME

Recognizer rec;
int main(int argc, char** argv){
	rec.loadModel(modelpath);
	//string filename("../../data/vid.wmv");
	cout<<"finish loading model"<<endl;


#ifdef REALTIME
	// use the first camera it finds
	cv::VideoCapture cap(0); 
#endif 

#ifdef VIDEO
	string filename(testfile);
	cv::VideoCapture cap(filename); 
#endif
	if(!cap.isOpened())  
		return -1;

	int key = 0;
	bool isDetect = true;
	int maxNumFaces = 3;
	string winname("Kuaibo Tracker");
	cv::namedWindow(winname,cv::WINDOW_NORMAL);
	//string target("Face Recognition: Target");
	string* references = new string[maxNumFaces];
	for (int i = 0; i < maxNumFaces; i++){
		char buf[128];
		sprintf(buf,"Match %d", i+1);
		references[i] = buf;
		cv::namedWindow(references[i],cv::WINDOW_NORMAL);
	}
	cout<<"windows created"<<endl;
	ifstream fin;
	fin.open(csvpath);
	if (fin.fail()){
		cout<<"Cannot open name file"<<endl;
	}
	char line[1024];
	fin.getline(line,1024);
	vector<string> names;
	while(!fin.eof()){
		char *tok;
		tok = strtok(line, ",");
		int id = atoi(tok);
		tok = strtok(NULL, ",");		
		string name(tok);
		names.push_back(name);
		fin.getline(line,1024);
	}

	char buffer [50];
	int count = 1;

	
	int numFacesRet = 0;
	Mat* faceArray = new Mat[maxNumFaces];
	Mat* landmarksArray = new Mat[maxNumFaces];
	Mat* landmarksOriginArray = new Mat[maxNumFaces];
	float* facex = new float[maxNumFaces];
	float* facey = new float[maxNumFaces];
	int** ids = new int*[maxNumFaces];
	int** sims = new int*[maxNumFaces];
	string** paths = new string*[maxNumFaces];
	int* num = new int[maxNumFaces];
	for (int i = 0; i < maxNumFaces; i++){
		ids[i] = new int[2];
		sims[i] = new int[2];
		paths[i] = new string[2];
	}
	
	deque<Mat> matches;
	deque<int> matchIds;
	while (key!=27) // Press Esc to quit
	{
		for (int i = 0; i < maxNumFaces; i++){
			facex[i] = 0;
			facey[i] = 0;
		}
		cv::Mat frame;
		cap >> frame; // get a new frame from camera
		if (frame.rows == 0 || frame.cols == 0)
			break;


		// face detection
		Mat landmarks_origin;
		Mat landmarks;
		//Mat face = rec.getFace(frame, landmarks_origin, landmarks);
		//int num = rec.classify(face, landmarks, 1, ids, sims, paths);
		rec.getFaces(frame, maxNumFaces, numFacesRet, faceArray, landmarksOriginArray, landmarksArray );
		//cout<<numFacesRet<<endl;
		for(int i = 0; i < numFacesRet; i++){
			num[i] = rec.classify(faceArray[i], landmarksArray[i], 1, ids[i], sims[i], paths[i]);
		}

		// if no face found, do nothing
		
		if (faceArray[0].empty()) {
			cv::imshow(winname,frame);
			key = cv::waitKey(5);
			continue ;
		}
		else
		{
			// plot facial landmarks
			for (int iface = 0; iface < numFacesRet; iface++){
				for (int i = 0 ; i < landmarksOriginArray[iface].cols ; i++){
					cv::circle(frame,cv::Point((int)landmarksOriginArray[iface].at<float>(0,i), (int)landmarksOriginArray[iface].at<float>(1,i)), 1, cv::Scalar(0,255,0), -1);
					if (landmarksOriginArray[iface].at<float>(0,i) > facex[iface]){
						facex[iface] = landmarksOriginArray[iface].at<float>(0,i);
					}

					if (landmarksOriginArray[iface].at<float>(1,i) > facey[iface]){
						facey[iface] = landmarksOriginArray[iface].at<float>(1,i);
					}

				}
			}
			
			
			int lineType = 8;
			for (int i = 0; i < numFacesRet; i++){
				if (num[i] > 0 && sims[i][0] > 80){
					Point org(facex[i], facey[i] + 10);
					putText( frame, names[ids[i][0]-1], org, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, lineType );
					Mat refFace = imread(paths[i][0], CV_LOAD_IMAGE_COLOR);
					Mat match(refFace.rows, refFace.cols*2, CV_8UC3);
					Mat left(match, Rect(0,0, faceArray[i].cols, faceArray[i].rows));
					Mat right(match, Rect(faceArray[i].cols,0, faceArray[i].cols, faceArray[i].rows));
					faceArray[i].copyTo(left);
					refFace.copyTo(right);
					putText( match, names[ids[i][0]-1], Point(10,30), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, lineType );
					if (matchIds.size() == 0 || matchIds[0] != ids[i][0]){
						matches.push_front(match);
						matchIds.push_front(ids[i][0]);
					}
					else{
						for (int j = 0; j < matchIds.size(); j++){
							if (matchIds[j] == ids[i][0]){
								matches[j] = match;
							}
						}
					}
					int n=sprintf (buffer, "../matches/match20140401%d.jpg", count);
					imwrite(buffer, match);
					//cv::imshow(references[i],match);
				}
			}

			while(matches.size() > maxNumFaces){
				matches.pop_back();
				matchIds.pop_back();
			}
			for (int i = 0; i < matches.size(); i++){
				cv::imshow(references[i], matches[i]);
			}
		}

		/*for (int i = 0; i < maxNumFaces; i++){
			faceArray[i].release();
		}*/
		cv::imshow(winname,frame);	
		key = cv::waitKey(5);
	}

	return 0;
}
