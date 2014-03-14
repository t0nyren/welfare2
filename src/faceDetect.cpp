#include <opencv2/highgui/highgui.hpp>
#include "detector.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
using namespace std;

int main(int argc, char** argv){
	Detector detector;
	
	if (argc != 2){
		cout<<"Usage: detect img"<<endl;
		return -1;
	}
	cout<<endl<<"--------------test--------------"<<endl;
	string img3= argv[1];
	Mat landmarks;
	int* pose = new int[3];	
	Mat face3 = detector.detectNorm(img3, 144, 192, 50, landmarks, 49, true);
	if (!face3.empty()){
		imwrite( "./tmp/face.jpg" , face3 );
		cout<<"Image saved to ./tmp/face.jpg"<<endl;
	}
	else{
		cout<<"Face detect failed"<<endl;
	}
	return 0;
}
