#include <opencv2/highgui/highgui.hpp>
#include "detector.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
using namespace std;

int main(int argc, char** argv){
	Detector detector;
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	if (argc != 2){
		cout<<"Usage: detect img"<<endl;
		return -1;
	}
	cout<<endl<<"--------------test--------------"<<endl;
	string img3= argv[1];
	int* pose = new int[3];
	Mat landmarks;
	//for (int i = 0; i < 100000; i++){
	Mat face3 = detector.detectNorm(img3, 144, 192, 50, landmarks, 4, true, true);
		//Mat face3 = detector.detect(img3);

	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	cout<<"Face detected in "<<elapsed<<" seconds"<<endl;	
	if (!face3.empty()){
		imwrite( "./tmp/face.jpg" , face3 );
		cout<<" Image saved to ./tmp/face.jpg"<<endl;
	}
	else{
		cout<<"Face detect failed"<<endl;
	}
	//}
	return 0;
}
