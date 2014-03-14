#include <opencv2/highgui/highgui.hpp>
#include "recognizer.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
using namespace std;

#define datpath "/mnt/B/datasets/production/model.dat"
int main(int argc, char** argv){
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	Recognizer rec;
	//rec.cleanSHM();
	//rec.loadModel(datpath);
	rec.loadModelFromSHM();
	gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	rec.evaluate();
	cout<<"Model loaded in "<<elapsed<<" seconds"<<endl;		
	int* ids = new int[3];
	int* sims = new int[3];
	string* paths = new string[3];	
	while(true){
		cout<<"please input test img"<<endl;
		string filename, simpath;
		cin>>filename;

		int num = rec.classify(filename.data(), 3, ids, sims, paths);
		cout<<num<<endl;
		for (int i = 0; i < num; i++){
			cout<<"Prediction "<<i<<" "<<ids[i]<<" similarity: "<<sims[i]<<endl;;
			cout<<"Most similar image: "<<paths[i]<<endl;
		}
	}
	return 0;
}
