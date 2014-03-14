#include <opencv2/highgui/highgui.hpp>
#include "recognizer.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
using namespace std;

#define datpath "/mnt/B/datasets/baidu180/model.dat"
int main(int argc, char** argv){
	struct timeval begin, end;
	
	Recognizer rec;
	//rec.cleanSHM();
	//rec.loadModel(datpath);
	rec.loadModelFromSHM();
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
			cout<<"Prediction "<<i<<" "<<ids[i]<<" ";
			cout<<"Most similar image: "<<paths[i]<<endl;
		}
	}
	return 0;
}
