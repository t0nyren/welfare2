#include <opencv2/highgui/highgui.hpp>
#include "recognizer.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
using namespace std;

#define datpath "/mnt/B/datasets/production/model.dat"
int main(int argc, char** argv){
	struct timeval begin, end;
	
	Recognizer rec;
	rec.cleanSHM();
	rec.loadModel(datpath);
	int ret = rec.writeModelToSHM();
	if (ret == 1)
		cout<<"Model wrote to SHM"<<endl;
	else
		cout<<"Model write failed"<<endl;
	return 0;
}
