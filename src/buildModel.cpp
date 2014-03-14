#include <opencv2/highgui/highgui.hpp>
#include "recognizer.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
using namespace std;

#define imgpath "/mnt/B/datasets/av/images"
#define facepath "/mnt/B/datasets/av/face"
#define datpath "/mnt/B/datasets/av/model.dat"
#define csvpath "/mnt/B/datasets/av/people.csv"

int main(int argc, char** argv){
	struct timeval begin, end;
	
	Recognizer rec;
	rec.buildModel(imgpath, datpath, csvpath, facepath);

	return 0;
}
