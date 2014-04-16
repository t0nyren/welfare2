#include <opencv2/highgui/highgui.hpp>
#include "recognizer.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
using namespace std;

#define imgpath "/mnt/B/datasets/qvoduser/images"
#define facepath "/mnt/B/datasets/qvoduser/face"
#define modelpath "/mnt/B/datasets/qvoduser/model"
#define csvpath "/mnt/B/datasets/qvoduser/people.csv"

int main(int argc, char** argv){
	struct timeval begin, end;

	Recognizer rec;
	rec.buildModel(imgpath, csvpath, facepath, modelpath);

	return 0;
}