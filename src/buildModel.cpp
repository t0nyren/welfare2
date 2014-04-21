#include <opencv2/highgui/highgui.hpp>
#include "recognizer.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
using namespace std;

#define imgpath "/mnt/C/tonyren/datasets/pubfig_complete"
#define facepath "/mnt/B/datasets/experiment/face"
#define modelpath "/mnt/B/datasets/experiment/model"
#define csvpath "/mnt/B/datasets/experiment/people.csv"

int main(int argc, char** argv){
	struct timeval begin, end;

	Recognizer rec(5);
	rec.buildModel(imgpath, csvpath, facepath, modelpath);
	return 0;
}