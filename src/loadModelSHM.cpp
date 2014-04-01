#include <opencv2/highgui/highgui.hpp>
#include "recognizer.h"
#include "detector.h"
#include <vector>
#include <ctime>
#include <sys/time.h>
#include <pthread.h>
using namespace std;

//#define datpath "/mnt/B/datasets/production/model.dat"
#define NUM_THREADS 5
Recognizer rec;

void *classify(void *threadid)
{
	Detector detector;
	long tid;
	tid = (long)threadid;
	char img[100];
	char buf[1024];
	sprintf(img, "data/%d.jpg", tid);
	int* ids = new int[3];
	int* sims = new int[3];
	string* paths = new string[3];
	for (int j = 0; j < 10; j++){
		Mat landmarks;
		Mat face = detector.detectNorm(img, rec.getWidth(), rec.getHeight(), rec.getPatchSize(), landmarks, rec.getNumLandmarks(), false);
		int num = rec.classify(face, landmarks, 3, ids, sims, paths);
		//int num = rec.classify(img, 3, ids, sims, paths);
		for (int i = 0; i < num; i++){
			sprintf(buf, "thread %d, prediction: %d %d, similarity: %d\n", tid, i, ids[i], sims[i]);
		}
		cout<<buf;
	}
   pthread_exit(NULL);
}

int main(int argc, char** argv){
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	pthread_t threads[NUM_THREADS];

	  
	//rec.cleanSHM();
	//rec.loadModel(datpath);
	rec.loadModelFromSHM();
	gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//rec.evaluate();
	cout<<"Model loaded in "<<elapsed<<" seconds"<<endl;		
	int* ids = new int[3];
	int* sims = new int[3];
	string* paths = new string[3];
	int count = 0;
	
	/*while(true){
		cout<<"please input test img"<<endl;
		string filename, simpath;
		cin>>filename;
		//filename = "data/test.jpg";
		int num = rec.classify(filename.data(), 3, ids, sims, paths);
		cout<<count<<" "<<num<<endl;
		count++;
		for (int i = 0; i < num; i++){
			cout<<"Prediction "<<i<<" "<<ids[i]<<" similarity: "<<sims[i]<<endl;;
			cout<<"Most similar image: "<<paths[i]<<endl;
		}
	}*/
	for(int i=0; i < NUM_THREADS; i++ ){
		cout << "main() : creating thread, " << i << endl;
		int rc = pthread_create(&threads[i], NULL, classify, (void *)i);
		if (rc){
		 cout << "Error:unable to create thread," << rc << endl;
		 exit(-1);
		}
	}
	pthread_exit(NULL); 
	return 0;
}


