#include "encoder.h"
#include "detector.h"

class Recognizer{
	public:
		Recognizer(int patchSize = 48, int cellSize = 8, int binSize = 58, int level = 3, int width = 144, int height = 192, int numLandmarks = 5);
		void buildModel(const char* dirname, const char* csvname, const char* facedir, const char* modeldir, int startingID = 1);
		void loadModel(const char* filename);
		int writeModelToSHM();
		int loadModelFromSHM();
		void cleanSHM();
		
		Mat getFace(const char* filename, Mat& landmarks);
		int classify(const char* filename, int numReturns, int* ids, int* sims, string* imgpaths);
		int classify(const Mat& img, const Mat& landmarks, int numReturns, int* ids, int* sims, string* imgpaths);
		double evaluate();
		int getWidth();
		int getHeight();
		int getPatchSize();
		int getNumLandmarks();
		
	private:
		Detector* detector;
		Encoder* encoder;
		int level;
		int width;
		int height;
		int numLandmarks;
		int descriptorSize;
		vector<cvflann::Index<cvflann::L2<float> >* > index;
		vector<string> imgs;
		vector<int> ids;
		vector<int> ppls;
		vector<vector<float> > features;
		float* f_map;
};