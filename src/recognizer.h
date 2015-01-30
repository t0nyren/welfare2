#include "encoder.h"
#include "detector.h"
#include <opencv2/flann/flann.hpp>

class Recognizer{
	public:
		Recognizer(int numLandmarks = 5, int patchSize = 60, int cellSize = 8, int binSize = 58, int level = 4, int width = 144, int height = 192);
		void buildModel(const char* dirname, const char* csvname, const char* facedir, const char* modeldir, int startingID = 1);
		void loadModel(const char* filename);
		int writeModelToSHM();
		int loadModelFromSHM();
		void cleanSHM();
		
		Mat getFace(const char* filename, Mat& landmarks);
		void getFaces(const Mat& frame,  int maxNumFaces, int& numFaceRet, Mat* faces, Mat* landmarks_origin, Mat* landmarks);
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
		vector<int> descriptorSize;
		vector<cvflann::Index<cvflann::L2<float> >* > index;
		vector<string> imgs;
		vector<int> ids;
		vector<int> ppls;
		vector<vector<float> > features;
		float* f_map;
};
