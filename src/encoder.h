#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;

class Encoder{
	public:
		Encoder(int patchSize = 48, int cellSize = 8, int descrSize = 58);
		vector<vector<float> > extractMultiLBP(Mat normMat, Mat landmarks, int level = 1);
		int getPatchSize();

	private:
		int cellSize;
		int patchSize;
		int descrSize;
};