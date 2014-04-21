#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

class Encoder{
	public:
		Encoder(int patchSize = 48, int cellSize = 8, int descrSize = 58);
		vector<vector<float> > extractMultiLBP(Mat normMat, Mat landmarks, int level = 1);
		vector<vector<float> > extractMultiDSIFT(Mat normMat, Mat landmarks, int level = 1);
		vector<vector<float> > extractTunedLBP(Mat normMat, Mat landmarks);
		vector<vector<float> > extractTunedDSIFT(Mat normMat, Mat landmarks);
		int getPatchSize();

	private:
		int cellSize;
		int patchSize;
		int descrSize;
};