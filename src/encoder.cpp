#include "encoder.h"
#include <lbp.h>
#include <iostream>


Encoder::Encoder(int patchSize, int cellSize, int descrSize){
	this->cellSize = cellSize;
	this->patchSize = patchSize;
	this->descrSize = descrSize;
}

vector<vector<float> > Encoder::extractMultiLBP(Mat img, Mat landmarks, int level){
	//Mat img = imread(img_path, CV_LOAD_IMAGE_COLOR);
	VlLbp * lbp = vl_lbp_new (VlLbpUniform, VL_TRUE) ;
	int dimensionx = patchSize / cellSize;
	int dimensiony = patchSize / cellSize;
	int dimensionc = vl_lbp_get_dimension(lbp) ;
	vector<vector<float> > ret;
	float* code = new float[dimensionx*dimensiony*dimensionc];
	//cout<<"dim: "<<dimensionx<<" "<<dimensiony<<" "<<dimensionc <<endl;
	for (int l = 0; l < level; l++){
		int tmpcellSize = cellSize - l;
		int tmppatchSize = tmpcellSize*dimensionx;
		
		for (unsigned int i = 0; i < landmarks.cols; i++){
			if (landmarks.at<float>(0, i) > patchSize/2 && landmarks.at<float>(1, i) > patchSize/2 && landmarks.at<float>(0, i) + patchSize/2 < img.cols && landmarks.at<float>(1, i) + patchSize/2 < img.rows){
				Mat roi(img, Rect(landmarks.at<float>(0, i) - tmppatchSize/2 , landmarks.at<float>(1, i) - tmppatchSize/2, tmppatchSize, tmppatchSize));
				vector<float> data;
				if (lbp == NULL) {
				  cout<<"fail to init LBP detector"<<endl;
				  return ret;
				}
				for (int j = 0; j < roi.cols; j++){
					for (int k = 0; k < roi.rows; k++){
						data.push_back((float)roi.at<unsigned char>(k, j)/255);
					}
				}

				//float* features = new float[dimensionx * dimensiony * dimensionc];
				//cout<<"code size: x: "<<dimensionx<<" y: "<<dimensiony<<" c: "<<dimensionc<<endl;
				
				for (int j = 0; j < dimensionx*dimensiony*dimensionc; j++){
					code[j] = 0;
				}
				vl_lbp_process(lbp, code, &data[0], tmppatchSize, tmppatchSize, tmpcellSize);
				vector<float> lbpCode;
				for (int j = 0; j < dimensionx*dimensiony*dimensionc; j++){
					//cout<<code[j]<<" ";
					lbpCode.push_back(code[j]);
				}
				ret.push_back(lbpCode);
				//cout<<"feature "<<i/2<<" size: "<<ret.size()<<endl;
			}
			else{
				cout<<"Patch out of bound: "<<landmarks.at<float>(0, i)<<" "<<landmarks.at<float>(1, i)<<endl;
				exit(1);
			}
		}
	}
	delete[] code;
	vl_lbp_delete(lbp);
	return ret;	
}

int Encoder::getPatchSize(){
	return patchSize;
}