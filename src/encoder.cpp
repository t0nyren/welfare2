#include "encoder.h"
#include <lbp.h>
#include <dsift.h>
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

vector<vector<float> > Encoder::extractMultiDSIFT(Mat normMat, Mat landmarks, int level){
	vector<vector<float> > ret;
	//hard code max LBP size
	int tunedCellSize = 10;
	int tunedCols, tunedRows;
	int dimension = patchSize / cellSize;
	vector<float> dsiftCode;
	for (int l = 0; l < level; l++){
		int tmpcellSize = cellSize - l;
		int tmppatchSize = tmpcellSize*dimension;
		for (unsigned int i = 0; i < landmarks.cols; i++){
			if (landmarks.at<float>(0, i) > tmppatchSize/2 && landmarks.at<float>(1, i) > tmppatchSize/2 && landmarks.at<float>(0, i) + tmppatchSize/2 < normMat.cols && landmarks.at<float>(1, i) + tmppatchSize/2 < normMat.rows){
				Mat roi(normMat, Rect(landmarks.at<float>(0, i) - tmppatchSize/2 , landmarks.at<float>(1, i) - tmppatchSize/2, tmppatchSize, tmppatchSize));
				vector<float> data;
				for (int j = 0; j < roi.cols; j++){
					for (int k = 0; k < roi.rows; k++){
						data.push_back((float)roi.at<unsigned char>(k, j)/255);
					}
				}
				//dsift
				int numFrames ;
				int descrSize ;
				VlDsiftKeypoint const *frames ;
				float const *descrs ;
				VlDsiftFilter *dsift ;
				VlDsiftDescriptorGeometry geom ;
				geom.numBinX = 2 ;
				geom.numBinY = 2 ;
				geom.numBinT = 4 ;
				geom.binSizeX = 4 ;
				geom.binSizeY = 4 ;
				dsift = vl_dsift_new (roi.rows, roi.cols) ;
				vl_dsift_set_geometry(dsift, &geom) ;
				vl_dsift_set_steps(dsift, 2, 2) ;
				vl_dsift_set_flat_window(dsift, 1) ;
				numFrames = vl_dsift_get_keypoint_num (dsift) ;
				descrSize = vl_dsift_get_descriptor_size (dsift) ;
				geom = *vl_dsift_get_geometry (dsift) ;
				vl_dsift_process (dsift, &data[0]) ;
				frames = vl_dsift_get_keypoints (dsift) ;
				descrs = vl_dsift_get_descriptors (dsift) ;	
				//cout<<"frames: "<<numFrames<<" descrs: "<<descrSize<<" cols: "<<roi.cols<<" rows: "<<roi.rows<<endl;
				float tranDescr[128];
				for (int f = 0; f < numFrames; f++){
					vl_dsift_transpose_descriptor (tranDescr, descrs + descrSize * f, geom.numBinT, geom.numBinX, geom.numBinY) ;
					for (int d = 0 ; d < descrSize ; d++) {
						tranDescr[d] = VL_MIN(512.0F * tranDescr[d], 255.0F) ;
						dsiftCode.push_back(tranDescr[d]);
						//cout<<tmpDescr[i]<<" ";
					}
				}
				//if (i != 0 && i != 2){
					ret.push_back(dsiftCode);
					dsiftCode.clear();			
				//}
				 vl_dsift_delete (dsift) ;
			}
			else{
				cout<<"Patch out of bound: "<<landmarks.at<float>(0, i)<<" "<<landmarks.at<float>(1, i)<<endl;
				cout<<"landmark: "<<i<<" Cols: "<<tunedCols<<" Rows: "<<tunedRows<<endl;
				imwrite("tmp/outOfBound.jpg", normMat);
				exit(1);
			}
		}
	}
	return ret;	
}

vector<vector<float> > Encoder::extractTunedLBP(Mat normMat, Mat landmarks){
	//40x20x2, 100x40, 40x50, 60x40
	VlLbp * lbp = vl_lbp_new (VlLbpUniform, VL_TRUE) ;
	int binSize = vl_lbp_get_dimension(lbp) ;
	vector<vector<float> > ret;
	//hard code max LBP size
	float* code = new float[50 * 50 * binSize];
	int tunedCellSize = 5;
	int tunedCols, tunedRows;

	vector<float> lbpCode;
	for (int l = 0; l <= 2; l++){
		for (int i = 0; i < landmarks.cols; i++){
			switch (i){
			case 0:
			case 1:
			case 2:
			case 3:
				tunedCols = 30 + 10*l;
				tunedRows = 10 + 10*l;
				break;			
			case 4:
				tunedCols = 70 + 20*l;
				tunedRows = 30 + 10*l;	
				break;
			case 5:
				tunedCols = 30 + 10*l;
				tunedRows = 40 + 20*l;
				break;
			case 6:
				tunedCols = 40 + 20*l;
				tunedRows = 30 + 10*l;
				break;
			case 7:
			case 8:
				tunedCols = 30 + 10*l;
				tunedRows = 40 + 20*l;
				break;
			case 9:
				tunedCols = 40 + 10*l;
				tunedRows = 20;		
				break;
			case 10:
				tunedCols = 60 + 20*l;
				tunedRows = 30;		
				break;	
			case 11:
				tunedCols = 40 + 10*l;
				tunedRows = 20 + 10*l;		
				break;			
			/*
			case 0:
			case 1:
			case 2:
			case 3:
				tunedCols = 40 - 10*l;
				tunedRows = 20 - 10*l;
				break;			
			case 4:
				tunedCols = 100 - 20*l;
				tunedRows = 40 - 10*l;	
				break;
			case 5:
				tunedCols = 40 - 10*l;
				tunedRows = 60 - 20*l;
				break;
			case 6:
				tunedCols = 60 - 20*l;
				tunedRows = 40 - 10*l;
				break;
			case 7:
			case 8:
				tunedCols = 40 - 10*l;
				tunedRows = 60 - 20*l;
				break;
			case 9:
				tunedCols = 60 - 20*l;
				tunedRows = 20;		
				break;
			*/
			default:
				cout<<"Wrong landmark size: "<<i<<endl;
				exit(1);
			}
			if (landmarks.at<float>(0, i) > tunedCols/2 && landmarks.at<float>(1, i) > tunedRows/2 && landmarks.at<float>(0, i) + tunedCols/2 < normMat.cols && landmarks.at<float>(1, i) + tunedRows/2 < normMat.rows){
				Mat roi(normMat, Rect(landmarks.at<float>(0, i) - tunedCols/2 , landmarks.at<float>(1, i) - tunedRows/2, tunedCols, tunedRows));
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
				for (int j = 0; j < (tunedCols/tunedCellSize) * (tunedRows/tunedCellSize) * binSize; j++){
					code[j] = 0;
				}
				vl_lbp_process(lbp, code, &data[0], tunedCols, tunedRows, tunedCellSize);
				
				for (int j = 0; j < (tunedCols/tunedCellSize) * (tunedRows/tunedCellSize) * binSize; j++){
					//cout<<code[j]<<" ";
					lbpCode.push_back(code[j]);
				}
				//if (i != 0 && i != 2){
					ret.push_back(lbpCode);
					lbpCode.clear();			
				//}
				//cout<<"feature "<<i/2<<" size: "<<ret.size()<<endl;
			}
			else{
				cout<<"Patch out of bound: "<<landmarks.at<float>(0, i)<<" "<<landmarks.at<float>(1, i)<<endl;
				cout<<"Cols: "<<tunedCols<<" Rows: "<<tunedRows<<endl;
				exit(1);
			}
		}
	}
	delete[] code;
	vl_lbp_delete(lbp);
	return ret;	
}

vector<vector<float> > Encoder::extractTunedDSIFT(Mat normMat, Mat landmarks){
	//eye, eyebrow, eye-band,  nose-ver, mouth, left face, right face, jaw, forehead, nose-hor
	//40x20x2, 40x20x2, 100x40, 40x60, 60x40, 40x60,40x60, 50x20, 100x40, 60x40
	vector<vector<float> > ret;
	//hard code max LBP size
	int tunedCellSize = 10;
	int tunedCols, tunedRows;

	vector<float> dsiftCode;
	for (int l = 0; l <= 2; l++){
		for (int i = 0; i < landmarks.cols; i++){
			switch (i){
			case 0:
			case 1:
			case 2:
			case 3:
				tunedCols = 30 + 10*l;
				tunedRows = 10 + 10*l;
				break;			
			case 4:
				tunedCols = 70 + 20*l;
				tunedRows = 30 + 10*l;	
				break;
			case 5:
				tunedCols = 30 + 10*l;
				tunedRows = 40 + 20*l;
				break;
			case 6:
				tunedCols = 40 + 20*l;
				tunedRows = 30 + 10*l;
				break;
			case 7:
			case 8:
				tunedCols = 30 + 10*l;
				tunedRows = 40 + 20*l;
				break;
			case 9:
				tunedCols = 40 + 10*l;
				tunedRows = 20;		
				break;
			case 10:
				tunedCols = 60 + 20*l;
				tunedRows = 30;		
				break;	
			case 11:
				tunedCols = 40 + 10*l;
				tunedRows = 20 + 10*l;		
				break;					
			default:
				cout<<"Wrong landmark size: "<<i<<endl;
				exit(1);
			}
			if (landmarks.at<float>(0, i) > tunedCols/2 && landmarks.at<float>(1, i) > tunedRows/2 && landmarks.at<float>(0, i) + tunedCols/2 < normMat.cols && landmarks.at<float>(1, i) + tunedRows/2 < normMat.rows){
				Mat roi(normMat, Rect(landmarks.at<float>(0, i) - tunedCols/2 , landmarks.at<float>(1, i) - tunedRows/2, tunedCols, tunedRows));
				vector<float> data;
				for (int j = 0; j < roi.cols; j++){
					for (int k = 0; k < roi.rows; k++){
						data.push_back((float)roi.at<unsigned char>(k, j)/255);
					}
				}

				//dsift
				int numFrames ;
				int descrSize ;
				VlDsiftKeypoint const *frames ;
				float const *descrs ;
				VlDsiftFilter *dsift ;
				VlDsiftDescriptorGeometry geom ;
				geom.numBinX = 2 ;
				geom.numBinY = 2 ;
				geom.numBinT = 4 ;
				geom.binSizeX = 4 ;
				geom.binSizeY = 4 ;
				dsift = vl_dsift_new (roi.rows, roi.cols) ;
				vl_dsift_set_geometry(dsift, &geom) ;
				vl_dsift_set_steps(dsift, 2, 2) ;
				vl_dsift_set_flat_window(dsift, 1) ;
				numFrames = vl_dsift_get_keypoint_num (dsift) ;
				descrSize = vl_dsift_get_descriptor_size (dsift) ;
				geom = *vl_dsift_get_geometry (dsift) ;
				vl_dsift_process (dsift, &data[0]) ;
				frames = vl_dsift_get_keypoints (dsift) ;
				descrs = vl_dsift_get_descriptors (dsift) ;	
				//cout<<"frames: "<<numFrames<<" descrs: "<<descrSize<<" cols: "<<roi.cols<<" rows: "<<roi.rows<<endl;
				float tranDescr[128];
				for (int f = 0; f < numFrames; f++){
					vl_dsift_transpose_descriptor (tranDescr, descrs + descrSize * f, geom.numBinT, geom.numBinX, geom.numBinY) ;
					for (int d = 0 ; d < descrSize ; d++) {
						tranDescr[d] = VL_MIN(512.0F * tranDescr[d], 255.0F) ;
						dsiftCode.push_back(tranDescr[d]);
						//cout<<tmpDescr[i]<<" ";
					}
				}
				//if (i != 0 && i != 2){
					ret.push_back(dsiftCode);
					dsiftCode.clear();			
				//}
				 vl_dsift_delete (dsift) ;
			}
			else{
				cout<<"Patch out of bound: "<<landmarks.at<float>(0, i)<<" "<<landmarks.at<float>(1, i)<<endl;
				cout<<"landmark: "<<i<<" Cols: "<<tunedCols<<" Rows: "<<tunedRows<<endl;
				imwrite("tmp/outOfBound.jpg", normMat);
				exit(1);
			}
		}
	}
	return ret;	
}

int Encoder::getPatchSize(){
	return patchSize;
}