#include "recognizer.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <dirent.h>
#include<sys/stat.h>
#include<sys/types.h>
#include <ctime>
#include <sys/time.h>
#include <string>
#include <stdio.h>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>
using namespace std;

Recognizer::Recognizer(int patchSize, int cellSize, int binSize, int level, int width, int height, int numLandmarks){
	this->detector = new Detector();
	this->encoder = new Encoder(patchSize, cellSize, binSize);
	this->width = width;
	this->height = height;
	this->level = level;
	this->numLandmarks = numLandmarks;
	descriptorSize = pow(patchSize/cellSize, 2) * binSize;
	for (int i = 0; i < level; i++){
		for (int j = 0; j < numLandmarks; j++){
			vector<float> empty;
			features.push_back(empty);
			cvflann::Index<cvflann::L2<float> >* ind = NULL;
			index.push_back(ind);
		}
	}
}

void Recognizer::buildModel(const char* dirname, const char* csvname, const char* facedir, const char* modeldir, int startingID){
	if (facedir != NULL){
		mkdir(facedir, S_IRWXU);
	}
	if (modeldir != NULL){
		mkdir(modeldir, S_IRWXU);
	}	

	
	ofstream csvout;
	csvout.open(csvname);
	if (csvout.fail()){
		cout<<"fail to open csvname"<<endl;
		exit(1);
	}

	DIR *pDIR;
	struct dirent *entry;
	pDIR = opendir(dirname);
	if( pDIR == NULL){
		std::cout<<"Cannot open image directory"<<std::endl;
		exit(1);
	}
	
	//entry = readdir(pDIR);
	vector<string> imgdirs;
	vector<int> dirids;
	int id = startingID;
	while((entry = readdir(pDIR)) != NULL){
		if(0 != strcmp( ".", entry->d_name) && 0 != strcmp( "..", entry->d_name) ){
			imgdirs.push_back(string(entry->d_name));
			dirids.push_back(id);
			cout<<entry->d_name<<endl;
			id++;
		}
	}
	closedir(pDIR);
	vector<int> imgcounts(imgdirs.size());

    
	omp_set_num_threads(30);
	#pragma omp parallel for
	for (int i = 0; i < imgdirs.size(); i++){
		Detector dd;
		std::string s1 = facedir;
		std::string s2 = imgdirs[i];
		imgcounts[i] = 0;
		char buf[100];
		sprintf(buf, "%s/%d.dat", modeldir, dirids[i]);
		ofstream fout;
		fout.open(buf);
		if (fout.fail()){
			cout<<"fail to open filename"<<endl;
			exit(1);
		}
		sprintf(buf, "%d", dirids[i]);
		std::string dir_path = s1 + '/' + buf;
		std::cout<<dir_path<<std::endl;
		mkdir(dir_path.data(), S_IRWXU);
		DIR* pDIR2;
		std::string s3 = dirname;
		std::string origin_path = s3 + '/' + s2;
		pDIR2 = opendir(origin_path.data());
		struct dirent *entry2;
		entry2 = readdir(pDIR2);
		while(entry2 != NULL){
			if(0 == strcmp( ".", entry2->d_name) || 0 == strcmp( "..", entry2->d_name) ){
				entry2 = readdir(pDIR2);
				continue;
			}
			//std::cout<<"\t"<<entry2->d_name<<std::endl;
			std::string s4 = entry2->d_name;
			std::string img_path = dir_path + '/' + s4;
			std::string img_origin_path = origin_path + '/' + s4;
			cout<<img_path<<endl;
			Mat landmarks;
			Mat face = dd.detectNorm(img_origin_path, width, height, encoder->getPatchSize(), landmarks, numLandmarks, true, false);
			
			if (!face.empty()){
				Mat gray;
				if (face.channels()==3 || face.channels()==4){
					cvtColor(face, gray, CV_RGB2GRAY);
				}
				else if (face.channels() != 1){
					cout<<"cvtcolor failed"<<endl;
					continue;
				}
				else{
					gray = face;
				}
				vector<vector<float> > code = encoder->extractMultiLBP(gray, landmarks, level);
				imwrite( img_path, face );
				fout<<dirids[i]<<" "<<img_path<<endl;
				for (int i = 0; i < code.size(); i++){
					for (int j = 0; j < code[i].size(); j++){
						fout<<code[i][j]<<" ";
					}
					fout<<endl;
				}
				imgcounts[i]++;
			}
			entry2 = readdir(pDIR2);
		}
		fout.close();
		closedir(pDIR2);      
	}
	for (int i = 0; i < imgdirs.size(); i++){
		csvout<<id<<','<<imgdirs[i]<<','<<imgcounts[i]<<endl;
	}
}

void Recognizer::loadModel(const char* modeldir){
	DIR *pDIR;
	struct dirent *entry;
	pDIR = opendir(modeldir);
	if( pDIR == NULL){
		std::cout<<"Cannot open image directory"<<std::endl;
		exit(1);
	}
	
	//entry = readdir(pDIR);
	vector<string> modelfiles;
	while((entry = readdir(pDIR)) != NULL){
		if(0 != strcmp( ".", entry->d_name) && 0 != strcmp( "..", entry->d_name) ){
			modelfiles.push_back(string(modeldir) + '/' + string(entry->d_name));
		}
	}
	for (int i = 0; i < modelfiles.size(); i++){
		ifstream fin;
		fin.open(modelfiles[i].data());
		if (fin.fail()){
			cout<<"Cannot open model file"<<endl;
			exit(1);
		}
		char line[5000];
		fin.getline(line,5000);
		cout<<"level: "<<level<<" landmarks: "<<numLandmarks<<endl;
		int preid = -1;
		while(!fin.eof()){
			char *tok;
			tok = strtok(line, " ");
			int id = atoi(tok);
			tok = strtok(NULL, " ");
			
			string imgname(tok);
			cout<<id<<" "<<imgname<<endl;
			ids.push_back(id);
			imgs.push_back(imgname);
			if (id != preid){
				ppls.push_back(id);
				preid = id;
			}
			//get descriptors
			for (int i = 0; i < level; i++){
				for (int j = 0; j < numLandmarks; j++){
					vector<float> descr;
					for (int k = 0; k < descriptorSize; k++){
						float val;
						fin>>val;
						features[i*numLandmarks + j].push_back(val);
					}
					//features.push_back(descr);
				}
			}
			fin.getline(line,5000);
			fin.getline(line,5000);
		}
		fin.close();
	}
	cout<<"Features loaded, Num trees: "<<features.size()<<" num imgs: "<<features[0].size()/descriptorSize<<endl;
	cout<<"ids: "<<ids.size()<<" num people: "<<ids[ids.size()-1]<<endl;
	
	#pragma omp parallel for
	for (int i = 0; i < features.size(); i++){
		cout<<"Building index for tree "<<i<<endl;
		cvflann::Matrix<float> dataset(features[i].data(), ids.size(), descriptorSize);
		index[i] = new cvflann::Index<cvflann::L2<float> >(dataset, cvflann::KDTreeIndexParams(10));
		index[i]->buildIndex();
	}
}
Mat Recognizer::getFace(const char* filename, Mat& landmarks){
	return detector->detectNorm(filename, width, height, encoder->getPatchSize(), landmarks, numLandmarks, true, false);
}
//return -1 for not found
int Recognizer::classify(const char* filename, int numReturns, int* maxId, int* sims, string* maxImg){
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	Mat landmarks;
	Mat face = detector->detectNorm(filename, width, height, encoder->getPatchSize(), landmarks, numLandmarks, true, false);
	return this->classify(face, landmarks, numReturns, maxId, sims, maxImg);
}

int Recognizer::classify(const Mat& face, const Mat& landmarks, int numReturns, int* maxId, int* sims, string* maxImg){
	srand (time(NULL));
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	vector<vector<float> > code;
	if (!face.empty()){
		Mat gray;
		if (face.channels()==3 || face.channels()==4){
			cvtColor(face, gray, CV_RGB2GRAY);
		}
		else if (face.channels() != 1){
			cout<<"cvtcolor failed"<<endl;
			return -1;
		}
		else{
			gray = face;
		}
		code = encoder->extractMultiLBP(gray, landmarks, level);
	}
	else
		return -1;
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Face encoded in "<<elapsed<<" seconds"<<endl;
	gettimeofday(&begin, NULL);
	int nn = 1;
	int* indices_array = new int[nn];
	float* dists_array = new float[nn];
	double correct = 0;
	vector<int> counts;
	vector<int> imgmatch;
	vector<int> imgcounts;
	for (int i = 0; i < ids[ids.size()-1] + 1; i++){
		counts.push_back(0);
	}
	//for (int i = 0; i < imgs.size(); i++){
	//	imgcounts.push_back(0);
	//}
	
	for (int j = 0; j < features.size(); j++){
		cvflann::Matrix<float> query(code[j].data(), 1, descriptorSize);
		cvflann::Matrix<int> indices(indices_array, 1, nn);
		cvflann::Matrix<float> dists(dists_array, 1, nn);
		char buf[100];
		index[j]->knnSearch(query, indices, dists, nn, cvflann::SearchParams(128));
		counts[ids[indices[0][0]]]++;
		bool added = false;
		for (int k = 0; k < imgmatch.size(); k++){
			if ( indices[0][0] == imgmatch[k]){
				imgcounts[k]++;
				added = true;
				break;
			}
		}
		if (!added){
			imgmatch.push_back(indices[0][0]);
			imgcounts.push_back(1);
		}
	}
	
	vector<int> maxCount;
	for (int i = 0; i < numReturns; i++){
		maxCount.push_back(0);
	}
	
	//max ids
	int numRet = 0;
	for (int i = 0; i < numReturns; i++){
		//default ret
		maxId[i] = i + 1;
		sims[i] = rand()%10;
		
		//default img
		for (int j = 0; j < ids.size(); j++){
			if (ids[j] == maxId[i]){
				maxImg[i] = imgs[j];
				break;
			}
		}
	}
	
	for (int i = 0; i < numReturns; i++){
		bool found = false;
		for (int j = 0; j < counts.size(); j++){
			if (counts[j] > maxCount[i]){
				bool add = 1;
				for (int k = 0; k < i; k++){
					if (maxId[k] == j){
						add = 0;
						break;
					}
				}
				if (add){
					maxCount[i] = counts[j];
					maxId[i] = j;
					//sims[i] = min(maxCount[i]*10, 100) - rand()%10;
					sims[i] = maxCount[i] * 10 * 10 / 15;
					if (sims[i] > 5 && sims[i] < 95){
						 sims[i] += (5 - rand()%10);
					}
					found = true;
				}
			}
		}
		if (!found){
			numRet = i;
			break;
		}
		else
			numRet++;
	}
	
	//imgs
	for (int i = 0; i < numRet; i++){
		int maxImgCount = 0;
		for (int j = 0; j < imgmatch.size(); j++){
			if (ids[imgmatch[j]] == maxId[i] && imgcounts[j] > maxImgCount){
				maxImgCount = imgcounts[j];
				maxImg[i] = imgs[imgmatch[j]];
			}
		}
	//	cout<<maxImgCount<<" ";
	}
	//cout<<endl;
	
	//max imgs
	/*
	int numRet = 0;
	for (int i = 0; i < numReturns; i++){
		bool found = false;
		for (int j = 0; j < imgcounts.size(); j++){
			if (imgcounts[j] > maxCount[i]){
				bool add = 1;
				for (int k = 0; k < i; k++){
					if (maxId[k] == ids[imgmatch[j]]){
						add = 0;
						break;
					}
				}
				if (add){
					found = true;
					maxCount[i] = imgcounts[j];
					maxId[i] = ids[imgmatch[j]];
					maxImg[i] = imgs[imgmatch[j]];
				}
			}
		}
		if (!found){
			numRet = i;
			break;
		}
		else
			numRet++;
	}*/
	
	
	gettimeofday(&end, NULL);
	elapsed = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
	//cout<<"Face searched in "<<elapsed<<" seconds"<<endl;
	delete [] indices_array;
	delete [] dists_array;
	return numRet;
}

int Recognizer::writeModelToSHM(){
	int shm_id, shm_id2, shm_id3;
	key_t key, key2, key3;
	float* f_map;
	int* id_map;
	char* img_map;
	char pathname[30] ;
	strcpy(pathname,"/tmp") ;
	
	//3,4,5: production
	//6,7,8: baidu180
	key = ftok(pathname,0x03);
	key2 = ftok(pathname, 0x04);
	key3 = ftok(pathname, 0x05);
	char a;
	if(key==-1 || key2 == -1 || key3 == -1)
	{
		perror("ftok error");
		cin>>a;
		return -1;
    }
	printf("key=%d\n",key) ;
	unsigned long size = 1024*1024*1024;
	size *= 30;
	unsigned long size2 = 1024*1024;
	unsigned long size3 = 1024*1024*50;
	
	shm_id = shmget(key,size,IPC_CREAT|IPC_EXCL|0600); 
	shm_id2 = shmget(key2,size2,IPC_CREAT|IPC_EXCL|0600); 
	shm_id3 = shmget(key3,size3,IPC_CREAT|IPC_EXCL|0600); 
	
	if(shm_id == -1 || shm_id2 == -1 || shm_id3 == -1){
		perror("shmget error");
		cin>>a;
		return -1;
    }
	printf("shm_id=%d\n", shm_id) ;
	printf("shm_id2=%d\n", shm_id2) ;
	printf("shm_id3=%d\n", shm_id3) ;
    
	f_map=(float*)shmat(shm_id,NULL,0);
	id_map = (int*)shmat(shm_id2,NULL,0);
	img_map = (char*)shmat(shm_id3,NULL,0);

	for (int i = 0; i < features.size(); i++){
		cout<<"Writing features to shm for landmark "<<i<<endl;
		for (int j = 0; j < features[i].size(); j++){
			*(f_map + (i*features[0].size() + j) ) = features[i][j];
		}
	}

	*(id_map) = ids.size();
	cout<<"Writing ids to shm"<<endl;
	for (int i = 0; i < ids.size(); i++){
		*(id_map + i + 1 ) = ids[i];
	}

	cout<<"Writing img path to shm "<<endl;
	char* buf = new char[256 * imgs.size()];
	int count = 0;
	for (int i = 0; i < imgs.size(); i++){
		const char* str = imgs[i].data();
		for (int j = 0; j < strlen(str); j++){
			buf[count] = str[j];
			count++;
		}
		//buf[i*256 + j] = '\0';
		buf[count] = '\t';
		count++;
	}
	buf[count-1] = '\0';
	//cout<<buf<<endl;
	for (int i = 0; i < imgs.size(); i++){
		strncpy(img_map,buf,256 * imgs.size());
	}	
	shmdt(f_map) ;
	shmdt(id_map) ;
	shmdt(img_map) ;
	delete [] buf;
	return 1;
}


int Recognizer::loadModelFromSHM(){
	int shm_id, shm_id2, shm_id3;
	key_t key, key2, key3;
	int* id_map;
	char* img_map;
	char pathname[30] ;
	strcpy(pathname,"/tmp") ;
	key = ftok(pathname,0x03);
	key2 = ftok(pathname, 0x04);
	key3 = ftok(pathname, 0x05);
	char a;
	if(key==-1 || key2 == -1 || key3 == -1){
		perror("ftok error");
		cin>>a;
		return -1;
    }
	printf("key=%d\n", key) ;

    shm_id = shmget(key,0, 0);   
	shm_id2 = shmget(key2,0, 0);
	shm_id3 = shmget(key3,0, 0);
    if(shm_id == -1 || shm_id == -2 || shm_id == -3){
		perror("shmget error");
		return -1;
    }
	printf("shm_id=%d\n", shm_id) ;

    f_map = (float*)shmat(shm_id,NULL,0);
	id_map = (int*)shmat(shm_id2,NULL,0);
	img_map = (char*) shmat(shm_id3,NULL,0);
	
	cout<<"loading ids from shm"<<endl;
	int numImgs = *id_map;
	for (int i = 0; i < numImgs; i++){
		ids.push_back(*(id_map + i + 1));
	}
	cout<<"loading imgs from shm"<<endl;
	char* buf = new char[256 * numImgs];
	strncpy(buf,img_map,256 * numImgs);
	char *tok;
	char *saveptr;
	tok = strtok_r(buf, "\t", &saveptr);
	while(tok != NULL){
		//cout<<tok<<endl;
		imgs.push_back(tok);
		tok = strtok_r(NULL, "\t", &saveptr);
	}
	cout<<ids.size()<<" "<<ids[ids.size()-1]<<" "<<imgs[ids.size()-1]<<endl;
	cout<<"building kdtrees from shm"<<endl;
	
	#pragma omp parallel for
	for (int i = 0; i < level * numLandmarks; i++){
		cout<<"Building index for tree "<<i<<endl;
		cvflann::Matrix<float> dataset(f_map + i*ids.size()*descriptorSize, ids.size(), descriptorSize);
		//float* tmp = f_map + i*ids.size()*descriptorSize;
		//cout<<tmp[0]<<" "<<tmp[1]<<endl;
		index[i] = new cvflann::Index<cvflann::L2<float> >(dataset, cvflann::KDTreeIndexParams(10));
		index[i]->buildIndex();
	}
	
	/*
	cout<<"Evaluating"<<endl;
	int nn = 2;
	int* indices_array = new int[nn];
	float* dists_array = new float[nn];
	double correct = 0;
	vector<int> counts;
	for (int i = 0; i < ids[ids.size()-1] + 1; i++){
		counts.push_back(0);
	}
	
	for (int i = 0; i < ids.size(); i++){	
		vector<int> imgmatch;
		vector<int> imgcounts;
		for (int j = 0; j < level * numLandmarks; j++){
			vector<float> code;
			float* tmp = f_map + j*ids.size()*descriptorSize;
			for (int k = 0; k < descriptorSize; k++){
				code.push_back(tmp[i*descriptorSize + k]);
			}
			//cout<<"code size: "<<code.size()<<endl;
			cvflann::Matrix<float> query(code.data(), 1, descriptorSize);
			cvflann::Matrix<int> indices(indices_array, 1, nn);
			cvflann::Matrix<float> dists(dists_array, 1, nn);
			index[j]->knnSearch(query, indices, dists, nn, cvflann::SearchParams(128));
			//cout<<i<<" "<<ids.size()<<" predict: "<<ids[indices[0][1]]<<" actual: "<<ids[i]<<endl;
			counts[ids[indices[0][1]]]++;
			bool added = false;
			for (int k = 0; k < imgmatch.size(); k++){
				if ( indices[0][1] == imgmatch[k]){
					imgcounts[k]++;
					added = true;
					break;
				}
			}
			if (!added){
				imgmatch.push_back(indices[0][1]);
				imgcounts.push_back(1);
			}			
		}
		int maxCount = -1;
		int maxId = -1;
		//for (int j = 0; j < imgcounts.size(); j++){
		//	if(imgcounts[j] >= maxCount){
		//		maxCount = imgcounts[j];
		//		maxId = ids[imgmatch[j]];
		//	}
		//}
		for (int j = 0; j < counts.size(); j++){
			if (counts[j] >= maxCount){
				maxCount = counts[j];
				maxId = j;
			}
		}
		cout<<ids[i]<<" "<<maxId<<" "<<maxCount<<endl;
		if (maxId == ids[i])
			correct++;
		for (int j = 0; j < counts.size(); j++){
			counts[j] = 0;
		}
	}
	cout<<"accuracy: "<<correct/ids.size()<<endl;*/
	return 1;
}

void Recognizer::cleanSHM(){
	int shm_id, shm_id2, shm_id3;
	key_t key, key2, key3;
	char pathname[30] ;
	strcpy(pathname,"/tmp") ;
	key = ftok(pathname,0x06);
	key2 = ftok(pathname, 0x07);
	key3 = ftok(pathname, 0x08);
	shm_id = shmget(key,0, 0); 
	shm_id2 = shmget(key2,0, 0);
	shm_id3 = shmget(key3,0, 0);
	shmctl(shm_id, IPC_RMID, NULL) ;
	shmctl(shm_id2, IPC_RMID, NULL) ;
	shmctl(shm_id3, IPC_RMID, NULL) ;
}

double Recognizer::evaluate(){
	cout<<"Evaluating"<<endl;
	vector<int> success_num;
	vector<int> fail_num;
	vector<float> success_dist;
	vector<float> fail_dist;
	
	int nn = 2;
	int* indices_array = new int[nn];
	float* dists_array = new float[nn];
	double correct = 0;
	vector<int> counts;
	for (int i = 0; i < ppls.size(); i++){
		counts.push_back(0);
	}
	for (int i = 0; i < ids.size(); i++){
		//cout<<"eval id: "<<ids[i]<<endl;
		vector<int> imgmatch;
		vector<int> imgcounts;
		for (int j = 0; j < features.size(); j++){
			
			vector<float> code;
			for (int k = 0; k < descriptorSize; k++){
				code.push_back(features[j][i*descriptorSize + k]);
			}
			//cout<<"checking landmark "<<j<<" code size: "<<code.size()<<" descrSize: "<<descriptorSize<<endl;
			cvflann::Matrix<float> query(code.data(), 1, descriptorSize);
			cvflann::Matrix<int> indices(indices_array, 1, nn);
			cvflann::Matrix<float> dists(dists_array, 1, nn);
			index[j]->knnSearch(query, indices, dists, nn, cvflann::SearchParams(128));
				//cout<<i<<" "<<ids.size()<<" predict: "<<ids[indices[0][1]]<<" actual: "<<ids[i/10]<<endl;
			counts[ids[indices[0][1]]]++;
			//cout<<"landmark "<<j<<": "<<ids[indices[0][1]]<<endl;
	/*for (int i = 0; i < ids.size(); i++){	
		vector<int> imgmatch;
		vector<int> imgcounts;
		for (int j = 0; j < level * numLandmarks; j++){
			vector<float> code;
			float* tmp = f_map + j*ids.size()*descriptorSize;
			for (int k = 0; k < descriptorSize; k++){
				code.push_back(tmp[i*descriptorSize + k]);
			}
			//cout<<"code size: "<<code.size()<<endl;
			cvflann::Matrix<float> query(code.data(), 1, descriptorSize);
			cvflann::Matrix<int> indices(indices_array, 1, nn);
			cvflann::Matrix<float> dists(dists_array, 1, nn);
			index[j]->knnSearch(query, indices, dists, nn, cvflann::SearchParams(128));
			//cout<<i<<" "<<ids.size()<<" predict: "<<ids[indices[0][1]]<<" actual: "<<ids[i]<<endl;
			counts[ids[indices[0][1]]]++;	*/		
			bool added = false;
			for (int k = 0; k < imgmatch.size(); k++){
				if ( indices[0][1] == imgmatch[k]){
					imgcounts[k]++;
					added = true;
					break;
				}
			}
			if (!added){
				imgmatch.push_back(indices[0][1]);
				imgcounts.push_back(1);
			}			
		}
		int maxCount = -1;
		int maxId = -1;
		//for (int j = 0; j < imgcounts.size(); j++){
		//	if(imgcounts[j] >= maxCount){
		//		maxCount = imgcounts[j];
		//		maxId = ids[imgmatch[j]];
		//	}
		//}
		for (int j = 0; j < counts.size(); j++){
			if (counts[j] >= maxCount){
				maxCount = counts[j];
				maxId = j;
			}
		}
		cout<<ids[i]<<" "<<maxId<<" "<<maxCount<<endl;
		if (maxId == ids[i]){
			correct++;
			success_num.push_back(maxCount);
		}
		else{
			fail_num.push_back(maxCount);
		}
		for (int j = 0; j < counts.size(); j++){
			counts[j] = 0;
		}
	}
	delete[] indices_array;
	delete[] dists_array;
	cout<<"Num of queries: "<<ids.size()<<" Num of corrects: "<<correct<<endl;
	cout<<"accuracy: "<<correct/ids.size()<<endl;
	
	double avg = 0;
	double max = -1;
	double min = 20;
	for (int i = 0; i < success_num.size(); i++){
		avg += success_num[i];
		if (success_num[i] > max)
			max = success_num[i];
		if (success_num[i] < min)
			min = success_num[i];
	}
	avg /= success_num.size();
	cout<<"Correct: max: "<<max<<" min: "<<min<<" avg: "<<avg<<endl;
	avg = 0;
	max = -1;
	min = 20;
	for (int i = 0; i < fail_num.size(); i++){
		avg += fail_num[i];
		if (fail_num[i] > max)
			max = fail_num[i];
		if (fail_num[i] < min)
			min = fail_num[i];
	}
	avg /= fail_num.size();
	cout<<"Incorrect: max: "<<max<<" min: "<<min<<" avg: "<<avg<<endl;	
	return correct/ids.size();
}

int Recognizer::getWidth(){
	return width;
}

int Recognizer::getHeight(){
	return height;
}

int Recognizer::getPatchSize(){
	return encoder->getPatchSize();
}

int Recognizer::getNumLandmarks(){
	return numLandmarks;
}
