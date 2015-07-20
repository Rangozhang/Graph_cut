//
//  GCApp.cpp
//  Graph_Cut
//
//  Created by 张雨 on 7/17/15.
//  Copyright (c) 2015 Zhang, Yu. All rights reserved.
//

#include "GCApp.h"

void GCApp::main_steps(int pos){
    
    float* para = new float[1];
    para[0] = 2500 + pos;
    GraphCut::get_singleton()->set_parameters(para, 1);
    
    cv::Mat resultImage;
    GraphCut::get_singleton()->graph_cut_pipline();
    GraphCut::get_singleton()->extract_results(resultImage);
    
    imshow("result", resultImage);
}

void GCApp::init(cv::string imageName){
    cv::Mat markupImage = cv::imread(k_project_dir + "/images_s/" + imageName);
    cv::Mat sourceImage = cv::imread(k_project_dir + "/images/" + imageName);
    GraphCut::get_singleton()->set_markupImage(markupImage);
    GraphCut::get_singleton()->set_sourceImage(sourceImage);
    imshow("Source Image", sourceImage);
}

void GCApp::run(){

    init("16.bmp");
    
    cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
    
    int nThreshold = 1500;
    cvCreateTrackbar("weight", "result", &nThreshold, 3000, main_steps);
    
    main_steps(4000);
    
    cv::waitKey(0);
}

