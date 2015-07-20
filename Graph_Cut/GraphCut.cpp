//
//  GraphCut.cpp
//  Graph_Cut
//
//  Created by 张雨 on 7/17/15.
//  Copyright (c) 2015 Zhang, Yu. All rights reserved.
//

#include "GraphCut.h"
#include <stdlib.h>

float BGRDistSq(cv::Vec3b, cv::Vec3b);

GraphCut* GraphCut::graph_cut_ = NULL;

void GraphCut::set_markupImage(cv::Mat markup_image){
    this->markup_image_ = markup_image;
}

void GraphCut::set_sourceImage(cv::Mat sourceImage){
    this->source_image_ = sourceImage;
}

void GraphCut::set_parameters(float* para, int n){
    para_ = new float[n];
    for (int i = 0; i < n; ++i) {
        para_[i] = para[i];
    }
}

void GraphCut::graph_cut_pipline(){
    establish_prior_model();
    build_graph();
    compute_maxflow_mincut();
}

void GraphCut::extract_results(cv::Mat& result){
    //for test
    final_result_ = source_image_.clone();
    Graph::node_id* nodeIt = nodes_;
    
    for (int i = 0; i < source_image_.rows; ++i) {
        for (int j = 0; j < source_image_.cols; ++j, ++nodeIt) {
            if (graph_->what_segment(*nodeIt) == Graph::SINK) {
                final_result_.at<cv::Vec3b>(i, j)[0] = 0;
                final_result_.at<cv::Vec3b>(i, j)[1] = 0;
                final_result_.at<cv::Vec3b>(i, j)[2] = 0;
            }else{
                final_result_.at<cv::Vec3b>(i, j)[0] = 255;
                final_result_.at<cv::Vec3b>(i, j)[1] = 255;
                final_result_.at<cv::Vec3b>(i, j)[2] = 255;
            }
        }
    }
    result = final_result_;
}

void GraphCut::establish_prior_model(){
    cv::vector<cv::Vec3f> fgLabeledPixels, bgLabeledPixels;
    
    cv::Mat_<cv::Vec3b> markupImage = markup_image_;
    cv::Mat_<cv::Vec3b> sourceImage = source_image_;

    for (int i = 0; i < markup_image_.rows; ++i) {
        for (int j = 0; j < markup_image_.cols; ++j) {
            if (markupImage(i, j)[0] == 0 &&
                markupImage(i, j)[1] == 0 &&
                markupImage(i, j)[2] == 255 ){
                fgLabeledPixels.push_back(sourceImage(i, j));
            }else if (
                markupImage(i, j)[0] == 255 &&
                markupImage(i, j)[1] == 0 &&
                markupImage(i, j)[2] == 0){
                bgLabeledPixels.push_back(sourceImage(i, j));
            }
        }
    }
    
    cv::Mat fgLabeledPixelsMat(fgLabeledPixels),
            bgLabeledPixelsMat(bgLabeledPixels),
            labels;
    
    cv::kmeans(fgLabeledPixelsMat, centroids_num, labels, cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0)
               , 1, cv::KMEANS_PP_CENTERS, fg_centroids_);
    cv::kmeans(bgLabeledPixelsMat, centroids_num, labels, cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0)
               , 1, cv::KMEANS_PP_CENTERS, bg_centroids_);
    
#ifdef DEBUG
    std::cout << fg_centroids_ << std::endl << std::endl << bg_centroids_ << std::endl;
    
    cv::Mat imageOfCentroids(256, 256, CV_8UC3);
    for (int i = 0; i < 64; ++i) {
        cv::circle(imageOfCentroids, *new cv::Point(bg_centroids_.at<cv::Vec3f>(i,0)[1], bg_centroids_.at<cv::Vec3f>(i,0)[2]), 2, cv::Scalar(255,0,0));
        cv::circle(imageOfCentroids, *new cv::Point(fg_centroids_.at<cv::Vec3f>(i,0)[1], fg_centroids_.at<cv::Vec3f>(i,0)[2]), 2, cv::Scalar(0,0,255));
    }
    imshow("centroid", imageOfCentroids);
    cv::waitKey(0);
#endif
    
}

void GraphCut::build_graph(){
    int cols=source_image_.cols, rows = source_image_.rows;
    int numOfPixels = rows * cols;
    cv::Mat_<cv::Vec3b> sourceImage = source_image_;
    
    nodes_ = new Graph::node_id[numOfPixels];
    graph_ = new Graph();
    
    for (unsigned int i = 0; i < numOfPixels; ++i){
        nodes_[i] = graph_->add_node();
    }
 
    //gradient
    cv::Vec3b cache;

    for (int i = 0; i < source_image_.rows; ++i) {
        cache = sourceImage(i, 0);
        for (int j = 1; j < source_image_.cols; ++j) {
            Graph::captype weight = para_[0] / (1+ BGRDistSq(cache, sourceImage(i, j)));
            graph_->add_edge(nodes_[i*cols+j], nodes_[i*cols+j-1], weight, weight);
            cache = sourceImage(i, j);
        }
    }
    
    for (int i = 0; i < source_image_.cols; ++i) {
        cache = sourceImage(0, i);
        for (int j = 1; j < source_image_.rows; ++j) {
            Graph::captype weight = para_[0] / (1 + BGRDistSq(cache, sourceImage(j, i)));
            graph_->add_edge(nodes_[j*cols+i], nodes_[j*cols+i-cols], weight, weight);
            cache = sourceImage(j, i);
        }
    }
    
    //color
#pragma omp for
    for (int i = 0; i < rows; ++i) {
#pragma omp for
        for (int j = 0; j < cols; ++j) {
            if (sourceImage(i, j)[0] == 0 &&
                sourceImage(i, j)[1] == 0 &&
                sourceImage(i, j)[2] == 255) {
                graph_->set_tweights(nodes_[i*cols+j], (Graph::captype)INFINITY, (Graph::captype)0);
            }else if(
                sourceImage(i, j)[0] == 255 &&
                sourceImage(i, j)[1] == 0 &&
                sourceImage(i, j)[2] == 0) {
                graph_->set_tweights(nodes_[i*cols+j], (Graph::captype)0, (Graph::captype)INFINITY);
            }else{
                float fgDist = minDistFromPriorModel(sourceImage(i, j), k_foreground);
                float bgDist = minDistFromPriorModel(sourceImage(i, j), k_background);
                graph_->set_tweights(nodes_[i*cols+j], (Graph::captype)(bgDist/(bgDist+fgDist)), (Graph::captype)(fgDist/(fgDist+bgDist)));
            }
        }
    }
    
}

void GraphCut::compute_maxflow_mincut(){
    graph_->maxflow();
}

float BGRDistSq(cv::Vec3b p1, cv::Vec3b p2){
    float result = 0;
    for (int i = 0; i < 3; ++i) {
        result += (p1[i]-p2[i])*(p1[i]-p2[i]);
    }
    return result;
}

float GraphCut::minDistFromPriorModel(cv::Vec3b node, int name){
    float min = INFINITY;
    for (int i = 0; i < centroids_num; ++i) {
        if (name == k_background) {
            float Dist = BGRDistSq(bg_centroids_.at<cv::Vec3f>(i), node);
            min = (Dist < min)? Dist:min;
        }else if (name == k_foreground){
            float Dist = BGRDistSq(fg_centroids_.at<cv::Vec3f>(i), node);
            min = (Dist < min)? Dist:min;
        }
    }
    return min;
}

















