//
//  GraphCut.h
//  Graph_Cut
//
//  Created by 张雨 on 7/17/15.
//  Copyright (c) 2015 Zhang, Yu. All rights reserved.
//

#ifndef __Graph_Cut__GraphCut__
#define __Graph_Cut__GraphCut__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>

#include "graph.h"

#define DEBUGING 0

class GraphCut{
public:
    inline static GraphCut* get_singleton(){
        if(!graph_cut_){
            graph_cut_ = new GraphCut();
        }
        return graph_cut_;
    }
    inline void print(){
        std::cout << "Testing..." << std::endl;
        imshow("markupImage",markup_image_);
        cv::waitKey(0);
    }
    
    //core functions
    void set_markupImage(cv::Mat);
    void set_sourceImage(cv::Mat);
    void set_parameters(float*, int n);
    void graph_cut_pipline();
    void extract_results(cv::Mat&);
    
private:
    static GraphCut *graph_cut_;
    
    float* para_;
    
    cv::Mat markup_image_,
            source_image_,
            fg_centroids_,      //foreground prior model
            bg_centroids_,      //background prior model
            final_result_;
    
    Graph* graph_;
    Graph::node_id* nodes_;
    
    //parameters
    const int   centroids_num = 64;
    const int   k_foreground = 1,
                k_background = 0;
    
    //main steps of graph cut
    void establish_prior_model();
    void build_graph();
    void compute_maxflow_mincut();
    float minDistFromPriorModel(cv::Vec3b, int);
};

#endif /* defined(__Graph_Cut__GraphCut__) */
