//
//  GCApp.h
//  Graph_Cut
//
//  Created by 张雨 on 7/17/15.
//  Copyright (c) 2015 Zhang, Yu. All rights reserved.
//

#ifndef __Graph_Cut__GCApp__
#define __Graph_Cut__GCApp__

#include "GraphCut.h"

class GCApp{
public:
    void init(cv::string);
    void run();
    static void main_steps(int pos);
    const cv::string k_project_dir = "/Users/Rango/Documents/coding/XCode/Graph_Cut";
};

#endif /* defined(__Graph_Cut__GCApp__) */
