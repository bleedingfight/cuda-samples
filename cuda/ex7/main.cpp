#include <bits/stdc++.h>
#include <opencv2>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace dnn;
using namespace cv;
using namespace std;
int main() {
    string name = "/home/liushuai/图片/logo.jpg";
    // Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    vector<string> classes;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    cv::String modelConfiguration = "yolov3.cfg";
    cv::String modelWeights = "yolov3.weights";


    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    cv::Mat img = cv::imread(name);
    while(true)
    {
        int k = cv::waitKey();
        if(k == 27)
            break;
        cv::imshow("name",img);
    }
    cv::destroyAllWindows();
    return 0;
}