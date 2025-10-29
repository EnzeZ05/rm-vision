#include "Armor.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <iostream>
// #include <onnxruntime_cxx_api.h>

void YOLO() {
    const std::string path = "samples/far_001.jpg";
    cv::Mat img = cv::imread(path);

    Armor armor;

    cv::Mat roi;
    std::vector<cv::Mat> mat = armor.process(path, &roi, 0);

    
}

int main() {
    YOLO();
}