#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <iostream>

int main() {
    const std::string onnx = "yolo11n.onnx";
    const std::string path = "samples/img.jpg";
    // std::cout << cv::getBuildInformation() << std::endl;
    cv::Mat img = cv::imread(path);
    cv::imshow("bad", img);
    if (img.empty()) { std::cerr << "load failed\n"; return 1; }

    cv::dnn::Net net = cv::dnn::readNet(onnx);
    // If you built with CUDA:
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(640,640),
                                       cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs);

    // visualize the preprocessed input (BGR in range [0,1] -> scale back to [0,255])
    std::vector<cv::Mat> imgs;
    cv::dnn::imagesFromBlob(blob, imgs);
    cv::Mat vis = imgs[0].clone();
    vis.convertTo(vis, CV_8U, 255.0);
    cv::imshow("preprocessed", vis);
    cv::waitKey(0);

    // Warmup
    // for (int i = 0; i < 10; ++i) {
    //     cv::Mat blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(640,640), cv::Scalar(), true, false);
    //     net.setInput(blob);
    //     std::vector<cv::Mat> outs;
    //     net.forward(outs);
    //     cv::imshow("image", blob);
    // }
    //
    // // Timed runs
    // int iters = 100;
    // auto t0 = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < iters; ++i) {
    //     cv::Mat blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(640,640), cv::Scalar(), true, false);
    //     net.setInput(blob);
    //     std::vector<cv::Mat> outs;
    //     net.forward(outs);
    // }
    // auto t1 = std::chrono::high_resolution_clock::now();
    // double dt = std::chrono::duration<double>(t1 - t0).count();
    // double fps = iters / dt;
    // std::cout << iters << " runs: " << dt << "s  ->  " << fps << " FPS\n";
}
