
//
// Created by 13578 on 2025/10/27.
//

#ifndef ARMOR_H
#define ARMOR_H

#include <string>
#include <opencv2/opencv.hpp>

class Armor {
public:
    int area_min;
    double max_dx_ratio;
    double height_ratio_lo;
    double height_ratio_hi;
    int pad_x_min;
    double min_w_ratio;
    double keep_ratio;

    Armor(int area_min = 100,
          double max_dx_ratio = 0.25,
          double height_ratio_lo = 0.5,
          double height_ratio_hi = 2.0,
          int pad_x_min = 2,
          double min_w_ratio = 0.03,
          double keep_ratio = 0.60);

    std::vector<cv::Mat> process(const std::string& img_path, cv::Mat* out_digit_roi = nullptr, bool visualize = false);
    virtual void show(const cv::Mat& ref, const std::string& win = "image");

    static std::array<cv::Point2f,4> get_points(const std::vector<cv::Point2f>& pts4);
    static std::pair<int,int> resize_from_quad(const std::array<cv::Point2f,4>& q);
    static cv::Mat get_mask(const cv::Mat& img /*, const std::string& team = "blue"*/ );
    static std::pair<cv::RotatedRect, std::vector<cv::Point2f>>
        extend_min_area_rect(const cv::RotatedRect& rect, float extra_ratio = 0.5f);
};

#endif //ARMOR_H