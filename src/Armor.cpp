//
// Created by 13578 on 2025/10/27.
//

#include "Armor.h"

#include <numeric>

Armor::Armor(int area_min_,
             double max_dx_ratio_,
             double height_ratio_lo_,
             double height_ratio_hi_,
             int pad_x_min_,
             double min_w_ratio_,
             double keep_ratio_)
    : area_min(area_min_),
      max_dx_ratio(max_dx_ratio_),
      height_ratio_lo(height_ratio_lo_),
      height_ratio_hi(height_ratio_hi_),
      pad_x_min(pad_x_min_),
      min_w_ratio(min_w_ratio_),
      keep_ratio(keep_ratio_) {
}

void Armor::show(const cv::Mat &ref, const std::string &win) {
    if (ref.empty()) return;
    cv::imshow(win, ref);
    cv::waitKey(0);
    cv::destroyWindow(win);
}

std::array<cv::Point2f, 4> Armor::get_points(const std::vector<cv::Point2f> &pts4) {
    CV_Assert(pts4.size() == 4);
    std::array<cv::Point2f, 4> pts{};
    auto sum = [](const cv::Point2f &p) { return p.x + p.y; };
    auto diff = [](const cv::Point2f &p) { return p.y - p.x; };

    int tl = 0, tr = 0, br = 0, bl = 0;
    float minSum = std::numeric_limits<float>::max();
    float maxSum = -std::numeric_limits<float>::max();
    float minDiff = std::numeric_limits<float>::max();
    float maxDiff = -std::numeric_limits<float>::max();
    for (int i = 0; i < 4; ++i) {
        float s = sum(pts4[i]);
        float d = diff(pts4[i]);
        if (s < minSum) {
            minSum = s;
            tl = i;
        }
        if (s > maxSum) {
            maxSum = s;
            br = i;
        }
        if (d < minDiff) {
            minDiff = d;
            tr = i;
        }
        if (d > maxDiff) {
            maxDiff = d;
            bl = i;
        }
    }
    pts[0] = pts4[tl];
    pts[1] = pts4[tr];
    pts[2] = pts4[br];
    pts[3] = pts4[bl];
    return pts;
}

std::pair<int, int> Armor::resize_from_quad(const std::array<cv::Point2f, 4> &q) {
    const cv::Point2f &tl = q[0], &tr = q[1], &br = q[2], &bl = q[3];
    int W = static_cast<int>(std::round(std::max(cv::norm(br - bl), cv::norm(tr - tl))));
    int H = static_cast<int>(std::round(std::max(cv::norm(tr - br), cv::norm(tl - bl))));
    W = std::max(W, 1);
    H = std::max(H, 1);
    return {W, H};
}

cv::Mat Armor::get_mask(const cv::Mat &img /*, const std::string& team = "blue"*/) {
    cv::Mat gray, mask;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 220, 255, cv::THRESH_BINARY);
    return mask;
}

std::pair<cv::RotatedRect, std::vector<cv::Point2f> >
Armor::extend_min_area_rect(const cv::RotatedRect &rect, float extra_ratio) {
    cv::RotatedRect out = rect;
    if (rect.size.width >= rect.size.height) {
        out.size.width = rect.size.width * (1.0f + 2.0f * extra_ratio);
        out.size.height = rect.size.height;
    } else {
        out.size.width = rect.size.width;
        out.size.height = rect.size.height * (1.0f + 2.0f * extra_ratio);
    }
    std::vector<cv::Point2f> box(4);
    out.points(box.data());
    return {out, box};
}

bool Armor::process(const std::string &img_path, cv::Mat *out_digit_roi, bool visualize) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << img_path << "\n";
        return false;
    }
    cv::Mat last = img.clone();

    cv::Mat mask = get_mask(img);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> bbs(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) bbs[i] = cv::boundingRect(contours[i]);
    std::vector<int> order(contours.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return bbs[a].x < bbs[b].x; });

    int H_img = img.rows, W_img = img.cols;
    int max_dx = std::max(20, static_cast<int>(max_dx_ratio * W_img));

    for (size_t oi = 0; oi < order.size(); ++oi) {
        int i = order[oi];
        if (cv::contourArea(contours[i]) < area_min) continue;

        const cv::Rect &Ai = bbs[i];
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            int j = order[oj];
            const cv::Rect &Bj = bbs[j];

            if (Bj.x - (Ai.x + Ai.width) > max_dx) break;
            if (cv::contourArea(contours[j]) < area_min) continue;

            double hr = static_cast<double>(Bj.height + 1e-6) / static_cast<double>(Ai.height + 1e-6);
            if (!(height_ratio_lo <= hr && hr <= height_ratio_hi)) continue;

            int top = std::max(Ai.y, Bj.y);
            int bot = std::min(Ai.y + Ai.height, Bj.y + Bj.height);
            if (bot - top <= 0) continue; // no vertical overlap

            cv::RotatedRect rectA = cv::minAreaRect(contours[i]);
            cv::RotatedRect rectB = cv::minAreaRect(contours[j]);

            auto extA = extend_min_area_rect(rectA, 0.5f);
            auto extB = extend_min_area_rect(rectB, 0.5f);

            std::vector<cv::Point2f> pts_all;
            pts_all.reserve(8);
            pts_all.insert(pts_all.end(), extA.second.begin(), extA.second.end());
            pts_all.insert(pts_all.end(), extB.second.begin(), extB.second.end());

            std::vector<cv::Point2f> hull;
            cv::convexHull(pts_all, hull, true, true);

            double peri = 0.0;
            for (size_t k = 0; k < hull.size(); ++k) {
                peri += cv::norm(hull[k] - hull[(k + 1) % hull.size()]);
            }
            std::vector<cv::Point2f> approx;
            cv::approxPolyDP(hull, approx, 0.02 * peri, true);

            if (approx.size() == 4) {
                auto quad = get_points(approx);
                auto [Wp, Hp] = resize_from_quad(quad);

                std::vector<cv::Point2f> dst = {
                    {0.f, 0.f},
                    {static_cast<float>(Wp - 1), 0.f},
                    {static_cast<float>(Wp - 1), static_cast<float>(Hp - 1)},
                    {0.f, static_cast<float>(Hp - 1)}
                };

                cv::Mat dmat = cv::getPerspectiveTransform(std::vector<cv::Point2f>(quad.begin(), quad.end()), dst);
                if (cv::countNonZero(cv::Mat(dmat != dmat)) == 0) {
                    cv::Mat patch;
                    cv::warpPerspective(img, patch, dmat, cv::Size(Wp, Hp), cv::INTER_CUBIC, cv::BORDER_REPLICATE);

                    std::vector<cv::Point2f> boxA(4), boxB(4);
                    rectA.points(boxA.data());
                    rectB.points(boxB.data());
                    std::vector<cv::Point2f> boxA_w, boxB_w;
                    cv::perspectiveTransform(boxA, boxA_w, dmat);
                    cv::perspectiveTransform(boxB, boxB_w, dmat);

                    cv::Mat vis = patch.clone();
                    cv::Rect ra = cv::boundingRect(boxA_w);
                    cv::Rect rb = cv::boundingRect(boxB_w);

                    int H = patch.rows, W = patch.cols;

                    cv::Rect L = (ra.x <= rb.x) ? ra : rb;
                    cv::Rect R = (ra.x <= rb.x) ? rb : ra;

                    int pad_x = std::max(pad_x_min, static_cast<int>(0.000001 * W));
                    int x1 = std::max(0, L.x + L.width - pad_x);
                    int x2 = std::min(W, R.x + pad_x);
                    int y1 = 0, y2 = H;

                    int min_w = std::max(12, static_cast<int>(min_w_ratio * W));
                    int w = x2 - x1;
                    if (w >= min_w) {
                        int cx = (x1 + x2) / 2;
                        int w_keep = std::max(min_w, static_cast<int>(w * keep_ratio));
                        x1 = std::max(0, cx - w_keep / 2);
                        x2 = std::min(W, x1 + w_keep);
                        if (x2 > x1) {
                            cv::rectangle(vis, {x1, y1}, {x2, y2}, {0, 0, 255}, 2);
                            cv::Mat digit_roi = patch(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
                            if (visualize) {
                                show(vis, "patch");
                                show(digit_roi, "digit_roi");
                            }
                            if (out_digit_roi) *out_digit_roi = digit_roi;
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}
