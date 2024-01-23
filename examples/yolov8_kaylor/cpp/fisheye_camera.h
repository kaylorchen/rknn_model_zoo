//
// Created by kaylor on 1/18/24.
//

#pragma once
#include "kaylordut/logger.h"
#include "opencv2/opencv.hpp"

class FisheyeCamera {
 public:
  FisheyeCamera(uint16_t index, cv::InputArray K, cv::InputArray D);
  ~FisheyeCamera();
  void Refresh();
  std::shared_ptr<cv::Mat> GetRgbFrame();
  std::shared_ptr<cv::Mat> GetRgbFrame(int &&derired_size);

 private:
  cv::VideoCapture capture_;
  cv::Mat map1_, map2_;
  cv::Size size_;
};
