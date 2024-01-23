//
// Created by kaylor on 1/18/24.
//

#include "fisheye_camera.h"

FisheyeCamera::FisheyeCamera(uint16_t index, cv::InputArray K, cv::InputArray D)
    : capture_(index, cv::CAP_V4L2), size_(1920, 1080) {
  KAYLORDUT_LOG_INFO("Instantiate a FisheyeCamera object");
  // 这里使用V4L2捕获，因为使用默认的捕获不可以设置捕获的模式和帧率
  if (!capture_.isOpened()) {
    KAYLORDUT_LOG_ERROR("Error opening video stream or file");
    exit(EXIT_FAILURE);
  }
  capture_.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
  capture_.set(cv::CAP_PROP_FPS, 30);
  capture_.set(cv::CAP_PROP_FOURCC,
               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  // 检查是否成功设置格式
  int fourcc = capture_.get(cv::CAP_PROP_FOURCC);
  if (fourcc != cv::VideoWriter::fourcc('M', 'J', 'P', 'G')) {
    KAYLORDUT_LOG_WARN("Set video format failed");
  }
  cv::Mat newK = cv::getOptimalNewCameraMatrix(K, D, size_, 1, size_, 0);
  cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat(), newK, size_, CV_32FC1,
                                       map1_, map2_);
}

FisheyeCamera::~FisheyeCamera() {
  KAYLORDUT_LOG_INFO("Release sources.")
  capture_.release();
  cv::destroyAllWindows();
}

void FisheyeCamera::Refresh() {
  cv::Mat frame;
  cv::Mat undistortedImage;
  while (true) {
    capture_ >> frame;
    if (frame.empty()) {
      break;
    }
    cv::remap(frame, undistortedImage, map1_, map2_, cv::INTER_LINEAR);
    cv::imshow("Fisheye camera", undistortedImage);
    if (cv::waitKey(1) >= 0) {
      break;
    }
  }
}

std::shared_ptr<cv::Mat> FisheyeCamera::GetRgbFrame() {
  cv::Mat frame;
  cv::Mat undistortedImage;
  capture_ >> frame;
  if (frame.empty()) {
    KAYLORDUT_LOG_WARN("frame is empty");
    return nullptr;
  }
  cv::remap(frame, undistortedImage, map1_, map2_, cv::INTER_LINEAR);
  auto tmp = std::make_shared<cv::Mat>();
  cv::cvtColor(undistortedImage, *tmp, cv::COLOR_BGR2RGB);
  cv::imwrite("origin.png", frame);
  cv::imwrite("un.png", undistortedImage);
  return std::move(tmp);
}

std::shared_ptr<cv::Mat> FisheyeCamera::GetRgbFrame(int &&derired_size) {
  cv::Mat frame;
  cv::Mat undistortedImage;
  capture_ >> frame;
  if (frame.empty()) {
    KAYLORDUT_LOG_WARN("frame is empty");
    return nullptr;
  }
  cv::imwrite("111.png", frame);
  cv::remap(frame, undistortedImage, map1_, map2_, cv::INTER_LINEAR);
  // 计算新的宽度和高度保持相同的宽高比
  int desired_size = 640;
  int new_width, new_height;
  if (undistortedImage.rows > undistortedImage.cols) {
    new_height = desired_size;
    new_width = undistortedImage.cols * desired_size / undistortedImage.rows;
  } else {
    new_width = desired_size;
    new_height = undistortedImage.rows * desired_size / undistortedImage.cols;
  }
  // 首先缩放图像
  cv::Mat scaledImage;
  cv::resize(undistortedImage, scaledImage, cv::Size(new_width, new_height));

  // 然后创建一个640x640的空白图像
  cv::Mat squareImage =
      cv::Mat::zeros(desired_size, desired_size, undistortedImage.type());

  // 计算中心位置进行粘贴
  int x_offset = (desired_size - new_width) / 2;
  int y_offset = (desired_size - new_height) / 2;

  // 粘贴缩放后的图像到中心
  scaledImage.copyTo(squareImage(
      cv::Rect(x_offset, y_offset, scaledImage.cols, scaledImage.rows)));
  auto tmp = std::make_shared<cv::Mat>(squareImage.rows, squareImage.cols, squareImage.type());
  cv::cvtColor(squareImage, *tmp, cv::COLOR_BGR2RGB);


  return std::move(tmp);
}