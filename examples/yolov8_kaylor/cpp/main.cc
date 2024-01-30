// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "file_utils.h"
#include "fisheye_camera.h"
#include "image_drawing.h"
#include "image_utils.h"
#include "yolov8.h"
#include "thread"
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv) {

  // 定义相机内参矩阵K
  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 518.5206194361012, 0.0, 932.0926791943779, 0.0,
          518.0241546073428, 507.22695301062527, 0.0, 0.0, 1.0);

  // 定义畸变系数矩阵D
  cv::Mat D =
      (cv::Mat_<double>(4, 1) << -0.09556402717747697, 0.012374049436718767,
          -0.010465758469831311, 0.0033159128053917544);
  FisheyeCamera fisheye_camera(0, K, D);
  if (argc != 3) {
    printf("%s <model_path> <image_path>\n", argv[0]);
    return -1;
  }

  const char *model_path = argv[1];
  const char *image_path = argv[2];

  int ret;
  rknn_app_context_t rknn_app_ctx;
  memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

  init_post_process();

  ret = init_yolov8_model(model_path, &rknn_app_ctx);
  if (ret != 0) {
    printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0) {
      printf("release_yolov8_model fail! ret=%d\n", ret);
    }
    exit(EXIT_FAILURE);
  }

  image_buffer_t src_image;
  memset(&src_image, 0, sizeof(image_buffer_t));

  src_image.width = 640;
  src_image.height = 640;
  src_image.size = src_image.width * src_image.height * 3;
  src_image.format = IMAGE_FORMAT_RGB888;
  std::shared_ptr<cv::Mat> tmp = fisheye_camera.GetRgbFrame(640);
  src_image.virt_addr = tmp->ptr();

//    ret = read_image(image_path, &src_image);
//  if (ret != 0) {
//    printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
//    goto out;
//  }

  object_detect_result_list od_results;

  ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
  if (ret != 0) {
    printf("init_yolov8_model fail! ret=%d\n", ret);
    deinit_post_process();
    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0) {
      printf("release_yolov8_model fail! ret=%d\n", ret);
    }
    exit(EXIT_FAILURE);
  }

  // 画框和概率
  char text[256];
  for (int i = 0; i < od_results.count; i++) {
    object_detect_result *det_result = &(od_results.results[i]);
    printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
           det_result->box.left, det_result->box.top, det_result->box.right,
           det_result->box.bottom, det_result->prop);
    int x1 = det_result->box.left;
    int y1 = det_result->box.top;
    int x2 = det_result->box.right;
    int y2 = det_result->box.bottom;

    draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

    sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id),
            det_result->prop * 100);
    draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
  }

  write_image("out.png", &src_image);

//out:
//  if (src_image.virt_addr != NULL) {
//    free(src_image.virt_addr);
//  }
//
  return 0;
}

