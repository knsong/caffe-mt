#ifndef CAFFE_DATA_AUGMENTOR_H_
#define CAFFE_DATA_AUGMENTOR_H_

#include "caffe/proto/caffe.pb.h"
#include "common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe{

class DataAugmentor
{
 public:
  explicit DataAugmentor(const LayerParameter& layer_param);
  ~DataAugmentor(){};
  void DoMotionBlur(const Datum& source, int radius, int angle, Datum& target);
  void Augumentation(const Datum& source, Datum& target);
  void InitRand();
  int Rand(int n);
 protected:
  void horizontal_blur_kernel(const char* src, int& channels, int& height, int& width,
                      int &radius, int& angle, std::string& dst);
  void vertical_blur_kernel(const char* src, int& channels, int& height, int& width,
                      int &radius, int& angle, std::string& dst);
  void left_oblique_blur_kernel(const char* src, int& channels, int& height, int& width,
                      int &radius, int& angle, std::string& dst);
  void right_oblique_blur_kernel(const char* src, int& channels, int& height, int& width,
                      int &radius, int& angle, std::string& dst);
 private:
  std::vector<int> motion_blur_angle_;
  std::vector<int> motion_blur_radius_;
  std::vector<float> guassian_blur_mean_;
  std::vector<float> guassian_blur_std_;
  shared_ptr<Caffe::RNG> rng_;
  int motion_blur_angles_num_;
  int motion_blur_radius_num_;
  int gaussian_blur_num_;
  bool motion_blur_;
  bool gaussian_blur_;
  int phase_;
};



}//namespace caffe
#endif//#ifndef CAFFE_DATA_AUGUMENTOR_H_