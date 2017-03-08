#include "caffe/data_augmentor.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>

//#define SHOW_RESULT
#ifdef SHOW_RESULT
#include "opencv2/opencv.hpp"
#else
#define uchar unsigned char
#endif
namespace caffe{

DataAugmentor::DataAugmentor(const LayerParameter& layer_param){
  const AugumentationParameter& param = layer_param.augumentation_param();
  phase_ = layer_param.phase();
  motion_blur_ = false;
  gaussian_blur_ = false;
  for (int i = 0; i < param.type_size(); ++i){
    switch (param.type(i)){
    case AugumentationParameter::MOTION_BLUR:
      motion_blur_ = true;
      break;
    case AugumentationParameter::GUASSIAN_BLUR:
      gaussian_blur_ = true;
      break;
    default:
      break;
    }
  }
  if (motion_blur_){
    CHECK(param.blur_radius_size()) << "Blur radius must be set for motion blur!";
    CHECK(param.bulr_direction_size()) << "Blur direction must be set for motion blur!";
    motion_blur_angles_num_ = param.bulr_direction_size();
    for (int i = 0; i < motion_blur_angles_num_; ++i){
      motion_blur_angle_.push_back(param.bulr_direction(i));
    }
    motion_blur_radius_num_ = param.blur_radius_size();
    for (int i = 0; i < motion_blur_radius_num_; ++i){
      motion_blur_radius_.push_back(param.blur_radius(i));
    }

  }
  if (gaussian_blur_){
    CHECK(param.mean_size()) << "Mean of gaussian must be set for gaussian blur!";
    CHECK(param.std_size()) << "Std of gaussian must be set for gaussian blur!";
    CHECK_EQ(param.mean_size(), param.std_size())
      << "mean and std must have same size!";
    gaussian_blur_num_ = param.mean_size();
    for (int i = 0; i < param.blur_radius_size(); ++i){
      guassian_blur_mean_.push_back(param.bulr_direction(i));
      guassian_blur_std_.push_back(param.blur_radius(i));
    }
  }
  InitRand();
}

void DataAugmentor::Augumentation(const Datum& source, Datum& target){
  CHECK_NE(&source, &target) << "Source and target Datum should not be the same one!";
  CHECK_EQ(source.width(), target.width());
  CHECK_EQ(source.height(), target.height());
  CHECK_EQ(source.channels(), target.channels());

  if (motion_blur_){
    const int angle_rnd = Rand(motion_blur_angles_num_);
    const int radius_rnd = (1 == motion_blur_radius_num_) ? 0 : Rand(motion_blur_radius_num_);
    //    LOG(ERROR) << "rnd:" << rnd << " motion_blur_radius_:" << motion_blur_radius_[rnd] << "motion_blur_angle_:" << motion_blur_angle_[rnd];
    DoMotionBlur(source, motion_blur_radius_[radius_rnd],
      motion_blur_angle_[angle_rnd], target);
  }
  if (gaussian_blur_){
    //TODO
  }
}

void DataAugmentor::InitRand(){
  if ((motion_blur_ || gaussian_blur_)/* && 
      phase_ == TRAIN */){
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  }
  else{
    rng_.reset();
  }
}
int DataAugmentor::Rand(int n){
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

void DataAugmentor::DoMotionBlur(const Datum& source, int radius, int angle, Datum& target){
  const string& source_data_string = source.data();
  const char* source_data_buffer = source_data_string.c_str();

  int height = source.height();
  int width = source.width();
  int channels = source.channels();
  string target_data_buffer(channels * height * width, ' ');
  
  if (0 == angle || 180 == angle){
    horizontal_blur_kernel(source_data_buffer, channels, height, width, radius, angle, target_data_buffer);
  }else if (90 == angle || 270 == angle){
    vertical_blur_kernel(source_data_buffer, channels, height, width, radius, angle, target_data_buffer);
  }else if (315 == angle || 135 == angle){
    left_oblique_blur_kernel(source_data_buffer, channels, height, width, radius, angle, target_data_buffer);
  }else if (225 == angle || 45 == angle){
    right_oblique_blur_kernel(source_data_buffer, channels, height, width, radius, angle, target_data_buffer);
  }else{
    LOG(FATAL) << "Unsupported motion blur angle:" << angle;
  }
  
//  CHECK_EQ(target_data_buffer.size(), channels*height*width) << "target size changed!";
  if (target.has_data())
    target.clear_data();
  target.set_data(target_data_buffer);
  //copy labels, note that the old float data in target will be released within this func
  if(source.float_data_size()){
    target.mutable_float_data()->CopyFrom(source.float_data()); 
  }
  if(source.has_label()){
    target.set_label(source.label());
  }
#ifdef SHOW_RESULT
  static int count = 0;
  char filename[256];
  int channel_step = height * width;
  cv::Mat result(height, width, CV_8UC3);
  for (int r = 0; r < height; ++r){
    uchar *row_buffer = result.ptr<uchar>(r);
    for (int c = 0; c < width; ++c){
      for (int ch = 0; ch < 3; ++ch){
        int src_idx = ch * channel_step + r * width + c;
        row_buffer[c * 3 + ch] = static_cast<uchar>(source_data_string[src_idx]);
      }
    }
  }
  sprintf(filename, "org_%d.bmp", count);
  cv::imwrite(filename, result);

  for (int r = 0; r < height; ++r){
    uchar *row_buffer = result.ptr<uchar>(r);
    for (int c = 0; c < width; ++c){
      for (int ch = 0; ch < 3; ++ch){
        int src_idx = ch * channel_step + r * width + c;
        row_buffer[c * 3 + ch] = static_cast<uchar>(target_data_buffer[src_idx]);
      }
    }
  }
  sprintf(filename, "blurred_%d.bmp", count);
  cv::imwrite(filename, result);
  count++;
  /*
  cv::namedWindow("Blurred result");
  imshow("Blurred result", result);
  cv::waitKey(-1);
  cv::destroyWindow("Blurred result");
  */
#endif
}
//Note that the radius should not be larger than height/width
void DataAugmentor::horizontal_blur_kernel(const char* src, int& channels, int& height, int& width,
                            int &radius, int& angle, std::string& dst){
  long radius_sum;
  long last_val;
  int i, j, k;
  int dst_idx = 0;
  int step, offset;

  const char* psr = src;
  const char* ps;
  int dr = 0;

 // LOG(ERROR) << "radius:" << radius << " angle:" << angle << " height:" << height << " width:" << width;
  if (0 == angle){
    step = 1;
    offset = 0;
  }
  else if (180 == angle){
    step = -1;
    offset = width - 1;
  }
  else
    LOG(FATAL) << "Unsupported angle!";

  //0, 180 degree
  for (i = 0; i < channels; ++i){
    for (j = 0; j < height; ++j){
      radius_sum = 0;
      last_val = 0;
      ps = src + offset;
      dst_idx = offset;
      for (k = 0; k < radius - 1; ++k){
        radius_sum += static_cast<long>(static_cast<uchar>(*ps));
        ps += step;
      }
      for (k = 0; k < width - radius; ++k){
        radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
        last_val = static_cast<long>(static_cast<uchar>(*(ps - (radius - 1) * step)));
        dst[dst_idx] = static_cast<char>(radius_sum / radius);
        dst_idx += step;
        ps += step;
      }
      for (k = radius - 1; k >= 0; --k){
        radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
        last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
        dst[dst_idx] = static_cast<char>(radius_sum / radius);
        dst_idx += step;
      }
      offset += width;
    }
  }
}
//Note that the radius should not be larger than height/width
void DataAugmentor::vertical_blur_kernel(const char* src, int& channels, int& height, int& width,
                                        int &radius, int& angle, std::string& dst){
  long radius_sum;
  long last_val;
  int i, j, k;
  int dst_idx = 0;
  int step, offset;
  const char* psc = src;
  const char* ps;
  int dc = 0;
  int channel_step = width * (height - 1);
//  LOG(ERROR) << "radius:" << radius << " angle:" << angle << " height:" << height << " width:" << width;
  if (270 == angle){
    step = width;
    offset = 0;
  }
  else if (90 == angle){
    step = -width;
    offset = channel_step;
  }
  else
    LOG(FATAL) << "Unsupported angle!";

  int back_end = (radius - 1) * step;
  //90, 270 degree
  for (i = 0; i < channels; ++i){
    for (j = 0; j < width; ++j){
      radius_sum = 0;
      last_val = 0;
      ps = src + offset;
      dst_idx = offset;
      for (k = 0; k < radius - 1; ++k){
        radius_sum += static_cast<long>(static_cast<uchar>(*ps));
        ps += step;
      }
      for (k = 0; k < height - radius; ++k){
        radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
        last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
        dst[dst_idx] = static_cast<char>(radius_sum / radius);
        dst_idx += step;
        ps += step;
      }
      for (k = radius - 1; k >= 0; --k){
        radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
        last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
        dst[dst_idx] = static_cast<char>(radius_sum / radius);
        dst_idx += step;
      }
      offset++;
    }
    offset += channel_step;
  }
}
void DataAugmentor::left_oblique_blur_kernel(const char* src, int& channels, int& height, int& width, 
                                            int &radius, int& angle, std::string& dst){
  int i, j, k, m;
  int org_h, org_w, dst_idx = 0;
  long radius_sum, last_val;
  const char* ps;
  int num;
  int channel_step = width * height;
  if (315 == angle){
    int step = width + 1;
    int back_end = (radius - 1) * step;
    org_h = 0;
    org_w = 0;
    for (i = 0; i < channels; ++i){
      for (j = 0; j < height; ++j){
        radius_sum = 0;
        last_val = 0;
        ps = src + org_h;
        dst_idx = org_h;
        m = org_h;
        num = std::min(width, height - j);
        if (num > radius){
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            ps += step;
          }
          for (k = 0; k < num - radius; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
            ps += step;
          }
          for (k = radius - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        else{ //if (num > radius)
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            m += step;
            ps = src + std::min(m, org_h + (num - 1) * step);
          }
          for (k = num - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        org_h += width;
      }
    
      for (j = 1; j < width; ++j){
        radius_sum = 0;
        last_val = 0;
        ps = src + org_w + j;
        dst_idx = org_w + j;
        m = org_w + j;
        num = std::min(height, width - j);
        if (num > radius){
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            ps += step;
          }
          for (k = 0; k < num - radius; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
            ps += step;
          }
          for (k = radius - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }//if (num > radius)
        else{
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            m += step;
            ps = src + std::min(m, org_w + j + (num - 1) * step);
          }
          for (k = num - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
      }
      org_w += channel_step;
    }
  }
  else if (135 == angle){
    int step = -(width + 1);
    int back_end = (radius - 1) * step;
  
    org_h = width - 1;
    org_w = (height - 1) * width - 1;
    for (i = 0; i < channels; ++i){
      for (j = 1; j <= height; ++j){
        radius_sum = 0;
        last_val = 0;
        ps = src + org_h;
        dst_idx = org_h;
        m = org_h;
        num = std::min(j, width);
        if (num > radius){
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            ps += step;
          }
          for (k = 0; k < num - radius; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
            ps += step;
          }
          for (k = radius - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        else{
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            m += step;
            ps = src + std::max(m, org_h + (num - 1) * step);
          }
          for (k = num - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        org_h += width;
      }
      for (j = 1; j < width; ++j){
        radius_sum = 0;
        last_val = 0;
        ps = src + org_w + j;
        dst_idx = org_w + j;
        m = org_w + j;
        num = std::min(height, j);
        if (num > radius){
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            ps += step;
          }
          for (k = 0; k < num - radius; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
            ps += step;
          }
          for (k = radius - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        else{
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            m += step;
            ps = src + std::max(m, dst_idx + (num - 1) * step);
          }
          for (k = num - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
      }
      org_w += channel_step;
    }
  }
}

void DataAugmentor::right_oblique_blur_kernel(const char* src, int& channels, int& height, int& width,
                                              int &radius, int& angle, std::string& dst){
  int i, j, k, m;
  int org_h, org_w, dst_idx = 0;
  long radius_sum, last_val;
  const char* ps;
  int num;
  int channel_step = width * height;
  if (45 == angle){
    int step = -width + 1;
    int back_end = (radius - 1) * step;
    org_h = 0;
    org_w = (height - 1) * width;
    for (i = 0; i < channels; ++i){
      for (j = 1; j <= height; ++j){
        radius_sum = 0;
        last_val = 0;
        ps = src + org_h;
        dst_idx = org_h;
        m = org_h;
        num = std::min(width, j);
        if (num > radius){
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            ps += step;
          }
          for (k = 0; k < num - radius; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
            ps += step;
          }
          for (k = radius - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        else{ //if (num > radius)
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            m += step;
            ps = src + std::max(m, org_h + (num - 1) * step);
          }
          for (k = num - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        org_h += width;
      }

      for (j = 1; j < width; ++j){
        radius_sum = 0;
        last_val = 0;
        ps = src + org_w + j;
        dst_idx = org_w + j;
        m = org_w + j;
        num = std::min(height, width - j);
        if (num > radius){
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            ps += step;
          }
          for (k = 0; k < num - radius; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
            ps += step;
          }
          for (k = radius - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }//if (num > radius)
        else{
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            m += step;
            ps = src + std::max(m, org_w + j + (num - 1) * step);
          }
          for (k = num - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
      }
      org_w += channel_step;
    }
  }
  else if (225 == angle){
    int step = width - 1;
    int back_end = (radius - 1) * step;

    org_h = width - 1;
    org_w = -1;
    for (i = 0; i < channels; ++i){
      for (j = 0; j < height; ++j){
        radius_sum = 0;
        last_val = 0;
        ps = src + org_h;
        dst_idx = org_h;
        m = org_h;
        num = std::min(height - j, width);
        if (num > radius){
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            ps += step;
          }
          for (k = 0; k < num - radius; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
            ps += step;
          }
          for (k = radius - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        else{
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            m += step;
            ps = src + std::min(m, org_h + (num - 1) * step);
          }
          for (k = num - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        org_h += width;
      }
      for (j = 1; j < width; ++j){
        radius_sum = 0;
        last_val = 0;
        ps = src + org_w + j;
        dst_idx = org_w + j;
        m = org_w + j;
        num = std::min(height, j);
        if (num > radius){
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            ps += step;
          }
          for (k = 0; k < num - radius; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - back_end)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
            ps += step;
          }
          for (k = radius - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
        else{
          for (k = 0; k < radius - 1; ++k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps));
            m += step;
            ps = src + std::min(m, dst_idx + (num - 1) * step);
          }
          for (k = num - 1; k >= 0; --k){
            radius_sum += static_cast<long>(static_cast<uchar>(*ps)) - last_val;
            last_val = static_cast<long>(static_cast<uchar>(*(ps - k * step)));
            dst[dst_idx] = static_cast<char>(radius_sum / radius);
            dst_idx += step;
          }
        }
      }
      org_w += channel_step;
    }
  }
}

}//namepace caffe
