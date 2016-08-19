#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dyn_conv_layer.hpp"


#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {


template <typename TypeParam>
class DynamicConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DynamicConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 3, 3)),
        blob_bottom_2_(new Blob<Dtype>()),
        blob_bottom_3_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Blob<Dtype> tmp(1, 1, 3, 3);
    int spatial_dim = tmp.count(2);
    FillerParameter filler_param;
    //fill input data 
    for(int i = 0; i < blob_bottom_->shape(0); ++i){
      int num_offset = blob_bottom_->count(1);
      for(int j = 0; j < blob_bottom_->shape(1); ++j){
        filler_param.set_value(j + 1);
        ConstantFiller<Dtype> filler(filler_param);
        filler.Fill(&tmp);
        caffe_copy(tmp.count(), tmp.cpu_data()
          , this->blob_bottom_->mutable_cpu_data() + i * num_offset + j * spatial_dim);
      }
    }
	this->bias_val_ = 1;
	blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DynamicConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_top_;;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }
  Dtype bias_val_;
  Blob<Dtype>* const blob_bottom_;   //input data
  Blob<Dtype>* const blob_bottom_2_; //kernels
  Blob<Dtype>* const blob_bottom_3_; //bias
  Blob<Dtype>* const blob_top_;;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DynamicConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(DynamicConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->set_num_output(3);

   //fill kernels
  FillerParameter filler_param;
  vector<int> kernel_shape;
  kernel_shape.push_back(2); //number
  kernel_shape.push_back(3); //output number of dynamic convolution
  kernel_shape.push_back(3); //input channels / group
  kernel_shape.push_back(3); //kernel_h
  kernel_shape.push_back(3); //kernel_w
  this->blob_bottom_2_->Reshape(kernel_shape);
  filler_param.set_value(1.);
  ConstantFiller<Dtype> kernel_filler(filler_param);
  kernel_filler.Fill(this->blob_bottom_2_);
  
  //fill bias
  vector<int> bias_shape;
  bias_shape.push_back(2);  //number
  bias_shape.push_back(3);  //output number of dynamic convolution
  this->blob_bottom_3_->Reshape(bias_shape);
  filler_param.set_value(this->bias_val_);
  ConstantFiller<Dtype> bias_filler(filler_param);
  bias_filler.Fill(this->blob_bottom_3_);

  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_3_);
  
  shared_ptr<Layer<Dtype> > layer(
      new DynamicConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);

  // setting group should not change the shape
  convolution_param->set_group(3);
  vector<int> kernel_shape2;
  kernel_shape2.push_back(2); //number
  kernel_shape2.push_back(3); //output number of dynamic convolution
  kernel_shape2.push_back(1); //input channels / group
  kernel_shape2.push_back(3); //kernel_h
  kernel_shape2.push_back(3); //kernel_w
  this->blob_bottom_2_->Reshape(kernel_shape2);
  kernel_filler.Fill(this->blob_bottom_2_);
  
  layer.reset(new DynamicConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(DynamicConvolutionLayerTest, TestSimpleDynamicConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  
  //fill kernels
  FillerParameter filler_param;
  vector<int> kernel_shape;
  kernel_shape.push_back(2); //number
  kernel_shape.push_back(3); //output number of dynamic convolution
  kernel_shape.push_back(1); //input channels / group
  kernel_shape.push_back(3); //kernel_h
  kernel_shape.push_back(3); //kernel_w
  this->blob_bottom_2_->Reshape(kernel_shape);
  filler_param.set_value(1.);
  ConstantFiller<Dtype> kernel_filler(filler_param);
  kernel_filler.Fill(this->blob_bottom_2_);

  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
	
  shared_ptr<Layer<Dtype> > layer(
      new DynamicConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data = this->blob_top_->cpu_data();
  for(int i = 0; i < this->blob_top_->shape(0); ++i){
    int num_off = i * this->blob_top_->count(1);
    for (int j = 0; j < this->blob_top_->count(1); ++j) {
      EXPECT_NEAR(top_data[num_off + j], (j + 1) * 3 * 3, 1e-4);
    }
  }
}

TYPED_TEST(DynamicConvolutionLayerTest, TestSimpleBiasDynamicConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  
  //fill kernels
  FillerParameter filler_param;
  vector<int> kernel_shape;
  kernel_shape.push_back(2); //number
  kernel_shape.push_back(3); //output number of dynamic convolution
  kernel_shape.push_back(1); //input channels / group
  kernel_shape.push_back(3); //kernel_h
  kernel_shape.push_back(3); //kernel_w
  this->blob_bottom_2_->Reshape(kernel_shape);
  filler_param.set_value(1.);
  ConstantFiller<Dtype> kernel_filler(filler_param);
  kernel_filler.Fill(this->blob_bottom_2_);
  
  //fill bias
  vector<int> bias_shape;
  bias_shape.push_back(2);  //number
  bias_shape.push_back(3);  //output number of dynamic convolution
  this->blob_bottom_3_->Reshape(bias_shape);
  LOG(ERROR) << "bias_val" << this->bias_val_;
  filler_param.set_value(this->bias_val_);
  ConstantFiller<Dtype> bias_filler(filler_param);
  bias_filler.Fill(this->blob_bottom_3_);

  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_3_);
  shared_ptr<Layer<Dtype> > layer(
      new DynamicConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data = this->blob_top_->cpu_data();
  for(int i = 0; i < this->blob_top_->shape(0); ++i){
    int num_off = i * this->blob_top_->count(1);
    for (int j = 0; j < this->blob_top_->count(1); ++j) {
      EXPECT_NEAR(top_data[num_off + j], (j + 1) * 3 * 3 + this->bias_val_, 1e-4);
    }
  }

}

TYPED_TEST(DynamicConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->set_num_output(3);
  
  //fill kernels
  FillerParameter filler_param;
  vector<int> kernel_shape;
  kernel_shape.push_back(2); //number
  kernel_shape.push_back(3); //output number of dynamic convolution
  kernel_shape.push_back(3); //input channels / group
  kernel_shape.push_back(3); //kernel_h
  kernel_shape.push_back(3); //kernel_w
  this->blob_bottom_2_->Reshape(kernel_shape);
  filler_param.set_value(1.);
  GaussianFiller<Dtype> kernel_filler(filler_param);
  kernel_filler.Fill(this->blob_bottom_2_);
  
  //fill bias
  vector<int> bias_shape;
  bias_shape.push_back(2);  //number
  bias_shape.push_back(3);  //output number of dynamic convolution
  this->blob_bottom_3_->Reshape(bias_shape);
  filler_param.set_value(this->bias_val_);
  GaussianFiller<Dtype> bias_filler(filler_param);
  bias_filler.Fill(this->blob_bottom_3_);

  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_3_);
  
  DynamicConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
  
  convolution_param->set_group(3);  
  //fill kernels
  vector<int> kernel_shape2;
  kernel_shape2.push_back(2); //number
  kernel_shape2.push_back(3); //output number of dynamic convolution
  kernel_shape2.push_back(1); //input channels / group
  kernel_shape2.push_back(3); //kernel_h
  kernel_shape2.push_back(3); //kernel_w
  this->blob_bottom_2_->Reshape(kernel_shape2);
  filler_param.set_value(1.);
  GaussianFiller<Dtype> kernel_filler2(filler_param);
  kernel_filler2.Fill(this->blob_bottom_2_);
  DynamicConvolutionLayer<Dtype> layer2(layer_param);
  checker.CheckGradientExhaustive(&layer2, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
