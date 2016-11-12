#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/local_conv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

//#define SHOW_LOG
//
namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv_ref(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
	const vector<shared_ptr<Blob<Dtype> > >& weights,
	Blob<Dtype>* out) {
	// Kernel size, stride, and pad
	int kernel_h, kernel_w;
    //only for 2D convolution
	if (conv_param->kernel_size_size() == 1) {
		kernel_h = kernel_w = conv_param->kernel_size(0);
	}
	else {
		kernel_h = conv_param->kernel_h();
		kernel_w = conv_param->kernel_w();
	}
	int pad_h, pad_w;
	if (!conv_param->has_pad_h()) {
		pad_h = pad_w = conv_param->pad(0);
	}
	else {
		pad_h = conv_param->pad_h();
		pad_w = conv_param->pad_w();
	}
	int stride_h, stride_w;
	if (!conv_param->has_stride_h()) {
		stride_h = stride_w = conv_param->stride(0);
	}
	else {
		stride_h = conv_param->stride_h();
		stride_w = conv_param->stride_w();
	}
	// Groups
	int groups = conv_param->group();
	int o_g = out->channels() / groups;
	int k_g = in->channels() / groups;
	//LOG(INFO) << "groups: " << groups << "k_h/w: " << kernel_h << " " << kernel_w << "p_h/w: " << pad_h << " " << pad_w;
	int o_head, k_head;
	// Convolution
	const Dtype* in_data = in->cpu_data();
	const Dtype* weight_data = weights[0]->cpu_data();
	int w_step = in->channels() * weights[0]->height() * weights[0]->width() / groups;
	int w_spatial_dim = weights[0]->height() * weights[0]->width();
	Dtype* out_data = out->mutable_cpu_data();
	//LOG(INFO) << "caffe conv out shape:" << out->shape_string();
	
	for (int n = 0; n < out->num(); n++) {
		for (int g = 0; g < groups; g++) {
			o_head = o_g * g;
			k_head = k_g * g;
			for (int o = 0; o < o_g; o++) {
				for (int k = 0; k < k_g; k++) {
					for (int y = 0; y < out->height(); y++) {
						for (int x = 0; x < out->width(); x++) {
							for (int p = 0; p < kernel_h; p++) {
								for (int q = 0; q < kernel_w; q++) {
									int in_y = y * stride_h - pad_h + p;
									int in_x = x * stride_w - pad_w + q;
									int w_idx = (o + o_head)* w_step +  w_spatial_dim * k + p * kernel_w + q;
									if (in_y >= 0 && in_y < in->height()
										&& in_x >= 0 && in_x < in->width()) {
										out_data[out->offset(n, o + o_head, y, x)] +=
											in_data[in->offset(n, k + k_head, in_y, in_x)]
											* weight_data[w_idx];
									}
								}
							}
						}
					}
				}
			}
		}
	}
	// Bias
	if (conv_param->bias_term()) {
		const Dtype* bias_data = weights[1]->cpu_data();
		for (int n = 0; n < out->num(); n++) {
			for (int o = 0; o < out->channels(); o++) {
				for (int y = 0; y < out->height(); y++) {
					for (int x = 0; x < out->width(); x++) {
						out_data[out->offset(n, o, y, x)] += bias_data[o];
					}
				}
			}
		}
	}
}

template<typename Dtype>
void init_local_offset(Blob<Dtype> *idx_to_off_blob, int height, int width, int local_height, int local_width
							, int local_region_num_h, int local_region_num_w, int local_region_step_h, int local_region_step_w){
	Blob<Dtype> *idx_to_off = idx_to_off_blob;
	Dtype *idx_to_off_data = idx_to_off->mutable_cpu_data();

	int h, w, offset_h, offset_w, symmetry_offset_h, symmetry_offset_w;
	for (h = 0; h < local_region_num_h / 2; ++h){
		offset_h = h * local_region_step_h;
		symmetry_offset_h = height - (offset_h + local_height);
		for (w = 0; w < local_region_num_w / 2; ++w){
			offset_w = w * local_region_step_w;
			symmetry_offset_w = width - (offset_w + local_width);
			(idx_to_off_data + idx_to_off->offset(h, w))[0] = offset_h;
			(idx_to_off_data + idx_to_off->offset(h, w))[1] = offset_w;

			(idx_to_off_data + idx_to_off->offset(h, local_region_num_w - 1 - w))[0] = offset_h;
			(idx_to_off_data + idx_to_off->offset(h, local_region_num_w - 1 - w))[1] = symmetry_offset_w;

			(idx_to_off_data + idx_to_off->offset(local_region_num_h - 1 - h, w))[0] = symmetry_offset_h;
			(idx_to_off_data + idx_to_off->offset(local_region_num_h - 1 - h, w))[1] = offset_w;

			(idx_to_off_data + idx_to_off->offset(local_region_num_h - 1 - h, local_region_num_w - 1 - w))[0]
				= symmetry_offset_h;
			(idx_to_off_data + idx_to_off->offset(local_region_num_h - 1 - h, local_region_num_w - 1 - w))[1]
				= symmetry_offset_w;

		}
		if (local_region_num_w % 2){
			offset_w = (width - local_width) / 2;

			(idx_to_off_data + idx_to_off->offset(h, w))[0] = offset_h;
			(idx_to_off_data + idx_to_off->offset(h, w))[1] = offset_w;

			(idx_to_off_data + idx_to_off->offset(local_region_num_h - 1 - h, w))[0] = symmetry_offset_h;
			(idx_to_off_data + idx_to_off->offset(local_region_num_h - 1 - h, w))[1] = offset_w;
		}
	}
	if (local_region_num_h % 2){
		offset_h = (height - local_height) / 2;
		for (w = 0; w < local_region_num_w / 2; ++w){
			offset_w = w * local_region_step_w;
			symmetry_offset_w = width - (offset_w + local_width);

			(idx_to_off_data + idx_to_off->offset(h, w))[0] = offset_h;
			(idx_to_off_data + idx_to_off->offset(h, w))[1] = offset_w;

			(idx_to_off_data + idx_to_off->offset(h, local_region_num_w - 1 - w))[0] = offset_h;
			(idx_to_off_data + idx_to_off->offset(h, local_region_num_w - 1 - w))[1] = symmetry_offset_w;

		}
		if (local_region_num_w % 2){
			offset_w = (width - local_width) / 2;
			(idx_to_off_data + idx_to_off->offset(h, w))[0] = offset_h;
			(idx_to_off_data + idx_to_off->offset(h, w))[1] = offset_w;
		}
	}
}

template<typename Dtype>
void crop_loc_patch_cpu(const Dtype *src, int src_w, int src_h, int src_c, int crop_width, int crop_height, int w_off, int h_off, Dtype *local_patch_data)
{
	for (int c = 0; c < src_c; ++c){
		for (int h = 0; h < crop_height; ++h){
			for (int w = 0; w < crop_width; ++w){
				local_patch_data[(c * crop_height + h) * crop_width + w]
					= src[(c * src_h + (h + h_off)) * src_w + w + w_off];
			}
		}
	}
}

template <typename Dtype>
void realign_loc_conv_result_cpu(const vector<shared_ptr<Blob<Dtype> > > &local_conv_res, int loc_num_h, int loc_num_w
	, int top_height, int top_width, Dtype *dst_data)
{
//	CHECK_EQ(local_conv_res[0]->channels(), dst_blob->channels()) << " channels mismatch!";

	int width_out_ = local_conv_res[0]->width();
	int height_out_ = local_conv_res[0]->height();
	int num_output_ = local_conv_res[0]->channels();
	int top_channel_step = top_height * top_width;
	int mStep = width_out_ * loc_num_w;
//	int loc_conv_res_size = num_output_ * height_out_ * width_out_;
//	Dtype *dst_data = dst_blob->mutable_cpu_data();
	for (int n = 0; n < num_output_; ++n){
		int num_offset = n * height_out_ * width_out_;
		for (int h = 0; h < top_height; ++h){
			for (int w = 0; w < top_width; ++w){
				int dst_offset = h * top_width + w;
				int loc_w = dst_offset % mStep % width_out_;
				int loc_idx_w = dst_offset % mStep / width_out_;
				int loc_h = dst_offset / mStep % height_out_;
				int loc_idx_h = dst_offset / mStep / height_out_;

				const Dtype *loc_conv_data = local_conv_res[loc_idx_h * loc_num_w + loc_idx_w]->cpu_data();
				int src_idx = num_offset + loc_h * width_out_ + loc_w;
				int dst_idx = dst_offset + n * top_channel_step;
			//	int loc_data_offset = num_offset + loc_h * width_out_ + loc_w;
			//	int loc_offset = (loc_idx_h * loc_num_w + loc_idx_w) * loc_conv_res_size;
				dst_data[dst_idx] = loc_conv_data[src_idx];
			}
		}
	}
}
template <typename Dtype>
void caffe_loc_conv(const Blob<Dtype>* in, LocalConvolutionParameter* loc_conv_param,
	const vector<shared_ptr<Blob<Dtype> > >& weights, Blob<Dtype>* out) {
	ConvolutionParameter conv_param;
	// Kernel size, stride, and pad
	int kernel_h, kernel_w;
	if (loc_conv_param->has_kernel_size()) {
		kernel_h = kernel_w = loc_conv_param->kernel_size();
	}
	else {
		kernel_h = loc_conv_param->kernel_h();
		kernel_w = loc_conv_param->kernel_w();
	}
	conv_param.set_kernel_h(kernel_h);
	conv_param.set_kernel_w(kernel_w);
	int pad_h, pad_w;
	if (!loc_conv_param->has_pad_h()) {
		pad_h = pad_w = loc_conv_param->pad();
	}
	else {
		pad_h = loc_conv_param->pad_h();
		pad_w = loc_conv_param->pad_w();
	}
	conv_param.set_pad_h(pad_h);
	conv_param.set_pad_w(pad_w);
	int stride_h, stride_w;
	if (!loc_conv_param->has_stride_h()) {
		stride_h = stride_w = loc_conv_param->stride();
	}
	else {
		stride_h = loc_conv_param->stride_h();
		stride_w = loc_conv_param->stride_w();
	}
	conv_param.set_stride_h(stride_h);
	conv_param.set_stride_w(stride_w);
	// Groups
	int groups = loc_conv_param->group();
	conv_param.set_group(groups);
	//Just for caffe_conv()
	conv_param.mutable_bias_filler()->set_type("constant");

	int loc_num_w, loc_num_h;
	if (!loc_conv_param->has_local_region_number_h()) {
		loc_num_w = loc_num_h = loc_conv_param->local_region_number();
	}
	else {
		loc_num_h = loc_conv_param->local_region_number_h();
		loc_num_w = loc_conv_param->local_region_number_w();
	}
	float loc_ratio_w, loc_ratio_h;
	if (loc_conv_param->has_local_region_ratio()){
		loc_ratio_w = loc_ratio_h = loc_conv_param->local_region_ratio();
	}
	else{
		loc_ratio_w = loc_conv_param->local_region_ratio_w();
		loc_ratio_h = loc_conv_param->local_region_ratio_h();
	}
	int loc_step_w, loc_step_h;
	if (loc_conv_param->has_local_region_step()){
		loc_step_w = loc_step_h = loc_conv_param->local_region_step();
	}
	else{
		loc_step_w = loc_conv_param->local_region_step_w();
		loc_step_h = loc_conv_param->local_region_step_h();
	}
	
	int loc_region_w = in->width() * loc_ratio_w, loc_region_h = in->height()*loc_ratio_h;
	int num = in->num(), channels = in->channels(), height = in->height(), width = in->width();
	int conv_out_num = loc_conv_param->num_output();
	int loc_conv_out_h = (loc_region_h + 2 * pad_h - kernel_h) / stride_h + 1;
	int loc_conv_out_w = (loc_region_w + 2 * pad_w - kernel_w) / stride_w + 1;

	const Dtype *in_data = in->cpu_data();
	vector<shared_ptr<Blob<Dtype> > > loc_weights;
	loc_weights.resize(2);
	loc_weights[0].reset(new Blob<Dtype>(1, conv_out_num * channels / groups, kernel_h, kernel_w));
	loc_weights[1].reset(new Blob<Dtype>(1, conv_out_num, 1, 1));

	Blob<Dtype> loc_in(1, channels, loc_region_h, loc_region_w);
	Dtype *loc_in_data = loc_in.mutable_cpu_data();
	vector<shared_ptr<Blob<Dtype> > > loc_out;
	for (int i = 0; i < loc_num_h * loc_num_w; ++i){
		shared_ptr<Blob<Dtype> > loc_ot(new Blob<Dtype>(1, conv_out_num, loc_conv_out_h, loc_conv_out_w));
		loc_out.push_back(loc_ot);
	}

	Dtype *out_data = out->mutable_cpu_data();
	int out_height = out->height(), out_width = out->width();
//	LOG(INFO) << "out shape: " << out->shape_string();
	Blob<Dtype> idx_to_off(loc_num_h, loc_num_w, 2, 1);
	Dtype *idx_to_off_data = idx_to_off.mutable_cpu_data();

	init_local_offset(&idx_to_off, height, width, loc_region_h, loc_region_w, loc_num_h, loc_num_w, loc_step_h, loc_step_w);

//	LOG(INFO) << "weight shape:" << weights[0]->shape_string() << " bias shape:" << weights[1]->shape_string(); 
	Dtype *weights_data = weights[0]->mutable_cpu_data();
	Dtype *bias_data = weights[1]->mutable_cpu_data();
	
    for (int n = 0; n < num; n++){
		const Dtype *single_in_data = in_data + in->offset(n);
		Dtype *single_out_data = out_data + out->offset(n);
		for (int h = 0; h < loc_num_h; h++){
			for (int w = 0; w < loc_num_w; w++){
				int loc_num = h * loc_num_w + w;
				Blob<Dtype> *loc_out_blob = loc_out[loc_num].get();
				Dtype *loc_out_data = loc_out_blob->mutable_cpu_data();
		                memset(loc_out_data, 0, sizeof(Dtype) * loc_out_blob->count());
		                loc_weights[0]->set_cpu_data(weights_data + weights[0]->offset(loc_num));
				loc_weights[1]->set_cpu_data(bias_data + weights[1]->offset(loc_num));
				crop_loc_patch_cpu(single_in_data, width, height, channels
					, loc_region_w, loc_region_h
					, (idx_to_off_data + idx_to_off.offset(h, w))[1]
					, (idx_to_off_data + idx_to_off.offset(h, w))[0], loc_in_data);

				caffe_conv_ref(&loc_in, &conv_param, loc_weights, loc_out_blob);
			}
		}
		realign_loc_conv_result_cpu(loc_out, loc_num_h, loc_num_w, out_height, out_width, single_out_data);
	}
}

template void caffe_conv_ref(const Blob<float>* in,
	ConvolutionParameter* conv_param,
	const vector<shared_ptr<Blob<float> > >& weights,
	Blob<float>* out);
template void caffe_conv_ref(const Blob<double>* in,
	ConvolutionParameter* conv_param,
	const vector<shared_ptr<Blob<double> > >& weights,
	Blob<double>* out);

template void caffe_loc_conv(const Blob<float>* in,
	LocalConvolutionParameter* loc_conv_param,
	const vector<shared_ptr<Blob<float> > >& weights,
	Blob<float>* out);
template void caffe_loc_conv(const Blob<double>* in,
	LocalConvolutionParameter* loc_conv_param,
	const vector<shared_ptr<Blob<double> > >& weights,
	Blob<double>* out);

template <typename TypeParam>
class LocalConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
	 LocalConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 7, 8)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 7, 8)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
	/*
    filler_param.set_type("uniform");
    filler_param.set_min(0);
	filler_param.set_max(2*3*7*8);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    */
    filler_param.set_value(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~LocalConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
};

TYPED_TEST_CASE(LocalConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(LocalConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LocalConvolutionParameter* loc_conv_param = layer_param.mutable_local_conv_param();
  loc_conv_param->set_kernel_size(3);
  loc_conv_param->set_stride(2);
  loc_conv_param->set_num_output(2);
  loc_conv_param->set_local_region_number_w(4);
  loc_conv_param->set_local_region_number_h(3);
  loc_conv_param->set_local_region_ratio_w(0.4);
  loc_conv_param->set_local_region_ratio_h(0.5);
  loc_conv_param->set_local_region_step(2);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new LocalConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 2);
  EXPECT_EQ(this->blob_top_2_->height(), 3);
  EXPECT_EQ(this->blob_top_2_->width(), 4);
  // setting group should not change the shape
  loc_conv_param->set_num_output(3);
  loc_conv_param->set_group(3);
  layer.reset(new LocalConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 3);
  EXPECT_EQ(this->blob_top_2_->width(), 4);
}

TYPED_TEST(LocalConvolutionLayerTest, TestLocalConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  LocalConvolutionParameter* loc_conv_param =
      layer_param.mutable_local_conv_param();
  loc_conv_param->set_kernel_size(3);
  loc_conv_param->set_stride(2);
  loc_conv_param->set_num_output(3);
  loc_conv_param->set_local_region_number_w(4);
  loc_conv_param->set_group(3);
  loc_conv_param->set_local_region_number_h(3);
  loc_conv_param->set_local_region_ratio_w(0.4);
  loc_conv_param->set_local_region_ratio_h(0.5);
  loc_conv_param->set_local_region_step(2);
  loc_conv_param->mutable_weight_filler()->set_type("gaussian");
  loc_conv_param->mutable_bias_filler()->set_type("constant");
  loc_conv_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new LocalConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;

  caffe_loc_conv(this->blob_bottom_, loc_conv_param, layer->blobs()
						, this->MakeReferenceTop(this->blob_top_));

  top_data = this->blob_top_->cpu_data();
 // LOG(INFO) << "top shape:" << this->blob_top_->shape_string();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
     EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4) << "i " << i;
}

//  LOG(INFO) << "caffe_loc_conv 2 begins";
  caffe_loc_conv(this->blob_bottom_2_, loc_conv_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }

}

TYPED_TEST(LocalConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LocalConvolutionParameter* loc_conv_param =
	  layer_param.mutable_local_conv_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  loc_conv_param->set_kernel_size(3);
  loc_conv_param->set_stride(2);
  loc_conv_param->set_num_output(3);
  loc_conv_param->set_local_region_number_w(4);
  loc_conv_param->set_local_region_number_h(3);
  loc_conv_param->set_local_region_ratio_w(0.4);
  loc_conv_param->set_local_region_ratio_h(0.5);
  loc_conv_param->set_local_region_step(2);
  loc_conv_param->mutable_weight_filler()->set_type("gaussian");
  loc_conv_param->mutable_bias_filler()->set_type("gaussian");
  LocalConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
