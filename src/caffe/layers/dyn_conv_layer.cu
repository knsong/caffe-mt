#include <vector>

#include "caffe/layers/dyn_conv_layer.hpp"

namespace caffe {
/*
template <typename Dtype>
void DynamicConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* weight = bottom[1]->gpu_data();
	int weight_size = bottom[1]->count(1);
	
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	for (int n = 0; n < this->num_; ++n) {
		this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight + n * weight_size,
			top_data + n * this->top_dim_);	
		if (this->bias_term_) {
			const Dtype* bias = bottom[2]->gpu_data();
			this->forward_gpu_bias(top_data + n * this->top_dim_, bias + n * this->num_output_);
		}
	}

}
*/
template <typename Dtype>
void DynamicConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* weight = bottom[1]->gpu_data();
	int weight_size = bottom[1]->count(1);

	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	for (int n = 0; n < this->num_; ++n) {
		this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight + n * weight_size,
            top_data + n * this->top_dim_);
		if (this->bias_term_) {
			const Dtype* bias = bottom[2]->gpu_data();
			this->forward_gpu_bias(top_data + n * this->top_dim_, bias + n * this->num_output_);
		}
	}
}

template <typename Dtype>
void DynamicConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* weight = bottom[1]->gpu_data();
	Dtype* weight_diff = bottom[1]->mutable_gpu_diff();
	int weight_size = bottom[1]->count(1);

	const Dtype* top_diff = top[0]->gpu_diff();
	
	if (this->param_propagate_down_[0]) {
		caffe_gpu_set(bottom[1]->count(), Dtype(0), weight_diff);
	}
	if (this->bias_term_ && this->param_propagate_down_[1]) {
		caffe_gpu_set(bottom[2]->count(), Dtype(0),
			bottom[2]->mutable_gpu_diff());
    }
	// Bias gradient, if necessary.
	if (this->bias_term_ && this->param_propagate_down_[1]) {
		Dtype* bias_diff = bottom[2]->mutable_gpu_diff();
		for (int n = 0; n < this->num_; ++n) {
			this->backward_gpu_bias(bias_diff + n *  this->num_output_, top_diff + n * this->top_dim_);
		}
	}
	if (this->param_propagate_down_[0] || propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		for (int n = 0; n < this->num_; ++n) {
		// gradient w.r.t. weight. Note that we will accumulate diffs.
		if (this->param_propagate_down_[0]) {
			this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
				top_diff + n * this->top_dim_, weight_diff + n * weight_size);
		}
		// gradient w.r.t. bottom data, if necessary.
		if (propagate_down[0]) {
			this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight + n * weight_size,
				bottom_diff + n * this->bottom_dim_);
		}
		}
	}
 
}

INSTANTIATE_LAYER_GPU_FUNCS(DynamicConvolutionLayer);

}  // namespace caffe
