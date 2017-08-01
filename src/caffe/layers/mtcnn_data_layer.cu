#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/mtcnn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
void MTCNNDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Batch<Dtype> *batch =
      this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
	int batch_size = batch->data_.shape(0);
	const Dtype* label_data = batch->label_.mutable_cpu_data();
	if(this->phase_ == TRAIN){
		if (this->output_labels_) {
			const Dtype* one_label_data = label_data;
			Dtype* classification_target_data = top[1]->mutable_cpu_data();
			for(int i = 0; i < batch_size; i++){
				if(0 == one_label_data[0] || 1 == one_label_data[0]){
					classification_target_data[i] = one_label_data[0];
				}
				else if(-2 == one_label_data[i]){// indicate landmark data
					classification_target_data[i] = 1.0f;
				}
				else{
					classification_target_data[i] = -1.0f;
				}
				one_label_data += this->total_label_size_;
			}
		}
		if (this->output_roi_) {
			const Dtype* one_label_data = label_data;
			Dtype* roi_target_data = top[2]->mutable_cpu_data();
			int roi_target_size = top[2]->shape(1);
			for(int i = 0; i < batch_size; i++){
				if(1 == one_label_data[0]){
					for(int r = 0; r < roi_target_size; r++){
						roi_target_data[r] = one_label_data[r + 1];
					}
				}
				else{
					for(int r = 0; r < roi_target_size; r++){
						roi_target_data[r] = -1.0f;
					}
				}
				one_label_data += this->total_label_size_;
				roi_target_data += roi_target_size;
			}
		}
		if (this->output_pts_) {
			const Dtype* one_label_data = label_data;
			Dtype* pts_target_data = top[3]->mutable_cpu_data();
			int pts_target_size = top[3]->shape(1);
			for(int i = 0; i < batch_size; i++){
				if(-2 == one_label_data[0]){
					for(int r = 0; r < pts_target_size; r++){
						pts_target_data[r] = one_label_data[r + 1];
					}
				}
				else{
					for(int r = 0; r < pts_target_size; r++){
						pts_target_data[r] = -1.0f;
					}
				}
				label_data += this->total_label_size_;
				pts_target_data += pts_target_size;
			}
		}
	}
	else if(this->phase_ == TEST){
		caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
							 top[1]->mutable_cpu_data());
	}
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}

template <typename Dtype>
void MTCNNDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
INSTANTIATE_LAYER_GPU_FUNCS(MTCNNDataLayer);

}  // namespace caffe
