#include <vector>

#include "caffe/layers/multitask_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiTaskDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  Batch<Dtype>* batch = this->prefetch_full_.pop("Waiting for batch...");
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
    top[0]->mutable_gpu_data());
  // Copy the labels
  if (this->output_labels_) {
    const Dtype* label_data = batch->label_.gpu_data();
    int batchsize = top[0]->shape(0);
    int label_offset = 0;
    int label_count = 0;
    for (int i = 0; i < this->actual_label_top_num_; i++){
    //  LOG(INFO) << "top label shape: " << top[i + 1]->shape_string();
      label_count = this->label_dimensions_[i] * batchsize;
      caffe_copy(label_count, label_data + label_offset,
                top[i + 1]->mutable_gpu_data());
      label_offset += label_count;
	}
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(MultiTaskDataLayer);

}  // namespace caffe
