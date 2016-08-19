#include <vector>

#include "caffe/layers/multitask_data_layer.hpp"

namespace caffe{

template <typename Dtype>
void MultiTaskDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	CHECK_EQ(top.size(), task_num_ + 1);
	this->StopInternalThread();
	// Copy the data
	caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
		top[0]->mutable_gpu_data());
	if (this->output_labels_) {
        for (int i = 0; i < task_num_; i++)
        {
            int label_count = prefetch_labels_[i]->count() ;
            const Dtype *pPrefetchLabel = prefetch_labels_[i]->cpu_data();
            caffe_copy(label_count, pPrefetchLabel,
                       top[i + 1]->mutable_gpu_data());
//			LOG(INFO) << "task:" << i << "label: " << pPrefetchLabel[0];
		}
	}
	// Start a new prefetch thread
	this->data_transformer_->InitRand();
	this->StartInternalThread();
}
INSTANTIATE_LAYER_GPU_FORWARD(MultiTaskDataLayer);

}//namespace caffe
