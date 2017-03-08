#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/multitask_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void caffe_static_cast_copy(int N, const float*src, Dtype*dst){
  for (int i = 0; i < N; ++i){
    *dst++ = static_cast<Dtype>(*src++);
  }
}
template <typename Dtype>
void MultiTaskDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(data_reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);

  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();
  // label
  task_num_ = this->layer_param_.data_param().task_num();
  CHECK_EQ(task_num_, this->layer_param_.data_param().label_dimension_size()) <<
    "number of tasks must equal to number of label_dimension!";
  label_dimensions_.clear();
  for (int i = 0; i < task_num_; ++i){
    label_dimensions_.push_back(this->layer_param_.data_param().label_dimension(i));
  }
  if (this->output_labels_) {
    actual_label_top_num_ = top.size() - 1;
    CHECK_LE(actual_label_top_num_, task_num_) << "Number of top blobs containing labels in" << this->layer_param_.name()
                                      << " must be smaller than task numbers!";
    label_count_ = 0;
    int total_label_length = 0;
    for (int i = 0; i < actual_label_top_num_; ++i){
      vector<int> label_shape;
      label_shape.push_back(batch_size);
      label_shape.push_back(label_dimensions_[i]);
      top[i + 1]->Reshape(label_shape);
      label_count_ += label_dimensions_[i];
      LOG(INFO) << "output label shape: " << top[i + 1]->shape_string();
    }
    vector<int> prefetch_label_shape;
    prefetch_label_shape.push_back(task_num_);
    prefetch_label_shape.push_back(label_count_ * batch_size);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(prefetch_label_shape);
    }
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void MultiTaskDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  Batch<Dtype>* batch = this->prefetch_full_.pop("Waiting for batch...");
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
    top[0]->mutable_cpu_data());
  // Copy the labels
  if (this->output_labels_) {
    const Dtype* label_data = batch->label_.cpu_data();
    int batchsize = top[0]->shape(0);
    int label_offset = 0;
    int label_count = 0;
    for (int i = 0; i < this->actual_label_top_num_; i++){
    //  LOG(INFO) << "top label shape: " << top[i + 1]->shape_string();
      label_count = this->label_dimensions_[i] * batchsize;
      caffe_copy(label_count, label_data + label_offset,
                top[i + 1]->mutable_cpu_data());
      label_offset += label_count;
	}
  }
  this->prefetch_free_.push(batch);
}

template <typename Dtype>
void MultiTaskDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  bool do_augumentation = this->layer_param_.has_augumentation_param();

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(data_reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  Datum augumentation;
  Datum* target;
  if (do_augumentation){
    augumentation.set_width(datum.width());
    augumentation.set_height(datum.height());
    augumentation.set_channels(datum.channels());
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(data_reader_.full().pop("Waiting for data"));
 
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (do_augumentation && this->data_augmentor_.Rand(2)){
      this->data_augmentor_.Augumentation(datum, augumentation);
    //  LOG(ERROR) << "transformed data shape:" << this->transformed_data_.shape_string();
      this->data_transformer_->Transform(augumentation, &(this->transformed_data_));
      target = &augumentation;
    }
    else{
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
      target = &datum;
    }

    // Copy label.
    if (this->output_labels_) {
      // Store batch_size labels for one task consecutively 
      int task_offset = 0;
      int label_offset = 0;
      //copy labels for the first task
      //LOG(ERROR) << "label dimension 0 " << " : " << label_dimensions_[0];
      caffe_static_cast_copy(label_dimensions_[0], target->float_data().data(),
                            top_label + item_id * label_dimensions_[0]);
      //copy labels for the left tasks 
      for (int i = 1; i < actual_label_top_num_; ++i){
      //  LOG(ERROR) << "label dimension " << i << " : " << label_dimensions_[i - 1];
        task_offset += label_dimensions_[i - 1] * batch_size;
        label_offset += label_dimensions_[i - 1];
        caffe_static_cast_copy(label_dimensions_[i], target->float_data().data() + label_offset,
                              top_label + task_offset + item_id * label_dimensions_[i]);
      }
    }
    trans_time += timer.MicroSeconds();

    data_reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

#ifdef CPU_ONLY
   STUB_GPU_FORWARD(MultiTaskDataLayer, Forward);
#endif

INSTANTIATE_CLASS(MultiTaskDataLayer);
REGISTER_LAYER_CLASS(MultiTaskData);

}  // namespace caffe
