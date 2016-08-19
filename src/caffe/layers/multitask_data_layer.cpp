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
MultiTaskDataLayer<Dtype>::~MultiTaskDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void MultiTaskDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  this->shuffle_on_init = false;
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());
  LOG(INFO) << "datum: encoded" << datum.encoded();
  LOG(INFO) << "datum: channels" << datum.channels();


  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if ((force_color && DecodeDatum(&datum, true)) ||
      DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }
  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    top[0]->Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
  } else {
    top[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
      datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if  (this->output_labels_) {
	task_num_ = this->layer_param_.data_param().task_num();
	CHECK_EQ(task_num_, this->layer_param_.data_param().label_dimension_size())
						<<" 'label_dimension's' number must be equal to task_num!";
	for (int task_idx = 0; task_idx < task_num_; task_idx++)
	{
		int label_dimension = this->layer_param_.data_param().label_dimension(task_idx);
		CHECK_GT(label_dimension, 0) << "label dimension can not be set to 0";
        shared_ptr<Blob<Dtype> > pLabelBlob(new Blob<Dtype>(this->layer_param_.data_param().batch_size(), label_dimension, 1, 1));
        prefetch_labels_.push_back(pLabelBlob);
		top[task_idx + 1]->Reshape(this->layer_param_.data_param().batch_size(), label_dimension, 1, 1);
	}

 //   vector<int> label_distribution_shape(1, this->layer_param_.data_param().batch_size());
  //  label_distribution_shape.push_back(total_label_count_);
  //  this->prefetch_label_.Reshape(label_distribution_shape);
  }
  LOG(INFO) << "set up done!";
}

template <typename Dtype>
void MultiTaskDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	CHECK_EQ(top.size(), task_num_ + 1);
	this->StopInternalThread();
	// Copy the data
	caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
		top[0]->mutable_cpu_data());
	if (this->output_labels_) {
        for (int i = 0; i < task_num_; i++)
        {
            int label_count = this->prefetch_labels_[i]->count() ;
            const Dtype *pPrefetchLabel = this->prefetch_labels_[i]->cpu_data();
            caffe_copy(label_count, pPrefetchLabel,
                       top[i + 1]->mutable_cpu_data());
		}
	}
	
	// Start a new prefetch thread
	this->data_transformer_->InitRand();
	this->StartInternalThread();
}
// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiTaskDataLayer<Dtype>::InternalThreadEntry() {

  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  CHECK_EQ(task_num_, prefetch_labels_.size())
					<<"task_num_ is not equal to prefetch_labels_.size()!";

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if (batch_size == 1 && crop_size == 0) {
    Datum datum;
    datum.ParseFromString(cursor_->value());
    if (datum.encoded()) {
      if (force_color) {
        DecodeDatum(&datum, true);
      } else {
        DecodeDatumNative(&datum);
      }
    }
    this->prefetch_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
  }
  //LOG(INFO) << "before prefetch data";
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
	if (this->layer_param_.data_param().rand_skip()) {  //skip a random number of images
        unsigned int skip = caffe_rng_rand() % batch_size; //caffe_rng_rand() % dataset_->size();
        //LOG(INFO) << "Random access! at the point " << skip << " data points.";
        while (skip-- > 0) {
			cursor_->Next();
			if (!cursor_->valid()) {
				DLOG(INFO) << "Restarting data prefetching from start.";
				cursor_->SeekToFirst();
			}
        }
	}
	// get a blob
	Datum datum;
	datum.ParseFromString(cursor_->value());

	// Apply data transformations (mirror, scale, crop...)
	int offset = this->prefetch_data_.offset(item_id);
	this->transformed_data_.set_cpu_data(top_data + offset);

	this->data_transformer_->Transform(datum,
		  &(this->transformed_data_));

	if (this->output_labels_) {
        int task_offset = 0;
		for (int i = 0; i < task_num_; i++)
		{
			int label_dimension = prefetch_labels_[i]->channels();
			Dtype *top_label = prefetch_labels_[i]->mutable_cpu_data();
            int num_offset = item_id * label_dimension;

			for (int j = 0; j < label_dimension; j++)
			{
				top_label[num_offset + j] = datum.float_data(task_offset + j);
			//	top_label[num_offset + j] = datum.multiple_label(task_offset + j);
			}
			task_offset += label_dimension;
			//	task_offset += 5;
		}
	}
	// go to the next iter
	cursor_->Next();
	if (!cursor_->valid()) {
	  DLOG(INFO) << "Restarting data prefetching from start.";
	  cursor_->SeekToFirst();
	}
   }

}

INSTANTIATE_CLASS(MultiTaskDataLayer);
REGISTER_LAYER_CLASS(MultiTaskData);

}  // namespace caffe
