#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif // USE_OPENCV
#include <stdint.h>

#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
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
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
template <typename Dtype>
void inline caffe_static_cast_copy(int N, const float*src, Dtype*dst){
  for (int i = 0; i < N; ++i){
    *dst++ = static_cast<Dtype>(*src++);
  }
}
template <typename T>
TaskSpecificQueuePair<T>::TaskSpecificQueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new T());
  }
}
template <typename T>
TaskSpecificQueuePair<T>::~TaskSpecificQueuePair() {
  T *datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}
template <typename T>
TaskDataSorter<T>::TaskDataSorter(DataReader<T>& data_reader,
								shared_ptr<TaskSpecificQueuePair<T> > positive_data,
								shared_ptr<TaskSpecificQueuePair<T> > negative_data,
								shared_ptr<TaskSpecificQueuePair<T> > landmark_regression_data,
								shared_ptr<TaskSpecificQueuePair<T> > hard_positive_data)
    : data_reader_(data_reader), positive_qp_(positive_data),
      negative_qp_(negative_data),
      landmark_regression_qp_(landmark_regression_data),
      hard_positive_qp_(hard_positive_data) {
  StartInternalThread();
}
/*
template<typename T>
TaskDataSorter<T>::~TaskDataSorter()
    StopInternalThread();
}
*/
template <typename T>
void TaskDataSorter<T>::InternalThreadEntry() {
  const int roi_label_size = 4;
  try {
    while (!must_stop()) {
		//	LOG(ERROR) << "Sorter working";
//      T &datum = *(data_reader_.full().pop("Sorter Waiting for data"));
      T &datum = *(data_reader_.full().pop());
      int label = static_cast<int>(datum.float_data(0));
      if (1 == label) {
        T *data;
        if (datum.float_data_size() > roi_label_size + 1) {
				//	LOG(ERROR) << "label:" << label << ", label size:" << datum.float_data_size()
				//							<< ", roi qp size:" << landmark_regression_qp_->free_.size();
          if (landmark_regression_qp_->free_.try_pop(&data)) {
            *data = datum; // deep copy
            landmark_regression_qp_->full_.push(data);
          } else{
						data_reader_.free().push(const_cast<Datum*>(&datum));
            continue;
					}
						
        } else {
			//		LOG(ERROR) << "label:" << label << ", pos qp size:" << positive_qp_->free_.size();					
          if (positive_qp_->free_.try_pop(&data)) {
            *data = datum; // deep copy
            positive_qp_->full_.push(data);
          } else{
						data_reader_.free().push(const_cast<Datum*>(&datum));
            continue;
					}
        }
      } else if (0 == label) {
			//	LOG(ERROR) << "label:" << label << ", neg qp size:" << negative_qp_->free_.size();									
        T *data;
        if (negative_qp_->free_.try_pop(&data)) {
          *data = datum; // deep copy
          negative_qp_->full_.push(data);
        } else{
						data_reader_.free().push(const_cast<Datum*>(&datum));
            continue;
				}
      } else if (-1 == label) {
		//		LOG(ERROR) << "label:" << label << ", part qp size:" << hard_positive_qp_->free_.size();													
        T *data;
        if (hard_positive_qp_->free_.try_pop(&data)) {
          *data = datum; // deep copy
          hard_positive_qp_->full_.push(data);
        } else{
					data_reader_.free().push(const_cast<Datum*>(&datum));
          continue;
				}
      }
			else{
				LOG(FATAL) << "Unsupported label!";
			}
			data_reader_.free().push(const_cast<Datum*>(&datum));
    }
  } catch (boost::thread_interrupted &) {
    // Interrupted exception is expected on shutdown
  }
}
template <typename Dtype>
MTCNNDataLayer<Dtype>::MTCNNDataLayer(const LayerParameter &param)
    : BasePrefetchingDataLayer<Dtype>(param), reader_(param) {
  this->phase_ = param.phase();
  if (TRAIN == this->phase_) {
		int batch_size = static_cast<int>(param.data_param().batch_size());
		shared_ptr<TaskSpecificQueuePair<Datum> > pqp(new TaskSpecificQueuePair<Datum>(batch_size));
		positive_qp_ = pqp;
		shared_ptr<TaskSpecificQueuePair<Datum> > nqp(new TaskSpecificQueuePair<Datum>(batch_size));
		negative_qp_ = nqp;
		shared_ptr<TaskSpecificQueuePair<Datum> > lqp(new TaskSpecificQueuePair<Datum>(batch_size));		
		landmark_regression_qp_ = lqp;
		shared_ptr<TaskSpecificQueuePair<Datum> > hqp(new TaskSpecificQueuePair<Datum>(batch_size));			
		hard_positive_qp_ = hqp;
    shared_ptr<TaskDataSorter<Datum> > ds(new TaskDataSorter<Datum>(reader_, positive_qp_,
																negative_qp_, landmark_regression_qp_,
																hard_positive_qp_));
		sorter_ = ds;																
	}
}
/*
    template <typename Dtype>
    MTCNNDataLayer<Dtype>::~MTCNNDataLayer() {
            this->StopInternalThread();
    }
*/
template <typename Dtype>
void MTCNNDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum &datum = *(reader_.full().peek());
  this->output_labels_ = top.size() > 1;
  this->output_roi_ = top.size() > 2;
  this->output_pts_ = top.size() > 3;
  this->tasks_num_ = top.size() - 1;
	this->landmarks_num_ = 5;
	CHECK_LT(this->tasks_num_, 4);
  // data, label, roi, pts
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;

  // data
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();

	if(this->phase_ == TRAIN){
		CHECK_GT(this->tasks_num_, 1);
  	this->total_label_size_ = 0;
	  // label, note in fact `output_labels_` should always be true
	  if (this->output_labels_) {
	    vector<int> label_shape(2);
	    label_shape[0] = batch_size;
	    label_shape[1] = 1;
	    top[1]->Reshape(label_shape);
	    this->total_label_size_ += 1;
	  }
	  // roi
	  if (this->output_roi_) {
	    vector<int> roi_shape(2);
	    roi_shape[0] = batch_size;
	    roi_shape[1] = 4;
	    top[2]->Reshape(roi_shape);
	    this->total_label_size_ += 4;
	  }
	  // pts
	  if (this->output_pts_) {
	    vector<int> pts_shape(2);
	    pts_shape[0] = batch_size;
	    pts_shape[1] = this->landmarks_num_ * 2; // 5 landmark points
	    top[3]->Reshape(pts_shape);
	    this->total_label_size_ += pts_shape[1];
	  }

	  vector<int> batch_label_shape;
	  batch_label_shape.push_back(batch_size);
	  batch_label_shape.push_back(this->total_label_size_);

	  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	    this->prefetch_[i].label_.Reshape(batch_label_shape);
	  }
	  const unsigned int rng_seed = caffe_rng_rand();
	  rng_.reset(new Caffe::RNG(rng_seed));
	}
	else if(this->phase_ == TEST){
		CHECK_EQ(this->tasks_num_, 1) << "Must use 1 task during training!";
		this->total_label_size_ = 0;
		// label, note in fact `output_labels_` should always be true
		if (this->output_labels_) {
			vector<int> label_shape(2);
			label_shape[0] = batch_size;
			label_shape[1] = 1;
			top[1]->Reshape(label_shape);
			this->total_label_size_ = 1;
		}
		// roi
		if (this->output_roi_) {
			vector<int> roi_shape(2);
			roi_shape[0] = batch_size;
			roi_shape[1] = 4;
			top[1]->Reshape(roi_shape);
			this->total_label_size_ = 4;
		}
		// pts
		if (this->output_pts_) {
			vector<int> pts_shape(2);
			pts_shape[0] = batch_size;
			pts_shape[1] = this->landmarks_num_ * 2; // 5 landmark points
			top[1]->Reshape(pts_shape);
			this->total_label_size_ = pts_shape[1];
		}

		vector<int> batch_label_shape;
		batch_label_shape.push_back(batch_size);
		batch_label_shape.push_back(this->total_label_size_);

		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].label_.Reshape(batch_label_shape);
		}
	}
}

template <typename Dtype>
void MTCNNDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                                        const vector<Blob<Dtype>* >& top) {
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
				else{// indicate face part data
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
  this->prefetch_free_.push(batch);
}
template <typename Dtype>
int MTCNNDataLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t *rng = static_cast<caffe::rng_t *>(rng_->generator());
  return ((*rng)() % n);
}
// This function is called on prefetch thread
template <typename Dtype>
void MTCNNDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum &datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype *batch_data = batch->data_.mutable_cpu_data();
	Dtype* batch_label_data = batch->label_.mutable_cpu_data();

	
  switch (this->phase_) {
  case TRAIN:
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      timer.Start();
			bool success = false;
			int offset = batch->data_.offset(item_id);
			this->transformed_data_.set_cpu_data(batch_data + offset);
			
			while(!success){
				int task_id = Rand(4); // 4 kinds of data labels
				int label_size;
				Datum* datum;
				switch (task_id) {
				case 0:{
				//	Datum &datum = *(negative_qp_->full_.pop("Waiting for negative data"));
					if(negative_qp_->full_.try_pop(&datum)){
						read_time += timer.MicroSeconds();
						timer.Start();
						label_size = datum->float_data_size();
						this->data_transformer_->Transform(*datum, &(this->transformed_data_));
						caffe_static_cast_copy(label_size, datum->float_data().data(),
								batch_label_data);
						trans_time += timer.MicroSeconds();
						negative_qp_->free_.push(datum);
						success = true;
					}
					else{
						success = false;
					}
					break;
				}
				case 1:{
				//	Datum &datum = *(positive_qp_->full_.pop("Waiting for positive data"));
					if(positive_qp_->full_.try_pop(&datum)){
						read_time += timer.MicroSeconds();
						timer.Start();
						label_size = datum->float_data_size();
						this->data_transformer_->Transform(*datum, &(this->transformed_data_));
						caffe_static_cast_copy(label_size, datum->float_data().data(),
								batch_label_data);
						trans_time += timer.MicroSeconds();
						positive_qp_->free_.push(datum);
						success = true;		
					}
					else{
						success = false;		
					}
					break;
				}
				case 2:{
			//		Datum &datum = *(hard_positive_qp_->full_.pop("Waiting for part data"));
					if(hard_positive_qp_->full_.try_pop(&datum)){
						read_time += timer.MicroSeconds();
						timer.Start();
						label_size = datum->float_data_size();
						this->data_transformer_->Transform(*datum, &(this->transformed_data_));
						caffe_static_cast_copy(label_size, datum->float_data().data(),
								batch_label_data);
						trans_time += timer.MicroSeconds();
						hard_positive_qp_->free_.push(datum);
						success = true;	
					}
					else{
						success = false;	
					}
					break;
				}
				case 3:{
				//	Datum& datum = *(landmark_regression_qp_->full_.pop("Waiting for landmark data"));
					if(landmark_regression_qp_->full_.try_pop(&datum)){
						read_time += timer.MicroSeconds();
						timer.Start();
						label_size = datum->float_data_size();
						this->data_transformer_->Transform(*datum, &(this->transformed_data_));
						batch_label_data[0] = -2.0f;
						caffe_static_cast_copy(label_size - 1, datum->float_data().data() + 1,
								batch_label_data + 1);
						trans_time += timer.MicroSeconds();
						landmark_regression_qp_->free_.push(datum);
						success = true;		
					}
					else{
						success = false;		
					}
					break;
				}
				default:
					LOG(FATAL) << "Task id error!";
				}
			}
      
			batch_label_data += this->total_label_size_;
    }
    break;
  case TEST: // During TEST, we only consider there is only 1 task!
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      timer.Start();
      // get a datum
      Datum &datum = *(reader_.full().pop("Waiting for data"));
      read_time += timer.MicroSeconds();
      timer.Start();
      int label_size = datum.float_data_size();
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(batch_data + offset);
      // Apply data transformations (mirror, scale, crop...)
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
			caffe_static_cast_copy(label_size, datum.float_data().data(),
					batch->label_.mutable_cpu_data() + item_id * this->total_label_size_);
      trans_time += timer.MicroSeconds();
      reader_.free().push(&datum);
    }
    break;
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}
#ifdef CPU_ONLY
STUB_GPU(MTCNNDataLayer);
#endif

template class TaskDataSorter<Datum>;
template class TaskSpecificQueuePair<Datum>;
INSTANTIATE_CLASS(MTCNNDataLayer);
REGISTER_LAYER_CLASS(MTCNNData);
} // namespace caffe
