#ifndef CAFFE_MULTITASK_DATA_LAYER_HPP_
#define CAFFE_MULTITASK_DATA_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class MultiTaskDataLayer: public BaseDataLayer<Dtype>, public InternalThread{
 public:
     explicit MultiTaskDataLayer(const LayerParameter & param)
     : BaseDataLayer<Dtype>(param){}
	 virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	 virtual inline int MaxTopBlobs() const { return 100;}
	 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		 const vector<Blob<Dtype>*>& top);
	 virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		 const vector<Blob<Dtype>*>& top);
	 virtual ~MultiTaskDataLayer();
 protected:
	 virtual void InternalThreadEntry();
     	 vector<shared_ptr<Blob<Dtype> > > prefetch_labels_;
	 int task_num_;

 private:
	shared_ptr<db::DB> db_;
	shared_ptr<db::Cursor> cursor_;
	bool shuffle_on_init;
	
	//still use these three variables which were in old version of class BasePrefetchingDataLayer
	Blob<Dtype> prefetch_data_;
  	Blob<Dtype> prefetch_label_;
  	Blob<Dtype> transformed_data_;
};

} // namespace caffe

#endif //CAFFE_MULTITASK_DATA_LAYER_HPP_
