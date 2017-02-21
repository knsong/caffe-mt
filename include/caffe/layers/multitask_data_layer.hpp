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
class MultiTaskDataLayer: public BasePrefetchingDataLayer<Dtype>{
 public:
     explicit MultiTaskDataLayer(const LayerParameter & param)
       : BasePrefetchingDataLayer<Dtype>(param), data_reader_(param)/*, data_augmentor_(param)*/{}
	 virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	 virtual inline int MaxTopBlobs() const { return 100;} //Max number of tasks
	 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		 const vector<Blob<Dtype>*>& top);
     virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
     ~MultiTaskDataLayer(){};

 private:
    //DataAugmentor data_augmentor_;

    virtual void load_batch(Batch<Dtype>* batch);
    DataReader data_reader_;
    //information about the tasks: number of tasks, label dimension of each task
    int task_num_, actual_label_top_num_;
    std::vector<int> label_dimensions_;
    int label_count_;
};

} // namespace caffe

#endif //CAFFE_MULTITASK_DATA_LAYER_HPP_
