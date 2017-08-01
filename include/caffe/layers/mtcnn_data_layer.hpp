#ifndef CAFFE_MTCNN_DATA_LAYER_HPP_
#define CAFFE_MTCNN_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

  template<typename T>
  class TaskSpecificQueuePair {
  public:
      explicit TaskSpecificQueuePair(int size);
      ~TaskSpecificQueuePair();

      BlockingQueue<T*> free_;
      BlockingQueue<T*> full_;

      DISABLE_COPY_AND_ASSIGN(TaskSpecificQueuePair);
  };

  template<typename T>
  class TaskDataSorter: public InternalThread{
  public:
      explicit TaskDataSorter(DataReader<T>& data_reader,
				shared_ptr<TaskSpecificQueuePair<T> > positive_data,
				shared_ptr<TaskSpecificQueuePair<T> > negative_data,
				shared_ptr<TaskSpecificQueuePair<T> > landmark_regression_data,
				shared_ptr<TaskSpecificQueuePair<T> > hard_positive_data);
      virtual void InternalThreadEntry();
  private:
      DataReader<T >& data_reader_;
      shared_ptr<TaskSpecificQueuePair<T > > positive_qp_;
      shared_ptr<TaskSpecificQueuePair<T> > negative_qp_;
      shared_ptr<TaskSpecificQueuePair<T> > landmark_regression_qp_;
      shared_ptr<TaskSpecificQueuePair<T> > hard_positive_qp_; // part positive data
      DISABLE_COPY_AND_ASSIGN(TaskDataSorter);
  };
	template <typename Dtype>
	class MTCNNDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit MTCNNDataLayer(const LayerParameter& param);
//		virtual ~MTCNNDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// DataLayer uses DataReader instead for sharing for parallelism
		virtual inline bool ShareInParallel() const { return false; }
		virtual inline const char* type() const { return "MTCNNData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }

		//data, label, roi, pts
		//virtual inline int ExactNumTopBlobs() const { return 4; }
		virtual inline int MinTopBlobs() const { return 2; }
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	protected:
        int Rand(int n);
		    virtual void load_batch(Batch<Dtype>* batch);
				DataReader<Datum> reader_;
				int total_label_size_;
        int tasks_num_;
				int landmarks_num_;
        shared_ptr<TaskDataSorter<Datum> > sorter_;
        // Buffer queue pairs for balancing samples with different labels
        shared_ptr<TaskSpecificQueuePair<Datum> > positive_qp_;
        shared_ptr<TaskSpecificQueuePair<Datum> > negative_qp_;
        shared_ptr<TaskSpecificQueuePair<Datum> > landmark_regression_qp_;
        shared_ptr<TaskSpecificQueuePair<Datum> > hard_positive_qp_; // part positive data

        int phase_;
				bool output_labels_;
				bool output_roi_;
				bool output_pts_;
        shared_ptr<Caffe::RNG> rng_;
	};

}  // namespace caffe

#endif  // CAFFE_MTCNN_DATA_LAYER_HPP_
