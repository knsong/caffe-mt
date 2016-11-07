#ifndef CAFFE_CHANNELWISE_BN_LAYER_HPP_
#define CAFFE_CHANNELWISE_BN_LAYER_HPP_
  /**
  * @brief Batch Normalization per-channel with scale & shift linear transform.
  *
  */
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {
  template <typename Dtype>
  class ChannlWiseBNLayer : public Layer<Dtype> {
   public:
    explicit ChannlWiseBNLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "BN"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    // batch mean & variance
    Blob<Dtype> batch_mean_, batch_variance_, batch_temp1_, spatial_temp_;
    // buffer blob
    Blob<Dtype> buffer_blob_;

    //saved mean and variance blob sequences to calculate the moving average of mean and variance usded in TEST phase
    vector<shared_ptr< Blob<Dtype> > > mean_sequence;
    vector<shared_ptr< Blob<Dtype> > > variance_sequence;

    int average_span;    //used for simple moving average method
	Dtype exponential_average_weight_; //used for exponential moving average method
	int initial_flag;
	//temporary variables for back propagation
    Blob<Dtype> diff_xbar, diff_var, diff_mean, diff_temp_, var_temp_ ;
    // x_sum_multiplier is used to carry out sum using BLAS
    Blob<Dtype> batch_sum_multiplier_, spatial_sum_multiplier_;


	Blob<Dtype> x_bar_;
    // dimension
    int N_;
    int C_;
    int H_;
    int W_;
    // eps
    Dtype var_eps_;
  };
}//namespace caffe
#endif
