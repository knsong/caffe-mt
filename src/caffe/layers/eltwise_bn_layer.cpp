#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
namespace caffe {
  template <typename Dtype>
  void EltWiseBNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(N_, bottom[0]->num())<<"N_ must be equal to bottom[0]->num()!";
	CHECK_EQ(C_, bottom[0]->channels())<<"C_ must be equal to bottom[0]->channels()!";
	CHECK_EQ(H_, bottom[0]->height())<<"H_ must be equal to bottom[0]->height()!";
	CHECK_EQ(W_, bottom[0]->width())<<"W_ must be equal to bottom[0]->width()!";
	
	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		bottom[0]->height(), bottom[0]->width());
	
	// mean
	batch_mean_.Reshape(1, C_, H_, W_);
	// temp
	batch_temp1_.Reshape(1, C_, H_, W_);
	// variance
	batch_variance_.Reshape(1, C_, H_, W_);
	// buffer blob
	buffer_blob_.Reshape(N_, C_, H_, W_);

	//recursion_mean.reset(new Blob<Dtype>(1, C_, H_, W_));
	//recursion_variance.reset(new Blob<Dtype>(1, C_, H_, W_)); 
	//temporary variables for back prop.
	diff_xbar.Reshape(N_, C_, H_, W_);
	diff_var.Reshape(1, C_, H_, W_);
	diff_mean.Reshape(1, C_, H_, W_);
	diff_temp_.Reshape(N_, C_, H_, W_);
	var_temp_.Reshape(1, C_, H_, W_);

	//saving normalization result
	 x_bar_.Reshape(N_, C_, H_, W_);
	 
	// fill batch multiplier
	if (N_ != batch_sum_multiplier_.num() ||
		N_ != batch_sum_multiplier_.count())
	{
		batch_sum_multiplier_.Reshape(N_, 1, 1, 1);
		Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
		caffe_set<Dtype>(batch_sum_multiplier_.count(), Dtype(1),

				batch_multiplier_data);
	}
//	this->param_propagate_down_.resize(this->blobs_.size(), true);
  }

  template <typename Dtype>
  void EltWiseBNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
	CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
	"allow in-place computation.";
	
	// Figure out the dimensions
	N_ = bottom[0]->num();
	C_ = bottom[0]->channels();
	H_ = bottom[0]->height();
	W_ = bottom[0]->width();
	var_eps_ = 1e-14;

	average_span = this->layer_param_.bn_param().average_span();

	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
	  LOG(INFO) << "Skipping parameter initialization";
	} else {
	  this->blobs_.resize(4);

	  // fill scale with scale_filler
	  this->blobs_[0].reset(new Blob<Dtype>(1, C_, H_, W_));
	  shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(

		  this->layer_param_.bn_param().scale_filler()));
	  scale_filler->Fill(this->blobs_[0].get());

	  // fill shift with shift_filler
	  this->blobs_[1].reset(new Blob<Dtype>(1, C_, H_, W_));
	  shared_ptr<Filler<Dtype> > shift_filler(GetFiller<Dtype>(

		  this->layer_param_.bn_param().shift_filler()));
	  shift_filler->Fill(this->blobs_[1].get());
	  
	  //mean for test phase
	  this->blobs_[2].reset(new Blob<Dtype>(1, C_, H_, W_));
	  caffe_set<Dtype>(C_*H_*W_, Dtype(0), this->blobs_[2]->mutable_cpu_data());
	  
	  //variance for test phase
	  this->blobs_[3].reset(new Blob<Dtype>(1, C_, H_, W_));
	  caffe_set<Dtype>(C_*H_*W_, Dtype(1), this->blobs_[2]->mutable_cpu_data());
	  
	}  // parameter initialization

	this->param_propagate_down_.resize(this->blobs_.size(), true);
	
  }

  template <typename Dtype>
  void EltWiseBNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* const_top_data = top[0]->cpu_data();

	const Dtype* scale_data = this->blobs_[0]->cpu_data();
	const Dtype* shift_data = this->blobs_[1]->cpu_data();
	
	Dtype* mean_test = this->blobs_[2]->mutable_cpu_data();
	Dtype* variance_test = this->blobs_[3]->mutable_cpu_data();
	
	int count =  N_ * C_ * H_ * W_;
	int dim = count / N_;

	if(this->phase_ == TRAIN)
	{
	/*
		// const Dtype* bottom_data = bottom[0]->cpu_data();
		// put the squares of bottom into buffer_blob_
		caffe_powx(count, bottom_data, Dtype(2),
			buffer_blob_.mutable_cpu_data());

		// computes variance using var(X) = E(X^2) - (EX)^2
		// EX across batch
		caffe_cpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), bottom_data,
			batch_sum_multiplier_.cpu_data(), Dtype(0),
			batch_mean_.mutable_cpu_data());

		// E(X^2) across batch
		caffe_cpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), buffer_blob_.cpu_data(),
			batch_sum_multiplier_.cpu_data(), Dtype(0),
			batch_temp1_.mutable_cpu_data());

		// (EX)^2
		caffe_powx(dim, batch_mean_.cpu_data(), Dtype(2),
			batch_temp2_.mutable_cpu_data());

		//Var(X) = E(X^2) - (EX)^2
		caffe_sub(dim, batch_temp1_.cpu_data(),
			batch_temp2_.cpu_data(),
			batch_variance_.mutable_cpu_data());
	*/

	
		//more stable method to compute variance
		caffe_cpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), bottom_data,
			batch_sum_multiplier_.cpu_data(), Dtype(0),
			batch_mean_.mutable_cpu_data());  			//E(X)

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(-1), batch_sum_multiplier_.cpu_data(),
			batch_mean_.cpu_data(), Dtype(0),
			buffer_blob_.mutable_cpu_data());	
		caffe_add<Dtype>(count, bottom_data, 
			buffer_blob_.cpu_data(),
			buffer_blob_.mutable_cpu_data()); 			//X - E(X)
			
		caffe_powx<Dtype>(count, buffer_blob_.cpu_data(), Dtype(2),
			buffer_blob_.mutable_cpu_data());			//[X - E(X)]^2
		
		caffe_cpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), buffer_blob_.cpu_data(),
			batch_sum_multiplier_.cpu_data(), Dtype(0),

			batch_variance_.mutable_cpu_data());  		//Var(X) = E{[X - E(X)]^2}
				
		//initialization for moving average of batch mean and variance
		if(mean_sequence.size() < average_span)
		{
			shared_ptr<Blob<Dtype> > blob_pointer_mean(new Blob<Dtype>(1, C_, H_, W_));
			shared_ptr<Blob<Dtype> > blob_pointer_variance(new Blob<Dtype>(1, C_, H_, W_));
			mean_sequence.push_back(blob_pointer_mean);
			variance_sequence.push_back(blob_pointer_variance);
			mean_sequence[mean_sequence.size() - 1]->CopyFrom(batch_mean_, false, false);
			variance_sequence[variance_sequence.size() - 1]->CopyFrom(batch_variance_, false, false);
			
			caffe_set<Dtype>(dim, Dtype(0), mean_test);
			caffe_set<Dtype>(dim, Dtype(0), variance_test);
			for(int i = 0; i < mean_sequence.size(); i++){
				caffe_add<Dtype>(dim, mean_sequence[i]->cpu_data(), this->blobs_[2]->cpu_data(), mean_test);
				caffe_add<Dtype>(dim, variance_sequence[i]->cpu_data(), this->blobs_[3]->cpu_data(), variance_test);
			}

			if(0 != mean_sequence.size())
			{
				caffe_scal<Dtype>(dim, Dtype(1. / mean_sequence.size()), mean_test);
				caffe_scal<Dtype>(dim, Dtype(1. / mean_sequence.size()), variance_test);
			}
			else{
				caffe_set<Dtype>(dim, Dtype(0), mean_test);
				caffe_set<Dtype>(dim, Dtype(1), variance_test);
			}
		}
		else if(mean_sequence.size() == average_span)  //update mean and variance sequences
		{
			shared_ptr<Blob<Dtype> > blob_pointer_mean(new Blob<Dtype>(1, C_, H_, W_));
			shared_ptr<Blob<Dtype> > blob_pointer_variance(new Blob<Dtype>(1, C_, H_, W_));

		//	recursion_mean = mean_sequence[0];
		//	recursion_variance = variance_sequence[0];
			for(int i = 0; i < average_span - 1; i++){
				mean_sequence[i] = mean_sequence[i + 1];
				variance_sequence[i] = variance_sequence[i + 1];
			}
			mean_sequence[average_span - 1] = blob_pointer_mean;
			variance_sequence[average_span - 1] = blob_pointer_variance;
			mean_sequence[average_span - 1]->CopyFrom(batch_mean_, false, false);
			variance_sequence[average_span - 1]->CopyFrom(batch_variance_, false, false);
			
			caffe_set<Dtype>(dim, Dtype(0), mean_test);
			caffe_set<Dtype>(dim, Dtype(0), variance_test);
			for(int i = 0; i < average_span; i++){
				caffe_add<Dtype>(dim, this->blobs_[2]->cpu_data(), mean_sequence[i]->cpu_data(), mean_test);
				caffe_add<Dtype>(dim, this->blobs_[3]->cpu_data(), variance_sequence[i]->cpu_data(), variance_test);
			}
			caffe_scal<Dtype>(dim, Dtype(1. / average_span), mean_test);
			caffe_scal<Dtype>(dim, Dtype(1. / average_span), variance_test);
			/*
			//Recursive method to calculate average mean and variance, deprecated because of numerical stability
			caffe_sub(dim, mean_sequence[average_span - 1]->cpu_data(), recursion_mean->cpu_data(), batch_temp1_.mutable_cpu_data());
			caffe_sub(dim, variance_sequence[average_span - 1]->cpu_data(), recursion_variance->cpu_data(), batch_temp2_.mutable_cpu_data());
			caffe_scal(dim, Dtype(1. / average_span), batch_temp1_.mutable_cpu_data());
			caffe_scal(dim, Dtype(1. / average_span), batch_temp2_.mutable_cpu_data());
			caffe_add(dim, this->blobs_[2]->cpu_data(), batch_temp1_.cpu_data(), mean_test);
			caffe_add(dim, this->blobs_[3]->cpu_data(), batch_temp2_.cpu_data(), variance_test);
			*/
		}
		
		caffe_add_scalar<Dtype>(dim, var_eps_,   	//addn eps
			batch_variance_.mutable_cpu_data());
		// normalize variance, namely, Var(x) = sqrt(Var(x) + var_eps)
		caffe_powx<Dtype>(dim,
			batch_variance_.cpu_data(), Dtype(0.5),
			batch_variance_.mutable_cpu_data());

		//normalize bottom data: xbar = (x - E(x)) / sqrt(Var(x) + var_eps), y = scale * xbar + shift
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.cpu_data(),
			batch_mean_.cpu_data(), Dtype(0),
			buffer_blob_.mutable_cpu_data());
		caffe_sub<Dtype>(count, bottom_data, buffer_blob_.cpu_data(), top_data);   //substract E(x)
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.cpu_data(),
			batch_variance_.cpu_data(), Dtype(0),
			buffer_blob_.mutable_cpu_data());
		caffe_div<Dtype>(count, const_top_data, buffer_blob_.cpu_data(), top_data); //divide square of  Var(x) + e
		caffe_copy<Dtype>(count, top_data, x_bar_.mutable_cpu_data());  //save normalized result

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.cpu_data(),
			scale_data, Dtype(0),
			buffer_blob_.mutable_cpu_data());
		caffe_mul<Dtype>(count, const_top_data, buffer_blob_.cpu_data(), top_data); // multiply scale

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.cpu_data(),
			shift_data, Dtype(0),
			buffer_blob_.mutable_cpu_data());
		caffe_add<Dtype>(count, const_top_data, buffer_blob_.cpu_data(), top_data); // add shift
	}
	else //this->phase_ == TEST
	{
		caffe_add_scalar<Dtype>(C_, var_eps_,   	//addn eps
			this->blobs_[3]->mutable_cpu_data());
		// normalize variance
		caffe_powx<Dtype>(dim, this->blobs_[3]->cpu_data(), Dtype(0.5), batch_temp1_.mutable_cpu_data());

		//normalize bottom data: xbar = (x - E(x)) / sqrt(Var(x) + var_eps), y = scale * xbar + shift
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.cpu_data(),
			this->blobs_[2]->cpu_data(), Dtype(0),
			buffer_blob_.mutable_cpu_data());
		caffe_sub<Dtype>(count, bottom_data, buffer_blob_.cpu_data(), top_data);   //substract E(x)

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.cpu_data(),
			batch_temp1_.cpu_data(), Dtype(0),
			buffer_blob_.mutable_cpu_data());
		caffe_div<Dtype>(count, const_top_data, buffer_blob_.cpu_data(), top_data); //divide square of  Var(x) + e

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1,			 // multiply scale
			Dtype(1), batch_sum_multiplier_.cpu_data(),
			scale_data, Dtype(0),
			buffer_blob_.mutable_cpu_data());
		caffe_mul<Dtype>(count, const_top_data, buffer_blob_.cpu_data(), top_data); 

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1,			// add shift
			Dtype(1), batch_sum_multiplier_.cpu_data(),
			shift_data, Dtype(0),
			buffer_blob_.mutable_cpu_data());
		caffe_add<Dtype>(count, const_top_data, buffer_blob_.cpu_data(), top_data); 
	}
  }

  template <typename Dtype>
  void EltWiseBNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down,
	  const vector<Blob<Dtype>*>& bottom) {
   const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();

	const Dtype* scale_data = this->blobs_[0]->cpu_data();
	Dtype* temp_diff = diff_temp_.mutable_cpu_data();
	Dtype* temp_var = var_temp_.mutable_cpu_data();
	int count =  N_ * C_ * H_ * W_;
	int dim = count / N_;

	int num = N_;

	//gradient w.r.t. xbar
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
		Dtype(1), batch_sum_multiplier_.cpu_data(),
		scale_data, Dtype(0),
		buffer_blob_.mutable_cpu_data());
	caffe_mul<Dtype>(count, top_diff, buffer_blob_.cpu_data(), diff_xbar.mutable_cpu_data());

	// gradient w.r.t. Var(x)
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 
		Dtype(1), batch_sum_multiplier_.cpu_data(),
		batch_mean_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());


	caffe_sub<Dtype>(count, bottom_data, buffer_blob_.cpu_data(), buffer_blob_.mutable_cpu_data());   //substract E(x)

	caffe_mul<Dtype>(count, diff_xbar.cpu_data(), buffer_blob_.cpu_data(), buffer_blob_.mutable_cpu_data());
	caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, Dtype(1), 
		buffer_blob_.cpu_data(), batch_sum_multiplier_.cpu_data(), Dtype(0), diff_var.mutable_cpu_data()); //sum i=1,...,num

	caffe_powx<Dtype>(batch_variance_.count(), batch_variance_.cpu_data(), Dtype(3), temp_var);
	caffe_div<Dtype>(dim, diff_var.cpu_data(), var_temp_.cpu_data(), diff_var.mutable_cpu_data());	
	caffe_cpu_scale<Dtype>(dim, Dtype(-0.5), diff_var.cpu_data(), diff_var.mutable_cpu_data());

	// gradient w.r.t. E(x)
	caffe_cpu_gemv<Dtype>(CblasTrans, num, dim,
		Dtype(1), diff_xbar.cpu_data(),
		batch_sum_multiplier_.cpu_data(), Dtype(0),
		diff_mean.mutable_cpu_data());

	caffe_div<Dtype>(dim, diff_mean.cpu_data(), batch_variance_.cpu_data(), diff_mean.mutable_cpu_data());
	caffe_cpu_scale<Dtype>(dim, Dtype(-1), diff_mean.cpu_data(), diff_mean.mutable_cpu_data());

	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 
		Dtype(1), batch_sum_multiplier_.cpu_data(),
		batch_mean_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	caffe_sub<Dtype>(count, bottom_data, buffer_blob_.cpu_data(), buffer_blob_.mutable_cpu_data());   //substract E(x)
	caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 
		Dtype(1), buffer_blob_.cpu_data(), 
		batch_sum_multiplier_.cpu_data(),
		0, var_temp_.mutable_cpu_data());
		
	caffe_mul<Dtype>(dim, var_temp_.cpu_data(), diff_var.cpu_data(), var_temp_.mutable_cpu_data());
	caffe_cpu_scale<Dtype>(dim, Dtype(-2./ num), var_temp_.cpu_data(), var_temp_.mutable_cpu_data());
	caffe_add<Dtype>(dim, var_temp_.cpu_data(), diff_mean.cpu_data(), diff_mean.mutable_cpu_data());

	// gradient w.r.t. x,  propagate down
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,						//the first term
		Dtype(1), batch_sum_multiplier_.cpu_data(),
		batch_variance_.cpu_data(), Dtype(0),
		temp_diff);
	caffe_div<Dtype>(count, diff_xbar.cpu_data(), diff_temp_.cpu_data(), bottom_diff);   

	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,						//the second term
		Dtype(1), batch_sum_multiplier_.cpu_data(),
		diff_var.cpu_data(), Dtype(0),
		temp_diff);
	caffe_mul<Dtype>(count, buffer_blob_.cpu_data(), diff_temp_.cpu_data(), temp_diff);
	caffe_scal<Dtype>(count, Dtype(2. / num), diff_temp_.mutable_cpu_data());
	caffe_add<Dtype>(count, bottom[0]->cpu_diff(), diff_temp_.cpu_data(), bottom_diff);

	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,						//the third term
		Dtype(1. / num), batch_sum_multiplier_.cpu_data(),
		diff_mean.cpu_data(), Dtype(0),
		temp_diff);
	caffe_add<Dtype>(count, bottom[0]->cpu_diff(), diff_temp_.cpu_data(), bottom_diff);

	//gradient w.r.t. scale
	caffe_mul<Dtype>(count,  top[0]->cpu_diff(), x_bar_.cpu_data(), x_bar_.mutable_cpu_data());
	caffe_cpu_gemv<Dtype>(CblasTrans, N_, dim,
		Dtype(1), x_bar_.cpu_data(),
		batch_sum_multiplier_.cpu_data(), Dtype(0),
		scale_diff);


	//gradient w.r.t. shift
	caffe_cpu_gemv<Dtype>(CblasTrans, N_, dim,
		Dtype(1), top[0]->cpu_diff(),
		batch_sum_multiplier_.cpu_data(), Dtype(0),
		shift_diff);

  }
#ifdef CPU_ONLY
STUB_GPU(EltWiseBNLayer);
#endif

  INSTANTIATE_CLASS(EltWiseBNLayer);
 // REGISTER_LAYER_CLASS(BN);
}  // namespace caffe
