#include <algorithm>
#include <vector>

#include "caffe/layers/elementwise_bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

//#include <iostream>
//using namespace std;

namespace caffe {
  template <typename Dtype>
  void EltWiseBNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* const_top_data = top[0]->gpu_data();

	const Dtype* scale_data = this->blobs_[0]->gpu_data();
	const Dtype* shift_data = this->blobs_[1]->gpu_data();
	
	Dtype* mean_test = this->blobs_[2]->mutable_gpu_data();
	Dtype* variance_test = this->blobs_[3]->mutable_gpu_data();
	
	int count =  N_ * C_ * H_ * W_;
	int dim = count / N_;

	if(this->phase_ == TRAIN)
	{
		// const Dtype* bottom_data = bottom[0]->gpu_data();
		// put the squares of bottom into buffer_blob_
	/*
		caffe_gpu_powx<Dtype>(count, bottom_data, Dtype(2),
			buffer_blob_.mutable_gpu_data());
		
		// computes variance using var(X) = E(X^2) - (EX)^2
		// EX across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), bottom_data,
			batch_sum_multiplier_.gpu_data(), Dtype(0),
			batch_mean_.mutable_gpu_data());
		
		// E(X^2) across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), buffer_blob_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());

		// (EX)^2
		caffe_gpu_powx<Dtype>(dim, batch_mean_.gpu_data(), Dtype(2),
			batch_temp2_.mutable_gpu_data());
		
		//Var(X) = E(X^2) - (EX)^2
		caffe_gpu_sub<Dtype>(dim, batch_temp1_.gpu_data(),
			batch_temp2_.gpu_data(),
			batch_variance_.mutable_gpu_data());
	*/

		//more stable method to compute variance
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), bottom_data,
			batch_sum_multiplier_.gpu_data(), Dtype(0),
			batch_mean_.mutable_gpu_data());  			//E(X)
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(-1), batch_sum_multiplier_.gpu_data(),
			batch_mean_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());	
		
		caffe_gpu_add<Dtype>(buffer_blob_.count(), bottom_data,
			buffer_blob_.gpu_data(),
			buffer_blob_.mutable_gpu_data()); 			//X - E(X)
			
		caffe_gpu_powx<Dtype>(buffer_blob_.count(), buffer_blob_.gpu_data(), Dtype(2),
			buffer_blob_.mutable_gpu_data());			//[X - E(X)]^2
		
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), buffer_blob_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0),
			batch_variance_.mutable_gpu_data());  		//Var(X) = E{[X - E(X)]^2}
			
		caffe_gpu_add_scalar<Dtype>(dim, var_eps_,			//add eps
			batch_variance_.mutable_gpu_data());


		//initialization for moving average of batch mean and variance
		if(mean_sequence.size() < average_span)
		{
			//LOG(INFO) << "mean_sequence.size() < average_span";
			shared_ptr<Blob<Dtype> > blob_pointer_mean(new Blob<Dtype>(1, C_, H_, W_));
			shared_ptr<Blob<Dtype> > blob_pointer_variance(new Blob<Dtype>(1, C_, H_, W_));
			mean_sequence.push_back(blob_pointer_mean);
			variance_sequence.push_back(blob_pointer_variance);
			mean_sequence[mean_sequence.size() - 1]->CopyFrom(batch_mean_, false, false);
			variance_sequence[variance_sequence.size() - 1]->CopyFrom(batch_variance_, false, false);
			
			caffe_gpu_set<Dtype>(dim, Dtype(0), mean_test);
			caffe_gpu_set<Dtype>(dim, Dtype(0), variance_test);
			for(int i = 0; i < mean_sequence.size(); i++){
				caffe_gpu_add<Dtype>(dim, mean_sequence[i]->gpu_data(), this->blobs_[2]->gpu_data(), mean_test);
				caffe_gpu_add<Dtype>(dim, variance_sequence[i]->gpu_data(), this->blobs_[3]->gpu_data(), variance_test);
			}
			if(0 != mean_sequence.size())
			{
				caffe_gpu_scal<Dtype>(dim, Dtype(1. / mean_sequence.size()), mean_test);
				caffe_gpu_scal<Dtype>(dim, Dtype(1. / mean_sequence.size()), variance_test);
			}
			else{
				caffe_gpu_set<Dtype>(dim, Dtype(0), mean_test);
				caffe_gpu_set<Dtype>(dim, Dtype(1), variance_test);
			}
			
		}
		else if(mean_sequence.size() == average_span)  
		{
			//update mean and variance sequences
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
			
			caffe_gpu_set<Dtype>(dim, Dtype(0), mean_test);
			caffe_gpu_set<Dtype>(dim, Dtype(0), variance_test);
			for(int i = 0; i < average_span; i++){
				caffe_gpu_add<Dtype>(dim, this->blobs_[2]->gpu_data(), mean_sequence[i]->gpu_data(), mean_test);
				caffe_gpu_add<Dtype>(dim, this->blobs_[3]->gpu_data(), variance_sequence[i]->gpu_data(), variance_test);
			}
			caffe_gpu_scal<Dtype>(dim, Dtype(1. / average_span), mean_test);
			caffe_gpu_scal<Dtype>(dim, Dtype(1. / average_span), variance_test);
			/*
			//Recursive method to calculate average mean and variance
			caffe_gpu_sub<Dtype>(dim, mean_sequence[average_span - 1]->gpu_data(), recursion_mean->gpu_data(), batch_temp1_.mutable_gpu_data());
			caffe_gpu_sub<Dtype>(dim, variance_sequence[average_span - 1]->gpu_data(), recursion_variance->gpu_data(), batch_temp2_.mutable_gpu_data());
			caffe_gpu_scal<Dtype>(dim, Dtype(1. / average_span), batch_temp1_.mutable_gpu_data());
			caffe_gpu_scal<Dtype>(dim, Dtype(1. / average_span), batch_temp2_.mutable_gpu_data());
			caffe_gpu_add<Dtype>(dim, this->blobs_[2]->gpu_data(), batch_temp1_.gpu_data(), mean_test);
			caffe_gpu_add<Dtype>(dim, this->blobs_[3]->gpu_data(), batch_temp2_.gpu_data(), variance_test);
			*/
		}
	
		// normalize variance, namely, Var(x) = sqrt(Var(x) + var_eps)
		caffe_gpu_powx<Dtype>(dim,
			batch_variance_.gpu_data(), Dtype(0.5),
			batch_variance_.mutable_gpu_data());

		//normalize bottom data: xbar = (x - E(x)) / sqrt(Var(x) + var_eps), y = scale * xbar + shift
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_mean_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_sub<Dtype>(buffer_blob_.count(), bottom_data, buffer_blob_.gpu_data(), top_data);   //substract E(x)

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_variance_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_div<Dtype>(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data); //divide square of  Var(x) + e

		caffe_copy<Dtype>(x_bar_.count(), top_data, x_bar_.mutable_gpu_data());  //save normalized result

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			scale_data, Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_mul<Dtype>(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data); // multiply scale

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			shift_data, Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_add<Dtype>(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data); // add shift
	}
	else //this->phase_ == TEST
	{

		// normalize variance
		caffe_gpu_powx<Dtype>(dim, this->blobs_[3]->gpu_data(), Dtype(0.5), batch_temp1_.mutable_gpu_data());

		//normalize bottom data: xbar = (x - E(x)) / sqrt(Var(x) + var_eps), y = scale * xbar + shift
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			this->blobs_[2]->gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_sub<Dtype>(buffer_blob_.count(), bottom_data, buffer_blob_.gpu_data(), top_data);   //substract E(x)
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_div<Dtype>(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data); //divide square of  Var(x) + e

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1,			 // multiply scale
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			scale_data, Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_mul<Dtype>(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data); 

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1,			// add shift
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			shift_data, Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_add<Dtype>(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data); 
	}
  }

  template <typename Dtype>
  void EltWiseBNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
	Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
	const Dtype* scale_data = this->blobs_[0]->gpu_data();
	Dtype* temp_diff = diff_temp_.mutable_gpu_data();
	Dtype* temp_var = var_temp_.mutable_gpu_data();
	
	int count =  N_ * C_ * H_ * W_;
	int dim = count / N_;
	int num = N_;

	//gradient w.r.t. xbar
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		scale_data, Dtype(0),
		buffer_blob_.mutable_gpu_data());
	caffe_gpu_mul<Dtype>(buffer_blob_.count(), top_diff, buffer_blob_.gpu_data(), diff_xbar.mutable_gpu_data());

	// gradient w.r.t. Var(x)
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		batch_mean_.gpu_data(), Dtype(0),
		buffer_blob_.mutable_gpu_data());
	caffe_gpu_sub<Dtype>(buffer_blob_.count(), bottom_data, buffer_blob_.gpu_data(), buffer_blob_.mutable_gpu_data());   //substract E(x)

	caffe_gpu_mul<Dtype>(buffer_blob_.count(), diff_xbar.gpu_data(), buffer_blob_.gpu_data(), buffer_blob_.mutable_gpu_data());
	caffe_gpu_gemv<Dtype>(CblasTrans, num, dim, Dtype(1), 
		buffer_blob_.gpu_data(), batch_sum_multiplier_.gpu_data(), Dtype(0), diff_var.mutable_gpu_data()); //sum i=1,...,num

	caffe_gpu_powx<Dtype>(batch_variance_.count(), batch_variance_.gpu_data(), 3, temp_var);
	caffe_gpu_div<Dtype>(dim, diff_var.gpu_data(), var_temp_.gpu_data(), diff_var.mutable_gpu_data());	
	caffe_gpu_scale<Dtype>(dim, Dtype(-0.5), diff_var.gpu_data(), diff_var.mutable_gpu_data());

	// gradient w.r.t. E(x)
	caffe_gpu_gemv<Dtype>(CblasTrans, num, dim,
		Dtype(1), diff_xbar.gpu_data(),
		batch_sum_multiplier_.gpu_data(), Dtype(0),
		diff_mean.mutable_gpu_data());
	caffe_gpu_div<Dtype>(dim, diff_mean.gpu_data(), batch_variance_.gpu_data(), diff_mean.mutable_gpu_data());
	caffe_gpu_scale<Dtype>(dim, Dtype(-1), diff_mean.gpu_data(), diff_mean.mutable_gpu_data());

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		batch_mean_.gpu_data(), Dtype(0),
		buffer_blob_.mutable_gpu_data());
	caffe_gpu_sub<Dtype>(buffer_blob_.count(), bottom_data, buffer_blob_.gpu_data(), buffer_blob_.mutable_gpu_data());   //substract E(x)
	caffe_gpu_gemv<Dtype>(CblasTrans, num, dim, Dtype(1), 
		buffer_blob_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0, var_temp_.mutable_gpu_data());
	caffe_gpu_mul<Dtype>(dim, var_temp_.gpu_data(), diff_var.gpu_data(), var_temp_.mutable_gpu_data());
	caffe_gpu_scale<Dtype>(dim, Dtype(-2. / num), var_temp_.gpu_data(), var_temp_.mutable_gpu_data());
	caffe_gpu_add<Dtype>(dim, var_temp_.gpu_data(), diff_mean.gpu_data(), diff_mean.mutable_gpu_data());

	// gradient w.r.t. x,  propagate down
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,						//the first term
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		batch_variance_.gpu_data(), Dtype(0),
		temp_diff);
	caffe_gpu_div<Dtype>(diff_xbar.count(), diff_xbar.gpu_data(), diff_temp_.gpu_data(), bottom_diff);   

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,						//the second term
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		diff_var.gpu_data(), Dtype(0),
		temp_diff);
	caffe_gpu_mul<Dtype>(buffer_blob_.count(), buffer_blob_.gpu_data(), diff_temp_.gpu_data(), temp_diff);
	caffe_gpu_scale<Dtype>(diff_temp_.count(), Dtype(2./num), diff_temp_.gpu_data(), diff_temp_.mutable_gpu_data());
	caffe_gpu_add<Dtype>(diff_temp_.count(), bottom[0]->gpu_diff(), diff_temp_.gpu_data(), bottom_diff);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1,						//the third term
		Dtype(1. / num), batch_sum_multiplier_.gpu_data(),
		diff_mean.gpu_data(), Dtype(0),
		temp_diff);
	caffe_gpu_add<Dtype>(diff_temp_.count(), bottom[0]->gpu_diff(), diff_temp_.gpu_data(), bottom_diff);

	//gradient w.r.t. scale
	caffe_gpu_mul<Dtype>(x_bar_.count(), top[0]->gpu_diff(), x_bar_.gpu_data(), x_bar_.mutable_gpu_data());
	caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
		Dtype(1), x_bar_.gpu_data(),
		batch_sum_multiplier_.gpu_data(), Dtype(0),
		scale_diff);

	//gradient w.r.t. shift
	caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
		Dtype(1), top[0]->gpu_diff(),
		batch_sum_multiplier_.gpu_data(), Dtype(0),
		shift_diff);

  }

  INSTANTIATE_LAYER_GPU_FUNCS(EltWiseBNLayer);
}  // namespace caffe
