#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

//#include <iostream>
//using namespace std;

namespace caffe {
  template <typename Dtype>
  void ChannlWiseBNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
	int spatial_dim = H_ * W_;
	if(this->phase_ == TRAIN)
	{ 
	
	//more stable method to compute variance
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), bottom_data,
			batch_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());  			    //batch mean
		caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, spatial_dim,
			Dtype(1. / spatial_dim), batch_temp1_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_mean_.mutable_gpu_data()); 				//spatial mean, namely, E(X)
			
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
			Dtype(1), batch_mean_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(-1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());	          
		caffe_gpu_add<Dtype>(count, bottom_data, 
			buffer_blob_.gpu_data(),
			buffer_blob_.mutable_gpu_data()); 			//X - E(X)
			
		caffe_copy<Dtype>(count, buffer_blob_.gpu_data(), top_data); //first step to normalize the bottom data, namely substracting E(x), skip some lines to see the left steps	
			
		caffe_gpu_powx<Dtype>(count, buffer_blob_.gpu_data(), Dtype(2),
			buffer_blob_.mutable_gpu_data());			//[X - E(X)]^2
		
		caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
			Dtype(1. / N_), buffer_blob_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());  
		caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, spatial_dim,
			Dtype(1. / spatial_dim), batch_temp1_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_variance_.mutable_gpu_data()); 	 	//Var(X) = E{[X - E(X)]^2}
	
	//	caffe_gpu_add_scalar<Dtype>(C_, var_eps_,   	//addn eps
	//		batch_variance_.mutable_gpu_data());
	
		//LOG(INFO) << "average span: " << average_span;
		//initialization for moving average of batch mean and variance
		if(0 != average_span)//simple moving average method
		{	
			Dtype bias_correction_factor = N_ > 1 ? Dtype(N_)/(N_-1) : 1;
			if(mean_sequence.size() < average_span)
			{
				shared_ptr<Blob<Dtype> > blob_pointer_mean(new Blob<Dtype>(1, C_, 1, 1));
				shared_ptr<Blob<Dtype> > blob_pointer_variance(new Blob<Dtype>(1, C_, 1, 1));
				mean_sequence.push_back(blob_pointer_mean);
				variance_sequence.push_back(blob_pointer_variance);
				mean_sequence[mean_sequence.size() - 1]->CopyFrom(batch_mean_, false, false);
				variance_sequence[variance_sequence.size() - 1]->CopyFrom(batch_variance_, false, false);
				
				caffe_gpu_set<Dtype>(C_, Dtype(0), mean_test);
				caffe_gpu_set<Dtype>(C_, Dtype(0), variance_test);
				for(int i = 0; i < mean_sequence.size(); i++){
					caffe_gpu_add<Dtype>(C_, mean_sequence[i]->gpu_data(), this->blobs_[2]->gpu_data(), mean_test);
					caffe_gpu_add<Dtype>(C_, variance_sequence[i]->gpu_data(), this->blobs_[3]->gpu_data(), variance_test);
				}
				if(0 != mean_sequence.size())
				{
					caffe_gpu_scal<Dtype>(C_, Dtype(1. / mean_sequence.size()), mean_test);
					caffe_gpu_scal<Dtype>(C_, Dtype(1. / mean_sequence.size() * bias_correction_factor), variance_test);
				}
				else{
					caffe_gpu_set<Dtype>(C_, Dtype(0), mean_test);
					caffe_gpu_set<Dtype>(C_, Dtype(1), variance_test);
				}
			}
			else if(mean_sequence.size() == average_span)  //update mean and variance sequences
			{
				shared_ptr<Blob<Dtype> > blob_pointer_mean(new Blob<Dtype>(1, C_, 1, 1));
				shared_ptr<Blob<Dtype> > blob_pointer_variance(new Blob<Dtype>(1, C_, 1, 1));

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
				
				caffe_gpu_set<Dtype>(C_, Dtype(0), mean_test);
				caffe_gpu_set<Dtype>(C_, Dtype(0), variance_test);
				for(int i = 0; i < average_span; i++){
					caffe_gpu_add<Dtype>(C_, this->blobs_[2]->gpu_data(), mean_sequence[i]->gpu_data(), mean_test);
					caffe_gpu_add<Dtype>(C_, this->blobs_[3]->gpu_data(), variance_sequence[i]->gpu_data(), variance_test);
				}
				caffe_gpu_scal<Dtype>(C_, Dtype(1. / average_span), mean_test);
				caffe_gpu_scal<Dtype>(C_, Dtype(1. / average_span * bias_correction_factor), variance_test);
			}
		}
		else{
			Dtype bias_correction_factor = N_ > 1 ? Dtype(N_)/(N_-1) : 1;
			if(initial_flag)
			{
				caffe_gpu_axpby(batch_mean_.count(), Dtype(exponential_average_weight_), batch_mean_.gpu_data(),
					Dtype(1 - exponential_average_weight_), this->blobs_[2]->mutable_gpu_data());
				caffe_gpu_axpby(batch_variance_.count(), Dtype(exponential_average_weight_*bias_correction_factor), batch_variance_.gpu_data(),
					Dtype(1 - exponential_average_weight_), this->blobs_[3]->mutable_gpu_data());
			}
			else{
			/*
				caffe_set<Dtype>(this->blobs_[2]->count(), Dtype(0), this->blobs_[2]->mutable_gpu_data());
				
				caffe_set<Dtype>(this->blobs_[3]->count(), Dtype(0), this->blobs_[3]->mutable_gpu_data());
				
			*/		
				caffe_gpu_set<Dtype>(this->blobs_[2]->count(), Dtype(0), this->blobs_[2]->mutable_gpu_data());
				caffe_gpu_set<Dtype>(this->blobs_[3]->count(), Dtype(0), this->blobs_[3]->mutable_gpu_data());
				caffe_copy<Dtype>(this->blobs_[2]->count(), batch_mean_.gpu_data(), this->blobs_[2]->mutable_gpu_data());
				caffe_gpu_axpby(this->blobs_[3]->count(), Dtype(bias_correction_factor), batch_variance_.gpu_data(),
					Dtype(0), this->blobs_[3]->mutable_gpu_data());
				initial_flag = 1;
			}
	
		}
		caffe_gpu_add_scalar<Dtype>(C_, var_eps_,   	//addn eps
			batch_variance_.mutable_gpu_data());
		//Var(x) = sqrt(Var(x) + var_eps)
		caffe_gpu_powx<Dtype>(C_,
			batch_variance_.gpu_data(), Dtype(0.5),
			batch_variance_.mutable_gpu_data());

		//normalize bottom data: xbar = (x - E(x)) / sqrt(Var(x) + var_eps), y = scale * xbar + shift
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
			Dtype(1), batch_variance_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());	 
		caffe_gpu_div<Dtype>(count, const_top_data, buffer_blob_.gpu_data(), top_data); //divide square of  Var(x) + e
		caffe_copy<Dtype>(count, top_data, x_bar_.mutable_gpu_data());  //save normalized result
		
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
			Dtype(1), scale_data,
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());	 
		caffe_gpu_mul<Dtype>(count, const_top_data, buffer_blob_.gpu_data(), top_data); // multiply scale
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
			Dtype(1), shift_data,
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_add<Dtype>(count, const_top_data, buffer_blob_.gpu_data(), top_data); // add shift
	}
	else //this->phase_ == TEST
	{
		caffe_gpu_add_scalar<Dtype>(C_, var_eps_,   	//addn eps
			this->blobs_[3]->mutable_gpu_data());
		// normalize variance
		caffe_gpu_powx<Dtype>(C_, this->blobs_[3]->gpu_data(), Dtype(0.5), spatial_temp_.mutable_gpu_data());

		//normalize bottom data: xbar = (x - E(x)) / sqrt(Var(x) + var_eps), y = scale * xbar + shift
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
			Dtype(1), this->blobs_[2]->gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_sub<Dtype>(count, bottom_data, buffer_blob_.gpu_data(), top_data);   //substract E(x)
		
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
			Dtype(1), spatial_temp_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_div<Dtype>(count, const_top_data, buffer_blob_.gpu_data(), top_data); //divide square of  Var(x) + e

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
			Dtype(1), scale_data,
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data()); 
		caffe_gpu_mul<Dtype>(count, const_top_data, buffer_blob_.gpu_data(), top_data); // multiply scale
		
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
			Dtype(1), shift_data,
			spatial_sum_multiplier_.gpu_data(), Dtype(0),
			batch_temp1_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
			Dtype(1), batch_sum_multiplier_.gpu_data(),
			batch_temp1_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data()); 
		caffe_gpu_add<Dtype>(count, const_top_data, buffer_blob_.gpu_data(), top_data); // add shift
	}
  }

  template <typename Dtype>
  void ChannlWiseBNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
	int spatial_dim = H_ * W_;
	int num = N_;

	//gradient w.r.t. xbar
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
		Dtype(1), scale_data,
		spatial_sum_multiplier_.gpu_data(), Dtype(0),
		batch_temp1_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		batch_temp1_.gpu_data(), Dtype(0),
		buffer_blob_.mutable_gpu_data()); 
	caffe_gpu_mul<Dtype>(count, top_diff, buffer_blob_.gpu_data(), diff_xbar.mutable_gpu_data());

	// gradient w.r.t. Var(x)
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
		Dtype(1), batch_mean_.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), Dtype(0),
		batch_temp1_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
		Dtype(-1), batch_sum_multiplier_.gpu_data(),
		batch_temp1_.gpu_data(), Dtype(0),
		buffer_blob_.mutable_gpu_data());	          
	caffe_gpu_add<Dtype>(count, bottom_data, 
		buffer_blob_.gpu_data(),
		buffer_blob_.mutable_gpu_data());		//substract E(x)

	caffe_gpu_mul<Dtype>(count, diff_xbar.gpu_data(), buffer_blob_.gpu_data(), buffer_blob_.mutable_gpu_data());
	caffe_gpu_gemv<Dtype>(CblasTrans, num, dim, Dtype(1), 
		buffer_blob_.gpu_data(), batch_sum_multiplier_.gpu_data(), Dtype(0), batch_temp1_.mutable_gpu_data()); 
	caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, spatial_dim, Dtype(1), 
		batch_temp1_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0), diff_var.mutable_gpu_data());//sum i=1,...,num * height * width
	caffe_gpu_powx<Dtype>(batch_variance_.count(), batch_variance_.gpu_data(), Dtype(3), temp_var);
	caffe_gpu_div<Dtype>(C_, diff_var.gpu_data(), var_temp_.gpu_data(), diff_var.mutable_gpu_data());	
	caffe_gpu_scale<Dtype>(C_, Dtype(-0.5), diff_var.gpu_data(), diff_var.mutable_gpu_data());

	// gradient w.r.t. E(x)
	caffe_gpu_gemv<Dtype>(CblasTrans, num, dim, Dtype(1), 
		diff_xbar.gpu_data(), batch_sum_multiplier_.gpu_data(), Dtype(0), batch_temp1_.mutable_gpu_data()); 
	caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, spatial_dim, Dtype(1), 
		batch_temp1_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0), diff_mean.mutable_gpu_data());
		
	caffe_gpu_div<Dtype>(C_, diff_mean.gpu_data(), batch_variance_.gpu_data(), diff_mean.mutable_gpu_data());
	caffe_gpu_scale<Dtype>(C_, Dtype(-1), diff_mean.gpu_data(), diff_mean.mutable_gpu_data());

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
		Dtype(1), batch_mean_.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), Dtype(0),
		batch_temp1_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		batch_temp1_.gpu_data(), Dtype(0),
		buffer_blob_.mutable_gpu_data());
	caffe_gpu_sub<Dtype>(count, bottom_data, buffer_blob_.gpu_data(), buffer_blob_.mutable_gpu_data());   //substract E(x)
		
	caffe_gpu_gemv<Dtype>(CblasTrans, num, dim, Dtype(1), 
		buffer_blob_.gpu_data(), batch_sum_multiplier_.gpu_data(), Dtype(0), batch_temp1_.mutable_gpu_data()); 
	caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, spatial_dim, Dtype(1), 
		batch_temp1_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0), temp_var);
		
	caffe_gpu_mul<Dtype>(C_, var_temp_.gpu_data(), diff_var.gpu_data(), var_temp_.mutable_gpu_data());
	caffe_gpu_scale<Dtype>(C_, Dtype(-2./(num * spatial_dim)), var_temp_.gpu_data(), var_temp_.mutable_gpu_data());
	caffe_gpu_add<Dtype>(C_, var_temp_.gpu_data(), diff_mean.gpu_data(), diff_mean.mutable_gpu_data());

	// gradient w.r.t. x,  propagate down
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
		Dtype(1), batch_variance_.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), Dtype(0),
		batch_temp1_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		batch_temp1_.gpu_data(), Dtype(0),
		diff_temp_.mutable_gpu_data());
	caffe_gpu_div<Dtype>(count, diff_xbar.gpu_data(), diff_temp_.gpu_data(), bottom_diff);   //the first term

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 
		Dtype(1), diff_var.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), Dtype(0),
		batch_temp1_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
		Dtype(1), batch_sum_multiplier_.gpu_data(),
		batch_temp1_.gpu_data(), Dtype(0),
		diff_temp_.mutable_gpu_data());
	caffe_gpu_mul<Dtype>(count, buffer_blob_.gpu_data(), diff_temp_.gpu_data(), temp_diff);
	caffe_gpu_scal<Dtype>(count, Dtype(2. / (num * spatial_dim)), diff_temp_.mutable_gpu_data());
	caffe_gpu_add<Dtype>(count, bottom[0]->gpu_diff(), diff_temp_.gpu_data(), bottom_diff);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, spatial_dim, 1, 			//the third term
		Dtype(1), diff_mean.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), Dtype(0),
		batch_temp1_.mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, dim, 1, 
		Dtype(1. / (num * spatial_dim)), batch_sum_multiplier_.gpu_data(),
		batch_temp1_.gpu_data(), Dtype(0),
		diff_temp_.mutable_gpu_data());
	caffe_gpu_add<Dtype>(count, bottom[0]->gpu_diff(), diff_temp_.gpu_data(), bottom_diff);

	//gradient w.r.t. scale
	caffe_gpu_mul<Dtype>(count,  top[0]->gpu_diff(), x_bar_.gpu_data(), x_bar_.mutable_gpu_data());
	caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
		Dtype(1), x_bar_.gpu_data(),
		batch_sum_multiplier_.gpu_data(), Dtype(0),
		batch_temp1_.mutable_gpu_data());
	caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, spatial_dim,
		Dtype(1), batch_temp1_.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), Dtype(0),
		scale_diff);
		
	//gradient w.r.t. shift
	caffe_gpu_gemv<Dtype>(CblasTrans, N_, dim,
		Dtype(1), top[0]->gpu_diff(),
		batch_sum_multiplier_.gpu_data(), Dtype(0),
		batch_temp1_.mutable_gpu_data());
	caffe_gpu_gemv<Dtype>(CblasNoTrans, C_, spatial_dim,
		Dtype(1), batch_temp1_.gpu_data(),
		spatial_sum_multiplier_.gpu_data(), Dtype(0),
		shift_diff);

	
  }

  INSTANTIATE_LAYER_GPU_FUNCS(ChannlWiseBNLayer);
}  // namespace caffe
