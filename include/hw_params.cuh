

#ifndef HW_PARAMS_CUH
#define HW_PARAMS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define T_MAX 10.0f
#define N_steps 1000
const float a = 1.0f;
const float sigma = 0.1f;
const float r0 = 0.012f;
const float dt = T_MAX / N_steps;

// at time 0 the price is known and consequently its forward rate 
__host__ __device__ inline float P0T(float T){
    return expf(-r0 * T);
}

__host__ __device__  inline float f0T(){
    return r0;
}



#endif // HW_PARAMS_CUH