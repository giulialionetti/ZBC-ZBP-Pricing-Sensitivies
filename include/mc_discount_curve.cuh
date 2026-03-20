#ifndef MC_DISCOUNT_CURVE_CUH
#define MC_DISCOUNT_CURVE_CUH

#include "mc_engine.cuh"


__global__ void mc_P0T(float* P_estimator, curandState* states){

    int path_id = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_P[N_MAT];

    // each thread clears N_MAT/NTPB slots
    for(int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        s_P[m] = 0.0f;
    __syncthreads();

    if(path_id < N_PATHS){
        curandState local_state = states[path_id];

        float r_step_i = device_r0;
        float discount_factor_integral = 0.0f;
        int maturity_index = 0;

        for(int i = 0; i < N_STEPS; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r_step_i, discount_factor_integral, device_drift_table[i], G);

            if((i + 1) % SAVE_STRIDE == 0){
                atomicAdd(&s_P[maturity_index], expf(-discount_factor_integral));
                maturity_index++;
            }
        }
        states[path_id] = local_state;
    }

    __syncthreads();

    // one atomicAdd per maturity per block to global memory
    for(int m = threadIdx.x; m < N_MAT; m += blockDim.x)
        atomicAdd(&P_estimator[m], s_P[m]);
}

#endif // MC_DISCOUNT_CURVE_CUH