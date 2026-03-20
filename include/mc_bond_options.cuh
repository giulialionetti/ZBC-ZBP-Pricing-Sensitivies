#ifndef MC_BOND_OPTIONS_CUH
#define MC_BOND_OPTIONS_CUH

#include "mc_engine.cuh"
#include "hw_pricing.cuh"
#include "hw_greeks_first.cuh"


__global__ void mc_zbc_vega(float* ZBC_estimator, float* vega_estimator,
 curandState* states, float T_maturity, float S, float K,
 const float* P_market, const float* f_market){

        int path_id = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ float shared_zbc[NTPB];
        __shared__ float shared_vega[NTPB];

        float thread_zbc = 0.0f;
        float thread_vega = 0.0f;

        if(path_id < N_PATHS){

            curandState local_state = states[path_id];

            float r_step_i = device_r0;
            float discount_factor_integral = 0.0f;
            float drdsigma_step_i = 0.0f;
            float drdsigma_integral = 0.0f;

            int n_steps_T = (int)(T_maturity / device_dt);
            for(int i=0; i < n_steps_T; i++){
                float G = curand_normal(&local_state);
                evolve_short_rate(r_step_i, discount_factor_integral, device_drift_table[i], G);
                evolve_short_rate_derivative(drdsigma_step_i, drdsigma_integral,
                                            device_sensitivity_drift_table[i], G);
            }

            float bond_price_at_maturity = (P_market == nullptr) ?
            FlatCurve{device_a, device_sigma, device_r0}.P(T_maturity, S, r_step_i) :
            MarketCurve{device_a, device_sigma, P_market, f_market, MAT_SPACING, N_MAT}.P(T_maturity, S, r_step_i);
            float discount_factor = expf(-discount_factor_integral);
            float B_val = BtT(T_maturity, S, device_a);
            float convexity_dsigma = (device_sigma / (2.0f * device_a))
                       * (1.0f - expf(-2.0f * device_a * T_maturity))
                       * B_val * B_val;

            float dPricedsigma = -bond_price_at_maturity
                   * (B_val * drdsigma_step_i + convexity_dsigma);

            thread_zbc = discount_factor * fmaxf(bond_price_at_maturity - K, 0.0f);
            thread_vega = discount_factor * dPricedsigma * (bond_price_at_maturity > K ? 1.0f : 0.0f)
                        - drdsigma_integral * discount_factor * fmaxf(bond_price_at_maturity - K, 0.0f);
            states[path_id] = local_state;
        }

        shared_zbc[threadIdx.x] = thread_zbc;
        shared_vega[threadIdx.x] = thread_vega;
        __syncthreads();

        for(int i = NTPB/2; i > 0 ; i >>= 1){
            if(threadIdx.x < i){
                shared_zbc[threadIdx.x] += shared_zbc[threadIdx.x + i];
                shared_vega[threadIdx.x] += shared_vega[threadIdx.x + i];
            }
            __syncthreads();
        }

        if(threadIdx.x == 0){
            atomicAdd(ZBC_estimator, shared_zbc[0]);
            atomicAdd(vega_estimator, shared_vega[0]);
        }
}


#endif // MC_BOND_OPTIONS_CUH