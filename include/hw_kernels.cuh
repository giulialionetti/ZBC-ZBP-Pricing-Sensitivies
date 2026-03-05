#ifndef HW_KERNELS_CUH
#define HW_KERNELS_CUH

#include "hw_params.cuh"

__host__ __device__ inline float BtT(float t, float T_maturity){
    return (1.0f - expf(-a * (T_maturity - t))) / a;
}

__host__ __device__ inline float AtT(float t, float T_maturity){
    float B_t_T = BtT(t, T_maturity);
    float fwd_discount = expf(-r0 * (T_maturity - t));
    float fwd_rate = r0;
    float vol_squared = sigma * sigma;
    float convexity_adj = -vol_squared * (1.0f - expf(-2.0f * a * t)) / (4.0f * a) * B_t_T * B_t_T;
    return fwd_discount * expf(B_t_T * fwd_rate + convexity_adj);
}

__host__ __device__ inline float PtT(float t, float T_maturity, float rt){
    float A_t_T = AtT(t, T_maturity);
    float B_t_T = BtT(t, T_maturity);
    return A_t_T * expf(-B_t_T * rt);
}

// Brigo-Mercurio notation: PHI = normal CDF
// call option
__host__ __device__ inline float ZBC(float t, float T_maturity, float S, float K, float rt){
    float P_t_S = PtT(t, S, rt);
    float P_t_T = PtT(t, T_maturity, rt);
    float B_T_S = BtT(T_maturity, S);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    return P_t_S * normcdff(h) - K * P_t_T * normcdff(h - sigma_p);
}

// put option
__host__ __device__ inline float ZBP(float t, float T_maturity, float S, float K, float rt){
    float P_t_T = PtT(t, T_maturity, rt);
    float P_t_S = PtT(t, S, rt);
    float B_T_S = BtT(T_maturity, S);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;

    return K * P_t_T * normcdff(-h + sigma_p) - P_t_S * normcdff(-h);

}

// note : using Pascucci as a reference for phi_h
__host__ __device__ inline float vega_ZBC(float t, float T_maturity, float S, float K, float rt){

    float B_T_S = BtT(T_maturity, S);
    float P_t_S = PtT(t, S, rt);
    float P_t_T = PtT(t, T_maturity, rt);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    float phi_h = expf(-h*h*0.5f) / sqrtf(2.0f * 3.14159265f); // pdf of a Gaussian (first derivative of theta(h))

    float vega_zbc = P_t_S * phi_h * (sigma_p / sigma);
    
    return vega_zbc;
}

__host__ __device__ inline float delta_ZBC(float t, float T_maturity, float S, float K, float rt){
    float B_t_S = BtT(t, S);
    float B_t_T = BtT(t, T_maturity);
    float B_T_S = BtT(T_maturity, S);
    float P_t_S = PtT(t, S, rt);
    float P_t_T = PtT(t, T_maturity, rt);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;

    float delta_zbc = - (B_t_S * P_t_S * normcdff(h)) + (K * B_t_T * P_t_T * normcdff(h- sigma_p));

    return delta_zbc;
}

__host__ __device__ inline float delta_ZBP(float t, float T_maturity, float S, float K, float rt){
    float B_t_S = BtT(t, S);
    float B_t_T = BtT(t, T_maturity);
    float B_T_S = BtT(T_maturity, S);
    float P_t_S = PtT(t, S, rt);
    float P_t_T = PtT(t, T_maturity, rt);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    
    float delta_zbp = - (K * B_t_T * P_t_T * normcdff(-h + sigma_p)) + B_t_S * P_t_S * normcdff(-h);

    return delta_zbp;
}

// note : given the simmetry of phi(h)=phi(-h) --> ZBC vega = ZBP vega
__host__ __device__ inline float vega_ZBP(float t, float T_maturity, float S, float K, float rt){
 
     float B_T_S = BtT(T_maturity, S);
    float P_t_S = PtT(t, S, rt);
    float P_t_T = PtT(t, T_maturity, rt);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T_maturity - t))) / (2.0f * a)) * B_T_S;
    float h = (1.0f / sigma_p) * logf(P_t_S / (P_t_T * K)) + sigma_p / 2.0f;
    float phi_minus_h = expf(-(-h)*(-h)*0.5f) / sqrtf(2.0f * 3.14159265f); // pdf of a Gaussian (first derivative of theta(h))

    float vega_zbp = P_t_S * phi_minus_h * (sigma_p / sigma);
    
    return vega_zbp;
    
}

#endif // HW_KERNELS_CUH