#include "hw_kernels.cuh"
#include "hw_params.cuh"
#include "logger.h"

int main() {
  
    Logger& log = Logger::instance();
    log.open_file("hw_output.log");        
    log.set_level(LogLevel::DEBUG);        
   
    float t  = 0.0f;
    float T  = 1.0f;
    float S  = 5.0f;
    float K  = P0T(S);
    float rt = r0;

  
    float zbc       = ZBC(t, T, S, K, rt);
    float zbp       = ZBP(t, T, S, K, rt);
    float parity    = P0T(S) - K * P0T(T);
    float vega_zbc  = vega_ZBC(t, T, S, K, rt);
    float vega_zbp  = vega_ZBP(t, T, S, K, rt);
    float delta_zbc = delta_ZBC(t, T, S, K, rt);
    float delta_zbp = delta_ZBP(t, T, S, K, rt);

    LOG_INFO("Params: t=%.2f  T=%.2f  S=%.2f  K=%.6f  r0=%.6f",
             t, T, S, K, rt);
    

    LOG_INFO("ZBC               : %.6f", zbc);
    LOG_INFO("ZBP               : %.6f", zbp);
    LOG_INFO("ZBC - ZBP         : %.6f", zbc - zbp);
    LOG_INFO("P(0,S) - K*P(0,T) : %.6f", parity);

   
    float err = (zbc - zbp) - parity;
    if (fabsf(err) < 1e-5f)
        LOG_INFO("Put-call parity   : OK   (err=%.2e)", err);
    else
        LOG_WARN("Put-call parity   : FAIL (err=%.2e)", err);

    
    LOG_INFO("Vega  ZBC         : %.6f", vega_zbc);
    LOG_INFO("Delta ZBC         : %.6f", delta_zbc);
    LOG_INFO("Vega  ZBP         : %.6f", vega_zbp);
    LOG_INFO("Delta ZBP         : %.6f", delta_zbp);
   

    return 0;
}
