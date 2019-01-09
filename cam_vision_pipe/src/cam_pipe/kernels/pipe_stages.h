#ifndef _PIPE_STAGES_H_
#define _PIPE_STAGES_H_

#include "common/defs.h"

#define CHAN_SIZE 3

#define ISP 0x4
#define LD_PARAMS 0x5

#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

extern int num_ctrl_pts;

void isp_hw_impl(int row_size,
                 int col_size,
                 uint8_t* acc_input,
                 uint8_t* acc_result,
                 float* acc_input_scaled,
                 float* acc_result_scaled,
                 float* acc_TsTw,
                 float* acc_ctrl_pts,
                 float* acc_weights,
                 float* acc_coefs,
                 float* acc_tone_map,
                 float* acc_l2_dist);
#endif
