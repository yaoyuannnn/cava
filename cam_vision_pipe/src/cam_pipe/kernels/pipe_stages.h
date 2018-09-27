#ifndef _PIPE_STAGES_H_
#define _PIPE_STAGES_H_

#include "common/defs.h"

#define CHAN_SIZE 3

#define ISP 0x3
#define TRANSFORM 0x4

#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

#define abs(a) \
  ({ __typeof__ (a) _a = (a); \
    _a < 0 ? -_a : _a; })

extern int num_ctrl_pts;

void scale_fxp(uint8_t *input, int row_size, int col_size, float *result);

void descale_fxp(float *input, int row_size, int col_size, uint8_t *result);

void demosaic_fxp(float *input, int row_size, int col_size, float *result);

void denoise_fxp(float *input, int row_size, int col_size, float *result);

void transform_fxp(float *input, int row_size, int col_size, float *result,
                   float *TsTw_tran);

void gamut_map_fxp(float *input, int row_size, int col_size, float *result,
                   float *ctrl_pts, float *weights, float *coefs,
                   float *l2_dist);

void tone_map_fxp(float *input, int row_size, int col_size, float *tone_map,
                  float *result);

void tone_map_approx_fxp(float *input, int row_size, int col_size,
                         float *result);

#endif
