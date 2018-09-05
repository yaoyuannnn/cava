#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "config.h"
#include "kernels/pipe_stages.h"
#include "utility/load_camera_model.h"
#include "utility/utility.h"
#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

void demosaic_nn_hw(float *host_input, float *host_result, int row_size,
                    int col_size, int chan_size, float *acc_input,
                    float *acc_result) {
  dmaLoad(acc_input, host_input,
          row_size * col_size * chan_size * sizeof(float));
  demosaic_nn_fxp(acc_input, row_size, col_size, chan_size, acc_result);
  dmaStore(host_result, acc_result,
           row_size * col_size * chan_size * sizeof(float));
}

void transform_hw(float *host_input, float *host_result, int row_size,
                  int col_size, int chan_size, float *acc_input,
                  float *acc_result, float *TsTw_tran) {
  dmaLoad(acc_input, host_input,
          row_size * col_size * chan_size * sizeof(float));
  transform_fxp(acc_input, row_size, col_size, chan_size, acc_result,
                TsTw_tran);
  dmaStore(host_result, acc_result,
           row_size * col_size * chan_size * sizeof(float));
}

void camera_pipe(float *host_input, float *host_result, int row_size,
                int col_size, int chan_size) {
  float *Ts, *Tw, *TsTw, *TsTw_tran;
  float *ctrl_pts, *weights, *coefs;

  // Load model parameters from file
  // NOTE: Ts, Tw, and TsTw read only forward data
  // ctrl_pts, weights, and coefs are either forward or backward
  // tone mapping is always backward
  // This is due to the the camera model format

  Ts            = get_Ts       (cam_model_path);
  Tw            = get_Tw       (cam_model_path, wb_index);
  TsTw          = get_TsTw     (cam_model_path, wb_index);
  ctrl_pts      = get_ctrl_pts (cam_model_path, num_ctrl_pts);
  weights       = get_weights  (cam_model_path, num_ctrl_pts);
  coefs         = get_coefs    (cam_model_path, num_ctrl_pts);

  float *acc_input, *acc_result;
  int err = posix_memalign((void **)&acc_input, CACHELINE_SIZE,
                        sizeof(float) * row_size * col_size * chan_size);
  err |= posix_memalign((void **)&acc_result, CACHELINE_SIZE,
                        sizeof(float) * row_size * col_size * chan_size);
  assert(err == 0 && "Failed to allocate memory!");

  // Invoke the demosaic kernel
  MAP_ARRAY_TO_ACCEL(DEMOSAIC_NN, "host_input", host_input,
                     sizeof(float) * row_size * col_size * chan_size);
  MAP_ARRAY_TO_ACCEL(DEMOSAIC_NN, "host_result", host_result,
                     sizeof(float) * row_size * col_size * chan_size);
  INVOKE_KERNEL(DEMOSAIC_NN, demosaic_nn_hw, host_input, host_result, row_size,
                col_size, chan_size, acc_input, acc_result);

  // Invoke the transform (white balancing & color mapping) kernel
  MAP_ARRAY_TO_ACCEL(TRANSFORM, "host_input", host_input,
                     sizeof(float) * row_size * col_size * chan_size);
  MAP_ARRAY_TO_ACCEL(TRANSFORM, "host_result", host_result,
                     sizeof(float) * row_size * col_size * chan_size);
  MAP_ARRAY_TO_ACCEL(TRANSFORM, "TsTw_tran", TsTw_tran,
                     sizeof(float) * row_size * col_size * chan_size);
  TsTw_tran = transpose_mat(TsTw, 3, 3);
  INVOKE_KERNEL(TRANSFORM, transform_hw, host_input, host_result, row_size,
                col_size, chan_size, acc_input, acc_result, TsTw_tran);

  free(acc_input);
  free(acc_result);
  free(Ts);
  free(Tw);
  free(TsTw);
  free(TsTw_tran);
  free(ctrl_pts);
  free(weights);
  free(coefs);
}

