#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "kernels/pipe_stages.h"
#include "utility/load_camera_model.h"
#include "utility/utility.h"
#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

///////////////////////////////////////////////////////////////
// Camera Model Parameters
///////////////////////////////////////////////////////////////

// Path to the camera model to be used
char cam_model_path[] = "cam_models/NikonD7000/";

// White balance index (select white balance from transform file)
// The first white balance in the file has a wb_index of 1
// For more information on model format see the readme
int wb_index = 6;

// Number of control points
int num_ctrl_pts = 3702;

void load_camera_params_hw(float *host_TsTw, float *host_ctrl_pts,
                           float *host_weights, float *host_coefs,
                           float *acc_TsTw, float *acc_ctrl_pts,
                           float *acc_weights, float *acc_coefs) {
  dmaLoad(acc_TsTw, host_TsTw, 9 * sizeof(float));
  dmaLoad(acc_ctrl_pts, host_ctrl_pts,
          num_ctrl_pts * CHAN_SIZE * sizeof(float));
  dmaLoad(acc_weights, host_weights, num_ctrl_pts * CHAN_SIZE * sizeof(float));
  dmaLoad(acc_coefs, host_coefs, 12 * sizeof(float));
}

void isp_hw(uint8_t *host_input, uint8_t *host_result, int row_size,
            int col_size, uint8_t *acc_input, uint8_t *acc_result,
            float *acc_input_scaled, float *acc_result_scaled, float *acc_TsTw,
            float *acc_ctrl_pts, float *acc_weights, float *acc_coefs,
            float *acc_l2_dist) {
  dmaLoad(acc_input, host_input,
          row_size * col_size * CHAN_SIZE * sizeof(uint8_t));
  scale_fxp(acc_input, row_size, col_size, acc_input_scaled);
  demosaic_fxp(acc_input_scaled, row_size, col_size, acc_result_scaled);
  denoise_fxp(acc_result_scaled, row_size, col_size, acc_input_scaled);
  transform_fxp(acc_input_scaled, row_size, col_size, acc_result_scaled,
                acc_TsTw);
  gamut_map_fxp(acc_result_scaled, row_size, col_size, acc_input_scaled,
                acc_ctrl_pts, acc_weights, acc_coefs, acc_l2_dist);
  tone_map_approx_fxp(acc_input_scaled, row_size, col_size, acc_result_scaled);
  descale_fxp(acc_result_scaled, row_size, col_size, acc_result);
  dmaStore(host_result, acc_result,
           row_size * col_size * CHAN_SIZE * sizeof(uint8_t));
}

void camera_pipe(uint8_t *host_input, uint8_t *host_result, int row_size,
                 int col_size) {
  uint8_t *acc_input, *acc_result;
  float *acc_input_scaled, *acc_result_scaled;
  float *host_TsTw, *host_ctrl_pts, *host_weights, *host_coefs;
  float *acc_TsTw, *acc_ctrl_pts, *acc_weights, *acc_coefs, *acc_l2_dist;

  host_TsTw = get_TsTw(cam_model_path, wb_index);
  float *trans = transpose_mat(host_TsTw, CHAN_SIZE, CHAN_SIZE);
  free(host_TsTw);
  host_TsTw = trans;
  host_ctrl_pts = get_ctrl_pts(cam_model_path, num_ctrl_pts);
  host_weights = get_weights(cam_model_path, num_ctrl_pts);
  host_coefs = get_coefs(cam_model_path, num_ctrl_pts);

  acc_input = malloc_aligned(sizeof(uint8_t) * row_size * col_size * CHAN_SIZE);
  acc_result = malloc_aligned(sizeof(uint8_t) * row_size * col_size * CHAN_SIZE);
  acc_input_scaled = malloc_aligned(sizeof(float) * row_size * col_size * CHAN_SIZE);
  acc_result_scaled = malloc_aligned(sizeof(float) * row_size * col_size * CHAN_SIZE);
  acc_TsTw = malloc_aligned(sizeof(float) * 9);
  acc_ctrl_pts = malloc_aligned(sizeof(float) * num_ctrl_pts * CHAN_SIZE);
  acc_weights = malloc_aligned(sizeof(float) * num_ctrl_pts * CHAN_SIZE);
  acc_coefs = malloc_aligned(sizeof(float) * 12);
  acc_l2_dist = malloc_aligned(sizeof(float) * num_ctrl_pts);

  // Load camera model parameters for the ISP
  MAP_ARRAY_TO_ACCEL(ISP, "host_TsTw", host_TsTw,
                     sizeof(float) * 9);
  MAP_ARRAY_TO_ACCEL(ISP, "host_ctrl_pts", host_ctrl_pts,
                     sizeof(float) * num_ctrl_pts * CHAN_SIZE);
  MAP_ARRAY_TO_ACCEL(ISP, "host_weights", host_weights,
                     sizeof(float) * num_ctrl_pts * CHAN_SIZE);
  MAP_ARRAY_TO_ACCEL(ISP, "host_coefs", host_coefs,
                     sizeof(float) * 12);
  INVOKE_KERNEL(ISP, load_camera_params_hw, host_TsTw, host_ctrl_pts,
                host_weights, host_coefs, acc_TsTw, acc_ctrl_pts, acc_weights,
                acc_coefs);

  // Invoke the ISP
  MAP_ARRAY_TO_ACCEL(ISP, "host_input", host_input,
                     sizeof(uint8_t) * row_size * col_size * CHAN_SIZE);
  MAP_ARRAY_TO_ACCEL(ISP, "host_result", host_result,
                     sizeof(uint8_t) * row_size * col_size * CHAN_SIZE);
  INVOKE_KERNEL(ISP, isp_hw, host_input, host_result, row_size, col_size,
                acc_input, acc_result, acc_input_scaled, acc_result_scaled,
                acc_TsTw, acc_ctrl_pts, acc_weights, acc_coefs, acc_l2_dist);

  free(acc_input);
  free(acc_result);
  free(acc_input_scaled);
  free(acc_result_scaled);
  free(host_TsTw);
  free(host_ctrl_pts);
  free(host_weights);
  free(host_coefs);
  free(acc_TsTw);
  free(acc_ctrl_pts);
  free(acc_weights);
  free(acc_coefs);
  free(acc_l2_dist);
}

