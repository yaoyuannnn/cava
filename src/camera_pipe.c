#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "params.h"
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

///////////////////////////////////////////////////////////////
// Patches to test
///////////////////////////////////////////////////////////////

// Patch start locations
// [xstart, ystart]
//
// NOTE: These start locations must align with the demosiac
// pattern start if using the version of this pipeline with
// demosaicing

int patchstarts[12][2] = { { 551, 2751 },
                           { 1001, 2751 },
                           { 1501, 2751 },
                           { 2001, 2751 },
                           { 551, 2251 },
                           { 1001, 2251 },
                           { 1501, 2251 },
                           { 2001, 2251 },
                           { 551, 1751 },
                           { 1001, 1751 },
                           { 1501, 1751 },
                           { 2001, 1751 } };

// Height and width of patches
int patchsize = 10;

// Number of patches to test
int patchnum = 1;

void load_camera_params_hw(float *host_TsTw, float *host_ctrl_pts,
                           float *host_weights, float *host_coefs,
                           float *acc_TsTw, float *acc_ctrl_pts,
                           float *acc_weights, float *acc_coefs) {
  dmaLoad(acc_TsTw, host_TsTw, CHAN_SIZE * CHAN_SIZE * sizeof(float));
  dmaLoad(acc_ctrl_pts, host_ctrl_pts,
          num_ctrl_pts * CHAN_SIZE * sizeof(float));
  dmaLoad(acc_weights, host_weights, num_ctrl_pts * CHAN_SIZE * sizeof(float));
  dmaLoad(acc_coefs, host_coefs, CHAN_SIZE * CHAN_SIZE * sizeof(float));
}

void isp_hw(float *host_input, float *host_result, int row_size, int col_size,
            float *acc_input, float *acc_result, float *acc_TsTw,
            float *acc_ctrl_pts, float *acc_weights, float *acc_coefs) {
  dmaLoad(acc_input, host_input,
          row_size * col_size * CHAN_SIZE * sizeof(float));
  demosaic_nn_fxp(acc_input, row_size, col_size, CHAN_SIZE, acc_result);
  denoise_fxp(acc_input, row_size, col_size, CHAN_SIZE, acc_result);
  transform_fxp(acc_input, row_size, col_size, CHAN_SIZE, acc_result, acc_TsTw);
  gamut_map_fxp(acc_input, row_size, col_size, CHAN_SIZE, acc_result,
                acc_ctrl_pts, acc_weights, acc_coefs);
  tone_map_approx_fxp(acc_input, row_size, col_size, CHAN_SIZE, acc_result);
  dmaStore(host_result, acc_result,
           row_size * col_size * CHAN_SIZE * sizeof(float));
}

// void demosaic_nn_hw(float *host_input, float *host_result, int row_size,
//                    int col_size, float *acc_input, float *acc_result) {
//  dmaLoad(acc_input, host_input, row_size * col_size * CHAN_SIZE *
// sizeof(float));
//  demosaic_nn_fxp(acc_input, row_size, col_size, acc_result);
//  dmaStore(host_result, acc_result, row_size * col_size * CHAN_SIZE *
// sizeof(float));
//}
//
// void transform_hw(float *host_input, float *host_result, int row_size,
//                  int col_size, float *acc_input, float *acc_result,
//                  float *TsTw) {
//  dmaLoad(acc_input, host_input, row_size * col_size * CHAN_SIZE *
// sizeof(float));
//  transform_fxp(acc_input, row_size, col_size, acc_result, TsTw);
//  dmaStore(host_result, acc_result, row_size * col_size * CHAN_SIZE *
// sizeof(float));
//}

void camera_pipe(float *host_input, float *host_result, int row_size,
                 int col_size) {
  float *host_TsTw, *host_ctrl_pts, *host_weights, *host_coefs;
  float *acc_TsTw, *acc_ctrl_pts, *acc_weights, *acc_coefs;
  float *acc_input, *acc_result;

  host_TsTw = get_TsTw(cam_model_path, wb_index);
  host_TsTw = transpose_mat(host_TsTw, CHAN_SIZE, CHAN_SIZE);
  host_ctrl_pts = get_ctrl_pts(cam_model_path, num_ctrl_pts);
  host_weights = get_weights(cam_model_path, num_ctrl_pts);
  host_coefs = get_coefs(cam_model_path, num_ctrl_pts);

  int err = posix_memalign((void **)&acc_input, CACHELINE_SIZE,
                           sizeof(float) * row_size * col_size * CHAN_SIZE);
  err |= posix_memalign((void **)&acc_result, CACHELINE_SIZE,
                        sizeof(float) * row_size * col_size * CHAN_SIZE);
  err |= posix_memalign((void **)&acc_TsTw, CACHELINE_SIZE,
                        sizeof(float) * CHAN_SIZE * CHAN_SIZE);
  err |= posix_memalign((void **)&acc_ctrl_pts, CACHELINE_SIZE,
                        sizeof(float) * num_ctrl_pts * CHAN_SIZE);
  err |= posix_memalign((void **)&acc_weights, CACHELINE_SIZE,
                        sizeof(float) * num_ctrl_pts * CHAN_SIZE);
  err |= posix_memalign((void **)&acc_coefs, CACHELINE_SIZE,
                        sizeof(float) * CHAN_SIZE * CHAN_SIZE);
  assert(err == 0 && "Failed to allocate memory!");

  // Load camera model parameters for the ISP
  MAP_ARRAY_TO_ACCEL(ISP, "host_TsTw", host_TsTw,
                     sizeof(float) * CHAN_SIZE * CHAN_SIZE);
  MAP_ARRAY_TO_ACCEL(ISP, "host_ctrl_pts", host_ctrl_pts,
                     sizeof(float) * num_ctrl_pts * CHAN_SIZE);
  MAP_ARRAY_TO_ACCEL(ISP, "host_weights", host_weights,
                     sizeof(float) * num_ctrl_pts * CHAN_SIZE);
  MAP_ARRAY_TO_ACCEL(ISP, "host_coefs", host_coefs,
                     sizeof(float) * CHAN_SIZE * CHAN_SIZE);
  INVOKE_KERNEL(ISP, load_camera_params_hw, host_TsTw, host_ctrl_pts,
                host_weights, host_coefs, acc_TsTw, acc_ctrl_pts, acc_weights,
                acc_coefs);

  // Invoke the ISP
  MAP_ARRAY_TO_ACCEL(ISP, "host_input", host_input,
                     sizeof(float) * row_size * col_size * CHAN_SIZE);
  MAP_ARRAY_TO_ACCEL(ISP, "host_result", host_result,
                     sizeof(float) * row_size * col_size * CHAN_SIZE);
  INVOKE_KERNEL(ISP, isp_hw, host_input, host_result, row_size, col_size,
                acc_input, acc_result, acc_TsTw, acc_ctrl_pts, acc_weights,
                acc_coefs);

  free(acc_input);
  free(acc_result);
  free(host_TsTw);
  free(host_ctrl_pts);
  free(host_weights);
  free(host_coefs);
  free(acc_TsTw);
  free(acc_ctrl_pts);
  free(acc_weights);
  free(acc_coefs);
}

