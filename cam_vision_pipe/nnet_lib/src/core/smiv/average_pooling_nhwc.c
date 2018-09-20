#include <stdio.h>

#include "config.h"
#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"
#include "core/smiv/impls.h"

// Get the scaling factor for the averaging operation.
//
// Assuming a square pooling region of size s, this returns 1/s^2.  This
// contains precomputed scale factors for sizes up to 8.
const float get_avg_scale(int size) {
    return size == 1 ? 1.0 :
           size == 2 ? 1.0 / 4 :
           size == 3 ? 1.0 / 9 :
           size == 4 ? 1.0 / 16 :
           size == 5 ? 1.0 / 25 :
           size == 6 ? 1.0 / 36 :
           size == 7 ? 1.0 / 49 :
                       1.0 / 64;
}

// An average-pooling operation on SMIV.
//
// This requires a blocked channel data format (GNHWC), where G = channels/8,
// and the last dimension = chans = 8. The last dimension MUST be 8.
// This supports arbitrary pooling sizes and strides.
//
// Args:
//   inputs: A pointer to the input buffer.
//   curr_layer: A description of this pooling layer. Note that the
//     input/dimensions are still described logically as NCHW (e.g.
//     layer.input.rows = actual number of rows). The number of channels need
//     not be a multiple of 8; prior to calling this function the data shoud
//     have been converted into NHWC format, and that conversion will take care
//     of the required alignment.
//   results: A pointer to the output buffer.
void avgpooling_nhwc_smiv_fxp(float* inputs,
                              layer_t curr_layer,
                              int input_start_chan,
                              float* results) {
    const int a_rows = curr_layer.inputs.rows;
    const int a_cols = curr_layer.inputs.cols;
    const int a_chan_groups = FRAC_CEIL(curr_layer.inputs.height, VECTOR_SIZE);
    const int result_rows = curr_layer.outputs.rows;
    const int result_cols = curr_layer.outputs.cols;

    const int pool_size = curr_layer.weights.cols;
    const int row_stride = curr_layer.stride.rows;
    const int col_stride = curr_layer.stride.cols;

    const int end_row = a_rows - pool_size + 1;
    const int end_col = a_cols - pool_size + 1;

    const float scale = get_avg_scale(pool_size);

    ARRAY_4D(float, _a, inputs, a_rows, a_cols, VECTOR_SIZE);
    ARRAY_4D(float, _results, results, result_rows, result_cols, VECTOR_SIZE);

    avgpool_chan_grp:
    for (int chan_grp = 0; chan_grp < a_chan_groups; chan_grp++) {
        int out_row = 0;
        avgpool_chan_input_row:
        for (int row = 0; row < end_row; row += row_stride) {
            int out_col = 0;
            avgpool_chan_input_col:
            for (int col = 0; col < end_col; col += col_stride) {
                float curr_results[VECTOR_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0};
                avgpool_pool_row:
                for (int pool_i = 0; pool_i < pool_size; pool_i++) {
                    avgpool_pool_col:
                    for (int pool_j = 0; pool_j < pool_size; pool_j++) {
                        avgpool_load_chan_px:
                        for (int px = 0; px < VECTOR_SIZE; px++) {
                            curr_results[px] += _a[chan_grp][row + pool_i]
                                                  [col + pool_j][px];
                        }
                    }
                }
                // Commit, taking care of the scaling factor along the way.
                avgpool_scale_and_commit:
                for (int commit_i = 0; commit_i < VECTOR_SIZE; commit_i++) {
                    _results[chan_grp][out_row][out_col][commit_i] =
                        curr_results[commit_i] * scale;
                }
                out_col++;
            }
            out_row++;
        }
    }
}

void avgpooling_nhwc_smiv_simd_fxp(float* inputs,
                                   layer_t curr_layer,
                                   int input_start_chan,
                                   float* results) {
    const int a_rows = curr_layer.inputs.rows;
    const int a_cols = curr_layer.inputs.cols;
    const int a_chan_groups = FRAC_CEIL(curr_layer.inputs.height, VECTOR_SIZE);
    const int result_rows = curr_layer.outputs.rows;
    const int result_cols = curr_layer.outputs.cols;

    const int pool_size = curr_layer.weights.cols;
    const int row_stride = curr_layer.stride.rows;
    const int col_stride = curr_layer.stride.cols;

    const int end_row = a_rows - pool_size + 1;
    const int end_col = a_cols - pool_size + 1;

    const float scale = get_avg_scale(pool_size);
    const v8fp_t scale_vec = { scale, scale, scale, scale,
                               scale, scale, scale, scale };

    VEC_ARRAY_4D(v8fp_t, _a, inputs, a_rows, a_cols, VECTOR_SIZE);
    VEC_ARRAY_4D(
            v8fp_t, _results, results, result_rows, result_cols, VECTOR_SIZE);
    VEC_ARRAY_1D(v8fp_t, _results_flat, results);

    avgpool_chan_grp:
    for (int chan_grp = 0; chan_grp < a_chan_groups; chan_grp++) {
        int out_row = 0;
        avgpool_chan_input_row:
        for (int row = 0; row < end_row; row += row_stride) {
            int out_col = 0;
            avgpool_chan_input_col:
            for (int col = 0; col < end_col; col += col_stride) {
                v8fp_t curr_results = {0, 0, 0, 0, 0, 0, 0, 0};
                // Optimization: precompute the results location.
                // Aladdin doesn't do well with optimizations that move across
                // the end of loop boundaries.
                int result_idx = &_results[chan_grp][out_row][out_col][0] -
                                 &_results[0][0][0][0];
                out_col++;
                avgpool_pool_row:
                for (int pool_i = 0; pool_i < pool_size; pool_i++) {
                    avgpool_pool_col:
                    for (int pool_j = 0; pool_j < pool_size; pool_j++) {
                        curr_results +=
                                _a[chan_grp][row + pool_i][col + pool_j][0];
                    }
                }
                // Scale and commit.
                _results_flat[result_idx] = curr_results * scale_vec;
            }
            out_row++;
        }
    }
}

void avgpooling_nhwc_smiv(float* inputs,
                          layer_t curr_layer,
                          int input_start_chan,
                          float* results) {
#ifdef ENABLE_SIMD_IMPL
    avgpooling_nhwc_smiv_simd_fxp(
            inputs, curr_layer, input_start_chan, results);
#else
    avgpooling_nhwc_smiv_fxp(inputs, curr_layer, input_start_chan, results);
#endif
}
