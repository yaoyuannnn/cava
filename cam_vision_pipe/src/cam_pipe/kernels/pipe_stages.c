#include <stdio.h>
#include <math.h>
#include "pipe_stages.h"
#include "utility/cam_pipe_utility.h"

ALWAYS_INLINE
void scale_fxp(uint8_t *input, int row_size, int col_size, float *output) {
  ARRAY_3D(uint8_t, _input, input, row_size, col_size);
  ARRAY_3D(float, _output, output, row_size, col_size);
  sl_chan:
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    sl_row:
    for (int row = 0; row < row_size; row++)
      sl_col:
      for (int col = 0; col < col_size; col++)
        _output[chan][row][col] = _input[chan][row][col] * 1.0 / 255;
}

ALWAYS_INLINE
void descale_fxp(float *input, int row_size, int col_size, uint8_t *output) {
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(uint8_t, _output, output, row_size, col_size);
  dsl_chan:
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    dsl_row:
    for (int row = 0; row < row_size; row++)
      dsl_col:
      for (int col = 0; col < col_size; col++)
        _output[chan][row][col] = min(max(_input[chan][row][col] * 255, 0), 255);
}

// Demosaicing stage
// G R
// B G
ALWAYS_INLINE
void demosaic_fxp(float *input, int row_size, int col_size, float *result) {
  PRINT("Demosaicing.\n");
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);

  dm_row:
  for (int row = 1; row < row_size - 1; row++)
    dm_col:
    for (int col = 1; col < col_size - 1; col++)
        if (row % 2 == 0 && col % 2 == 0) {
            // Green pixel
            // Getting the R values
            float R1 = _input[0][row][col - 1];
            float R2 = _input[0][row][col + 1];
            // Getting the B values
            float B1 = _input[2][row - 1][col];
            float B2 = _input[2][row + 1][col];
            // R
            _result[0][row][col] = (R1 + R2) / 2;
            // G
            _result[1][row][col] = _input[1][row][col] * 2;
            // B
            _result[2][row][col] = (B1 + B2) / 2;
        } else if (row % 2 == 0 && col % 2 == 1) {
            // Red pixel
            // Getting the G values
            float G1 = _input[1][row - 1][col];
            float G2 = _input[1][row + 1][col];
            float G3 = _input[1][row][col - 1];
            float G4 = _input[1][row][col + 1];
            // Getting the B values
            float B1 = _input[2][row - 1][col - 1];
            float B2 = _input[2][row - 1][col + 1];
            float B3 = _input[2][row + 1][col - 1];
            float B4 = _input[2][row + 1][col + 1];
            // R
            _result[0][row][col] = _input[0][row][col];
            // G
            _result[1][row][col] = (G1 + G2 + G3 + G4) / 2;
            // B (center pixel)
            _result[2][row][col] = (B1 + B2 + B3 + B4) / 4;
        } else if (row % 2 == 1 && col % 2 == 0) {
            // Blue pixel
            // Getting the R values
            float R1 = _input[0][row - 1][col - 1];
            float R2 = _input[0][row + 1][col - 1];
            float R3 = _input[0][row - 1][col + 1];
            float R4 = _input[0][row + 1][col + 1];
            // Getting the G values
            float G1 = _input[1][row - 1][col];
            float G2 = _input[1][row + 1][col];
            float G3 = _input[1][row][col - 1];
            float G4 = _input[1][row][col + 1];
            // R
            _result[0][row][col] = (R1 + R2 + R3 + R4) / 4;
            // G
            _result[1][row][col] = (G1 + G2 + G3 + G4) / 2;
            // B
            _result[2][row][col] = _input[2][row][col];
        } else {
            // Bottom Green pixel
            // Getting the R values
            float R1 = _input[0][row - 1][col];
            float R2 = _input[0][row + 1][col];
            // Getting the B values
            float B1 = _input[2][row][col - 1];
            float B2 = _input[2][row][col + 1];
            // R
            _result[0][row][col] = (R1 + R2) / 2;
            // G
            _result[1][row][col] = _input[1][row][col] * 2;
            // B
            _result[2][row][col] = (B1 + B2) / 2;
        }
}

ALWAYS_INLINE
static void sort(float arr[], int n) {
    int i, j;
    dn_sort_i:
    for (i = 0; i < n - 1; i++)
        dn_sort_j:
        for (j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) {
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
}

// Simple denoise
ALWAYS_INLINE
void denoise_fxp(float *input, int row_size, int col_size, float *result) {
  PRINT("Denoising.\n");
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);

  dn_chan:
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    dn_row:
    for (int row = 0; row < row_size; row++)
      dn_col:
      for (int col = 0; col < col_size; col++)
        if (row >= 1 && row < row_size - 1 && col >= 1 && col < col_size - 1) {
          float filter[9];
          dn_slide_row:
          for (int i = row-1; i < row+2; i++)
            dn_slide_col:
            for (int j = col-1; j < col+2; j++) {
              int index = (i - row + 1) * 3 + j - col + 1;
              filter[index] = _input[chan][i][j];
            }
          sort(filter, 9);
          _result[chan][row][col] = filter[4];
        } else {
          _result[chan][row][col] = _input[chan][row][col];
        }
}

// Color map and white balance transform
ALWAYS_INLINE
void transform_fxp(float *input, int row_size, int col_size, float *result,
                   float *TsTw_tran) {
  PRINT("Color mapping.\n");
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);
  ARRAY_2D(float, _TsTw_tran, TsTw_tran, 3);

  tr_chan:
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    tr_row:
    for (int row = 0; row < row_size; row++)
      tr_col:
      for (int col = 0; col < col_size; col++)
        _result[chan][row][col] =
            max(_input[0][row][col] * _TsTw_tran[0][chan] +
                    _input[1][row][col] * _TsTw_tran[1][chan] +
                    _input[2][row][col] * _TsTw_tran[2][chan],
                0);
}

//
// Weighted radial basis function for gamut mapping
//
ALWAYS_INLINE
void gamut_map_fxp(float *input, int row_size, int col_size, float *result,
                   float *ctrl_pts, float *weights, float *coefs, float *l2_dist) {
  PRINT("Gamut mapping.\n");
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);
  ARRAY_2D(float, _ctrl_pts, ctrl_pts, 3);
  ARRAY_2D(float, _weights, weights, 3);
  ARRAY_2D(float, _coefs, coefs, 3);

  // First, get the L2 norm from every pixel to the control points,
  // Then, sum it and weight it. Finally, add the bias.
  gm_rbf_row:
  for (int row = 0; row < row_size; row++)
    gm_rbf_col:
    for (int col = 0; col < col_size; col++) {
      gm_rbf_cp0:
      for (int cp = 0; cp < num_ctrl_pts; cp++) {
        l2_dist[cp] =
            sqrt((_input[0][row][col] - _ctrl_pts[cp][0]) *
                     (_input[0][row][col] - _ctrl_pts[cp][0]) +
                 (_input[1][row][col] - _ctrl_pts[cp][1]) *
                     (_input[1][row][col] - _ctrl_pts[cp][1]) +
                 (_input[2][row][col] - _ctrl_pts[cp][2]) *
                     (_input[2][row][col] - _ctrl_pts[cp][2]));
      }
      gm_rbf_chan:
      for (int chan = 0; chan < CHAN_SIZE; chan++) {
        _result[chan][row][col] = 0.0;
        gm_rbf_cp1:
        for (int cp = 0; cp < num_ctrl_pts; cp++) {
          _result[chan][row][col] +=
              l2_dist[cp] * _weights[cp][chan];
        }
        // Add on the biases for the RBF
        _result[chan][row][col] += _coefs[0][chan] +
                                   _coefs[1][chan] * _input[0][row][col] +
                                   _coefs[2][chan] * _input[1][row][col] +
                                   _coefs[3][chan] * _input[2][row][col];
      }
    }
}

// Tone mapping
ALWAYS_INLINE
void tone_map_fxp(float *input, int row_size, int col_size, float *tone_map,
                  float *result) {
  PRINT("Tone mapping.\n");
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);
  ARRAY_2D(float, _tone_map, tone_map, 3);

  tm_chan:
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    tm_row:
    for (int row = 0; row < row_size; row++)
      tm_col:
      for (int col = 0; col < col_size; col++) {
        uint8_t x = _input[chan][row][col] * 255;
        _result[chan][row][col] = _tone_map[x][chan];
      }
}

// Approximate tone mapping
ALWAYS_INLINE
void tone_map_approx_fxp(float *input, int row_size, int col_size,
                         float *result) {
  PRINT("Approximate tone mapping.\n");
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);

  tm_apx_chan:
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    tm_apx_row:
    for (int row = 0; row < row_size; row++)
      tm_apx_col:
      for (int col = 0; col < col_size; col++) {
        if (_input[chan][row][col] < 32)
          _result[chan][row][col] = _input[chan][row][col] * 4;
        else if (_input[chan][row][col] < 128)
          _result[chan][row][col] = _input[chan][row][col] + 96;
        else
          _result[chan][row][col] = _input[chan][row][col] / 4 + 192;
        _result[chan][row][col] = max(min(_result[chan][row][col], 255), 0);
      }
}

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
                 float* acc_l2_dist) {
    float* input_scaled_internal;
    float* result_scaled_internal;
    input_scaled_internal = acc_input_scaled;
    result_scaled_internal = acc_result_scaled;

    scale_fxp(acc_input, row_size, col_size, acc_input_scaled);
    demosaic_fxp(
            input_scaled_internal, row_size, col_size, result_scaled_internal);
    SWAP_PTRS(input_scaled_internal, result_scaled_internal);
    denoise_fxp(
            input_scaled_internal, row_size, col_size, result_scaled_internal);
    SWAP_PTRS(input_scaled_internal, result_scaled_internal);
    transform_fxp(input_scaled_internal,
                  row_size,
                  col_size,
                  result_scaled_internal,
                  acc_TsTw);
    SWAP_PTRS(input_scaled_internal, result_scaled_internal);
    gamut_map_fxp(input_scaled_internal,
                  row_size,
                  col_size,
                  result_scaled_internal,
                  acc_ctrl_pts,
                  acc_weights,
                  acc_coefs,
                  acc_l2_dist);
    SWAP_PTRS(input_scaled_internal, result_scaled_internal);
    tone_map_fxp(input_scaled_internal,
                 row_size,
                 col_size,
                 acc_tone_map,
                 result_scaled_internal);
    SWAP_PTRS(input_scaled_internal, result_scaled_internal);
    descale_fxp(input_scaled_internal, row_size, col_size, acc_result);
}
