#include <stdio.h>
#include <math.h>
#include "pipe_stages.h"
#include "utility/cam_pipe_utility.h"

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
void demosaic_fxp(float *input, int row_size, int col_size, float *result) {
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);

  dm_nn_row:
  for (int row = 1; row < row_size; row++)
    dm_nn_col:
    for (int col = 1; col < col_size; col++)
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

// Simple denoise
void denoise_fxp(float *input, int row_size, int col_size, float *result) {
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);

  dn_chan:
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    dm_row:
    for (int row = 0; row < row_size; row++)
      dm_col:
      for (int col = 0; col < col_size; col++)
        if (row >= 2 && row < row_size - 2 && col >= 2 && col < col_size - 2) {
          float a =
              max(max(_input[chan][row][col - 2], _input[chan][row][col + 2]),
                  max(_input[chan][row - 2][col], _input[chan][row + 2][col]));
          _result[chan][row][col] = max(min(_input[chan][row][col], a), 0);
        } else {
          _result[chan][row][col] = _input[chan][row][col];
        }
}

// Color map and white balance transform
void transform_fxp(float *input, int row_size, int col_size, float *result,
                   float *TsTw_tran) {
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
void gamut_map_fxp(float *input, int row_size, int col_size, float *result,
                   float *ctrl_pts, float *weights, float *coefs, float *l2_dist) {
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
void tone_map_fxp(float *input, int row_size, int col_size, float *tone_map,
                  float *result) {
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);
  ARRAY_2D(float, _tone_map, tone_map, 3);

  tm_chan:
  for (int chan = 0; chan < CHAN_SIZE; chan++)
    tm_row:
    for (int row = 0; row < row_size; row++)
      tm_col:
      for (int col = 0; col < col_size; col++) {
        int x = _input[chan][row][col] * 255;
        _result[chan][row][col] = _tone_map[x][chan];
      }
}

// Approximate tone mapping
void tone_map_approx_fxp(float *input, int row_size, int col_size,
                         float *result) {
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
