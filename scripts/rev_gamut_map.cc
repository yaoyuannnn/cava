#include <iostream>
#include <math.h>

#include "rev_gamut_map.h"


void rev_gamut_map(float* input,
                   int row_size,
                   int col_size,
                   int chan_size,
                   float* result,
                   float* ctrl_pts,
                   float* weights,
                   float* coefs,
                   int num_cps
                   ) {

  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);
  ARRAY_2D(float, _ctrl_pts, ctrl_pts, chan_size);
  ARRAY_2D(float, _weights, weights, chan_size);
  ARRAY_2D(float, _coefs, coefs, chan_size);

  float* l2_dist = new float[num_cps];
  for (int row = 0; row < row_size; row++) {
    for (int col = 0; col < col_size; col++) {
      for (int cp = 0; cp < num_cps; cp++) {
        l2_dist[cp] =
            sqrt((_input[0][row][col] - _ctrl_pts[cp][0]) *
                     (_input[0][row][col] - _ctrl_pts[cp][0]) +
                 (_input[1][row][col] - _ctrl_pts[cp][1]) *
                     (_input[1][row][col] - _ctrl_pts[cp][1]) +
                 (_input[2][row][col] - _ctrl_pts[cp][2]) *
                     (_input[2][row][col] - _ctrl_pts[cp][2]));
      }
      for (int chan = 0; chan < chan_size; chan++) {
        _result[chan][row][col] = 0.0;
        for (int cp = 0; cp < num_cps; cp++) {
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
  delete l2_dist;
}

