#include <iostream>
#include <math.h>

#include "gamut_map.h"

void gamut_map(float* input,
               int row_size,
               int col_size,
               int chan_size,
               float* result,
               float* ctrl_pts,
               float* weights,
               float* coefs,
               int num_cps) {

    ARRAY_3D(float, _input, input, col_size, chan_size);
    ARRAY_3D(float, _result, result, col_size, chan_size);
    ARRAY_2D(float, _ctrl_pts, ctrl_pts, chan_size);
    ARRAY_2D(float, _weights, weights, chan_size);
    ARRAY_2D(float, _coefs, coefs, chan_size);

    float* l2_dist = new float[num_cps];
    for (int row = 0; row < row_size; row++) {
        for (int col = 0; col < col_size; col++) {
            for (int cp = 0; cp < num_cps; cp++) {
                l2_dist[cp] =
                        sqrt((_input[row][col][0] - _ctrl_pts[cp][0]) *
                                     (_input[row][col][0] - _ctrl_pts[cp][0]) +
                             (_input[row][col][1] - _ctrl_pts[cp][1]) *
                                     (_input[row][col][1] - _ctrl_pts[cp][1]) +
                             (_input[row][col][2] - _ctrl_pts[cp][2]) *
                                     (_input[row][col][2] - _ctrl_pts[cp][2]));
            }
            for (int chan = 0; chan < chan_size; chan++) {
                _result[row][col][chan] = 0.0;
                for (int cp = 0; cp < num_cps; cp++) {
                    _result[row][col][chan] += l2_dist[cp] * _weights[cp][chan];
                }
                // Add on the biases for the RBF
                _result[row][col][chan] +=
                        _coefs[0][chan] +
                        _coefs[1][chan] * _input[row][col][0] +
                        _coefs[2][chan] * _input[row][col][1] +
                        _coefs[3][chan] * _input[row][col][2];
            }
        }
    }
    delete l2_dist;
}

