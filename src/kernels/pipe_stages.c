#include "pipe_stages.h"

// Nearest neighbor demosaicing
// G R
// B G
void demosaic_nn_fxp(float *input, int row_size, int col_size, int chan_size,
                     float *result) {
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);

  dm_nn_row:
  for (int row = 0; row < row_size; row++)
    dm_nn_col:
    for (int col = 0; col < col_size; col++)
      dm_nn_chan:
      for (int chan = 0; chan < chan_size; chan++)
        if (row % 2 == 0 && col % 2 == 0) {
          if (chan == 0) {
            _result[chan][row][col] = _input[chan+1][row][col];
          } else if (chan == 1) {
            _result[chan][row][col] = _input[chan][row][col] * 2;
          } else {
            _result[chan][row][col] = _input[chan][row+1][col];
          }
        } else if (row % 2 == 0 && col % 2 == 1) {
          if (chan == 0) {
            _result[chan][row][col] = _input[chan][row][col];
          } else if (chan == 1) {
            _result[chan][row][col] = _input[chan][row][col-1] * 2;
          } else {
            _result[chan][row][col] = _input[chan][row+1][col-1];
          }
        } else if (row % 2 == 1 && col % 2 == 0) {
          if (chan == 0) {
            _result[chan][row][col] = _input[chan][row-1][col+1];
          } else if (chan == 1) {
            _result[chan][row][col] = _input[chan][row][col-1] * 2;
          } else {
            _result[chan][row][col] = _input[chan][row][col];
          }
        } else {
          if (chan == 0) {
            _result[chan][row][col] = _input[chan][row-1][col];
          } else if (chan == 1) {
            _result[chan][row][col] = _input[chan][row][col] * 2;
          } else {
            _result[chan][row][col] = _input[chan][row][col-1];
          }
        }
}

// Color map and white balance transform
void transform_fxp(float *input, int row_size, int col_size, int chan_size,
                    float *result, float *TsTw_tran) {
  ARRAY_3D(float, _input, input, row_size, col_size);
  ARRAY_3D(float, _result, result, row_size, col_size);
  ARRAY_2D(float, _TsTw_tran, TsTw_tran, 3);

  tr_row:
  for (int row = 0; row < row_size; row++)
    tr_col:
    for (int col = 0; col < col_size; col++)
      tr_chan:
      for (int chan = 0; chan < chan_size; chan++)
        if (chan == 0) {
          _result[chan][row][col] =
              max(_input[0][row][col] * _TsTw_tran[0][0] +
                      _input[1][row][col] * _TsTw_tran[1][0] +
                      _input[2][row][col] * _TsTw_tran[2][0],
                  0);
        } else if (chan == 1) {
          _result[chan][row][col] =
              max(_input[0][row][col] * _TsTw_tran[0][1] +
                  _input[1][row][col] * _TsTw_tran[1][1] + 
                  _input[2][row][col] * _TsTw_tran[2][1], 0);
        } else {
          max(_input[0][row][col] * _TsTw_tran[0][2] +
                  _input[1][row][col] * _TsTw_tran[1][2] +
                  _input[2][row][col] * _TsTw_tran[2][2],
              0);
        }
}

// Gamut mapping
//void gamut_map_fxp() {
//  // Weighted radial basis function for gamut mapping
//  Func rbf_ctrl_pts("rbf_ctrl_pts");
//    // Initialization with all zero
//    rbf_ctrl_pts(x,y,c) = cast<float>(0);
//    // Index to iterate with
//    RDom idx(0,num_ctrl_pts);
//    // Loop code
//    // Subtract the vectors
//    Expr red_sub   = (*in_func)(x,y,0) - (*ctrl_pts_h)(0,idx);
//    Expr green_sub = (*in_func)(x,y,1) - (*ctrl_pts_h)(1,idx);
//    Expr blue_sub  = (*in_func)(x,y,2) - (*ctrl_pts_h)(2,idx);
//    // Take the L2 norm to get the distance
//    Expr dist      = sqrt( red_sub*red_sub +
//                              green_sub*green_sub +
//                              blue_sub*blue_sub );
//    // Update persistant loop variables
//    rbf_ctrl_pts(x,y,c) = select( c == 0, rbf_ctrl_pts(x,y,c) +
//                                    ( (*weights_h)(0,idx) * dist),
//                                  c == 1, rbf_ctrl_pts(x,y,c) +
//                                    ( (*weights_h)(1,idx) * dist),
//                                          rbf_ctrl_pts(x,y,c) +
//                                    ( (*weights_h)(2,idx) * dist));
//
//  // Add on the biases for the RBF
//  Func rbf_biases("rbf_biases");
//    rbf_biases(x,y,c) = max( select(
//      c == 0, (*rbf_ctrl_pts)(x,y,0)     + (*coefs)[0][0] + (*coefs)[1][0]*(*in_func)(x,y,0) +
//        (*coefs)[2][0]*(*in_func)(x,y,1) + (*coefs)[3][0]*(*in_func)(x,y,2),
//      c == 1, (*rbf_ctrl_pts)(x,y,1)     + (*coefs)[0][1] + (*coefs)[1][1]*(*in_func)(x,y,0) +
//        (*coefs)[2][1]*(*in_func)(x,y,1) + (*coefs)[3][1]*(*in_func)(x,y,2),
//              (*rbf_ctrl_pts)(x,y,2)     + (*coefs)[0][2] + (*coefs)[1][2]*(*in_func)(x,y,0) +
//        (*coefs)[2][2]*(*in_func)(x,y,1) + (*coefs)[3][2]*(*in_func)(x,y,2))
//                            , 0);
//}
//
//// Tone mapping
//void tone_map_fxp( Func *in_func,
//                    Image<float> *rev_tone_h ) {
//  Var x, y, c;
//  // Forward tone mapping
//  Func tone_map("tone_map");
//    RDom idx2(0,256);
//    // Theres a lot in this one line! This line finds the entry in
//    // the reverse tone mapping function which is closest to this Func's
//    // input. It then scales back down to the 0-1 range expected as the
//    // output for every stage. Effectively it reverses the reverse
//    // tone mapping function.
//    tone_map(x,y,c) = (argmin( abs( (*rev_tone_h)(c,idx2)
//                                 - (*in_func)(x,y,c) ) )[0])/256.0f;
//}
