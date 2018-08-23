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
