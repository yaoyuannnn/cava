#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utility/utility.h"
#include "params.h"
#include "camera_pipe.h"

int main() {
  const int row_size = 128;
  const int col_size = 128;

  float *host_input, *host_result;
  int err = posix_memalign((void**)&host_input, CACHELINE_SIZE,
                 sizeof(float) * row_size * col_size * CHAN_SIZE);
  err |= posix_memalign((void**)&host_result, CACHELINE_SIZE,
                 sizeof(float) * row_size * col_size * CHAN_SIZE);
  assert(err == 0 && "Failed to allocate memory!");
  for (int i = 0; i < row_size * col_size * CHAN_SIZE; i++) {
    host_input[i] = 1.0;
  }

  camera_pipe(host_input, host_result, row_size, col_size);

  FILE *output_file = fopen("result.txt", "w");
  for (int i = 0; i < row_size * col_size * CHAN_SIZE; i++)
    fprintf(output_file, "%f ", host_result[i]);
  fclose(output_file);

  free(host_input);
  free(host_result);
  return 0;
}

