#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utility/utility.h"
#include "params.h"
#include "camera_pipe.h"

int main(int argc, char* argv[]) {
  float *host_input, *host_result;
  int row_size, col_size;

  // Read a raw image
  printf("Reading a raw image from the binary file\n");
  host_input = read_image_from_binary(argv[1], &row_size, &col_size);
  printf("Raw image shape: %d x %d x %d\n", row_size, col_size,
         CHAN_SIZE);
  int err = posix_memalign((void **)&host_result, CACHELINE_SIZE,
                           sizeof(float) * row_size * col_size * CHAN_SIZE);
  assert(err == 0 && "Failed to allocate memory!");

  // Invoke the camera pipeline
  camera_pipe(host_input, host_result, row_size, col_size);

  FILE *output_file = fopen(argv[2], "w");
  printf("Writing output image to %s\n", argv[2]);
  write_image_to_binary(argv[2], host_result, row_size, col_size);

  free(host_input);
  free(host_result);
  return 0;
}

