#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utility/utility.h"
#include "kernels/pipe_stages.h"
#include "camera_pipe.h"

int main(int argc, char* argv[]) {
  uint8_t *host_input = NULL;
  uint8_t *host_input_nwc = NULL;
  uint8_t *host_result = NULL;
  uint8_t *host_result_nwc = NULL;
  int row_size, col_size;

  // Read a raw image
  printf("Reading a raw image from the binary file\n");
  host_input_nwc = read_image_from_binary(argv[1], &row_size, &col_size);
  printf("Raw image shape: %d x %d x %d\n", row_size, col_size,
         CHAN_SIZE);
  // The input image is stored in HWC format. To make it more efficient for
  // future optimization (e.g., vectorization), I expect we would transform it
  // to CHW format at some point.
  convert_hwc_to_chw(host_input_nwc, row_size, col_size, &host_input);

  // Allocate a buffer for storing the output image data.
  host_result = malloc_aligned(sizeof(uint8_t) * row_size * col_size * CHAN_SIZE);

  // Invoke the camera pipeline
  camera_pipe(host_input, host_result, row_size, col_size);

  // Transform the output image back to HWC format.
  convert_chw_to_hwc(host_result, row_size, col_size, &host_result_nwc);

  // Output the image
  printf("Writing output image to %s\n", argv[2]);
  write_image_to_binary(argv[2], host_result_nwc, row_size, col_size);

  free(host_input);
  free(host_input_nwc);
  free(host_result);
  free(host_result_nwc);
  return 0;
}

