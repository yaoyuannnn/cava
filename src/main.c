#include <stdio.h>
#include <stdlib.h>
#include "kernels/pipe_stages.h"
#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#define DEMOSAIC_NN 0x3

// This is to avoid a ton of spurious unused variable warnings when
// we're not building for gem5.
#define UNUSED(x) (void)(x)

#ifdef GEM5_HARNESS

#define MAP_ARRAY_TO_ACCEL(req_code, name, base_addr, size)                    \
    mapArrayToAccelerator(req_code, name, base_addr, size)
#define INVOKE_KERNEL(req_code, kernel_ptr, args...)                           \
    do {                                                                       \
        UNUSED(kernel_ptr);                                                    \
        invokeAcceleratorAndBlock(req_code);                                   \
    } while (0)
#else

#define MAP_ARRAY_TO_ACCEL(req_code, name, base_addr, size)                    \
    do {                                                                       \
        UNUSED(req_code);                                                      \
        UNUSED(name);                                                          \
        UNUSED(base_addr);                                                     \
        UNUSED(size);                                                          \
    } while (0)
#define INVOKE_KERNEL(req_code, kernel_ptr, args...) kernel_ptr(args)
#endif

void demosaic_nn_hw(float *host_input, float *host_result, int row_size,
                    int col_size, int chan_size, float *acc_input,
                    float *acc_result) {
  dmaLoad(acc_input, host_input,
          row_size * col_size * chan_size * sizeof(float));
  demosaic_nn_fxp(acc_input, row_size, col_size, chan_size, acc_result);
  dmaStore(host_result, acc_result,
           row_size * col_size * chan_size * sizeof(float));
}

int main() {
  const int row_size = 128;
  const int col_size = 128;
  const int chan_size = 3;

  float *host_input =
      (float *)malloc(sizeof(float) * row_size * col_size * chan_size);
  float *host_result =
      (float *)malloc(sizeof(float) * row_size * col_size * chan_size);
  float *acc_input =
      (float *)malloc(sizeof(float) * row_size * col_size * chan_size);
  float *acc_result =
      (float *)malloc(sizeof(float) * row_size * col_size * chan_size);
  for (int i = 0; i < row_size * col_size * chan_size; i++)
    host_input[i] = 1.0;

  MAP_ARRAY_TO_ACCEL(DEMOSAIC_NN,
                     "host_input",
                     acc_input,
                     sizeof(float) * row_size * col_size * chan_size);
  MAP_ARRAY_TO_ACCEL(DEMOSAIC_NN,
                     "host_result",
                     acc_result,
                     sizeof(float) * row_size * col_size * chan_size);
  INVOKE_KERNEL(DEMOSAIC_NN,
                demosaic_nn_hw,
                host_input,
                host_result,
                row_size,
                col_size,
                chan_size,
                acc_input,
                acc_result);

  for (int i = 0; i < row_size * col_size * chan_size; i++)
    printf("%f ", host_result[i]);

  free(host_input);
  free(host_result);
  free(acc_input);
  free(acc_result);
  return 0;
}

