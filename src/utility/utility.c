#include "utility.h"

float *transpose_mat(float *inmat, int width, int height) {
  // Define vectors
  float *outmat;
  int err =
      posix_memalign((void **)&outmat, CACHELINE_SIZE, sizeof(float) * height * width);
  assert(err == 0 && "Failed to allocate memory!");

  // Transpose the matrix
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      outmat[j * height + i] = inmat[i * width + j];
    }
  }
  return outmat;
}
