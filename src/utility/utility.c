#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utility.h"
#include "params.h"

float *read_image_from_binary(char *file_path, int *row_size, int *col_size) {
  float *image;
  FILE *fp = fopen(file_path, "r");
  int chan_size;
  if (fread(row_size, sizeof(int), 1, fp) != 1)
    assert("Failed to read row size from binary file!");
  if (fread(col_size, sizeof(int), 1, fp) != 1)
    assert("Failed to read col size from binary file!");
  if (fread(&chan_size, sizeof(int), 1, fp) != 1)
    assert("Failed to read row size from binary file!");
  assert(chan_size == CHAN_SIZE && "Channel size read from the binary file "
                                   "doesn't equal to the default value!\n");

  int size = *row_size * *col_size * CHAN_SIZE;
  int err =
      posix_memalign((void **)&image, CACHELINE_SIZE, sizeof(float) * size);
  assert(err == 0 && "Failed to allocate memory!");
  unsigned char pixel;
  int i = 0;
  while (fread(&pixel, sizeof(unsigned char), 1, fp) == 1) {
    image[i++] = pixel;
  }
  fclose(fp);
  return image;
}

void write_image_to_binary(char *file_path, float *image, int row_size, int col_size) {
  FILE *fp = fopen(file_path, "w");

  int shape[3] = {row_size, col_size, CHAN_SIZE};
  fwrite(shape, sizeof(int), 3, fp); 

  int i = 0;
  int size = row_size * col_size * CHAN_SIZE;
  unsigned char pixel = image[0];
  while (i < size) {
    pixel = image[i++];
    fwrite(&pixel, sizeof(unsigned char), 1, fp); 
  }
  fclose(fp);
}

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
