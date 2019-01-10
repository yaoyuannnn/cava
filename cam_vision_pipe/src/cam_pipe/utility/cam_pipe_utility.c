#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "utility/cam_pipe_utility.h"
#include "kernels/pipe_stages.h"

uint8_t *read_image_from_binary(char *file_path, int *row_size, int *col_size) {
  uint8_t *image;
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
  image = malloc_aligned(sizeof(uint8_t) * size);
  if (fread(image, sizeof(uint8_t), size, fp) != size)
    assert("Failed to read the image from binary file!");
  fclose(fp);
  return image;
}

void write_image_to_binary(char *file_path, uint8_t *image, int row_size, int col_size) {
  FILE *fp = fopen(file_path, "w");

  int shape[3] = { row_size, col_size, CHAN_SIZE };
  fwrite(shape, sizeof(int), 3, fp);

  int size = row_size * col_size * CHAN_SIZE;
  fwrite(image, sizeof(uint8_t), size, fp);
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

void convert_hwc_to_chw(uint8_t *input, int row_size, int col_size,
                        uint8_t **result) {
  if (*result == NULL) {
    *result = (uint8_t *)malloc_aligned(row_size * col_size * CHAN_SIZE *
                                        sizeof(uint8_t));
  }
  ARRAY_3D(uint8_t, _input, input, col_size, CHAN_SIZE);
  ARRAY_3D(uint8_t, _result, *result, row_size, col_size);
  for (int h = 0; h < row_size; h++)
    for (int w = 0; w < col_size; w++)
      for (int c = 0; c < CHAN_SIZE; c++)
        _result[c][h][w] = _input[h][w][c];
}

void convert_chw_to_hwc(uint8_t *input, int row_size, int col_size,
                        uint8_t **result) {
  if (*result == NULL) {
    *result = (uint8_t *)malloc_aligned(row_size * col_size * CHAN_SIZE *
                                      sizeof(uint8_t));
  }
  ARRAY_3D(uint8_t, _input, input, row_size, col_size);
  ARRAY_3D(uint8_t, _result, *result, col_size, CHAN_SIZE);
  for (int c = 0; c < CHAN_SIZE; c++)
    for (int h = 0; h < row_size; h++)
      for (int w = 0; w < col_size; w++)
        _result[h][w][c] = _input[c][h][w];
}

static float sRGB_to_linear(float x) {
    if (x < 0.04045) return x/12.92;
    return pow((x+0.055)/1.055, 2.4);
}

static float linear_to_sRGB(float y) {
    if (y <= 0.0031308) return 12.92 * y;
    return 1.055 * pow(y, 1/2.4) - 0.055;
}

// Convert a RGB image into grayscale.
// The grayscale value is stored in the first channel
void convert_image_to_grayscale(uint8_t* input, int row_size, int col_size) {
    ARRAY_3D(uint8_t, _input, input, row_size, col_size);
    for (int row = 0; row < row_size; row++)
        for (int col = 0; col < col_size; col++) {
            float R_linear = sRGB_to_linear(_input[0][row][col] / 255.0);
            float G_linear = sRGB_to_linear(_input[1][row][col] / 255.0);
            float B_linear = sRGB_to_linear(_input[2][row][col] / 255.0);
            float gray_linear =
                    0.2126 * R_linear + 0.7152 * G_linear + 0.0722 * B_linear;
            _input[0][row][col] = round(linear_to_sRGB(gray_linear) * 255);
    }
}
