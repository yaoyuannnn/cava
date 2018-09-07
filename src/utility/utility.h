#ifndef _UTILITY_H_
#define _UTILITY_H_

float *read_image_from_binary(char *file_path, int *row_size, int *col_size);
void write_image_to_binary(char *file_path, float *image, int row_size,
                           int col_size);
float *transpose_mat(float *inmat, int width, int height);

#endif
