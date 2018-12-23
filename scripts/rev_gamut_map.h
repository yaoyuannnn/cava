#ifndef REV_GAMUT_MAP_H
#define REV_GAMUT_MAP_H

#define ARRAY_2D(TYPE, output_array_name, input_array_name, DIM_1)             \
    TYPE(*output_array_name)[DIM_1] = (TYPE(*)[DIM_1])input_array_name

#define ARRAY_3D(TYPE, output_array_name, input_array_name, DIM_1, DIM_2)      \
    TYPE(*output_array_name)[DIM_1][DIM_2] =                                   \
        (TYPE(*)[DIM_1][DIM_2])input_array_name

void rev_gamut_map(float* input,
                   int row_size,
                   int col_size,
                   int chan_size,
                   float* result,
                   float* ctrl_pts,
                   float* weights,
                   float* coefs,
                   int num_cps
                   );

#endif
