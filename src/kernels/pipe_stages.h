#ifndef _PIPE_STAGES_H_
#define _PIPE_STAGES_H_

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#define CACHELINE_SIZE 64
#define CHAN_SIZE 3

#define ISP 0x3
#define TRANSFORM 0x4

#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

#define abs(a) \
  ({ __typeof__ (a) _a = (a); \
    _a < 0 ? -_a : _a; })

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

#define ARRAY_2D(TYPE, output_array_name, input_array_name, DIM_1)             \
    TYPE(*output_array_name)[DIM_1] = (TYPE(*)[DIM_1])input_array_name

#define ARRAY_3D(TYPE, output_array_name, input_array_name, DIM_1, DIM_2)      \
    TYPE(*output_array_name)[DIM_1][DIM_2] =                                   \
        (TYPE(*)[DIM_1][DIM_2])input_array_name

extern int num_ctrl_pts;

void scale_fxp(uint8_t *input, int row_size, int col_size, float *result);

void descale_fxp(float *input, int row_size, int col_size, uint8_t *result);

void demosaic_fxp(float *input, int row_size, int col_size, float *result);

void denoise_fxp(float *input, int row_size, int col_size, float *result);

void transform_fxp(float *input, int row_size, int col_size, float *result,
                   float *TsTw_tran);

void gamut_map_fxp(float *input, int row_size, int col_size, float *result,
                   float *ctrl_pts, float *weights, float *coefs,
                   float *l2_dist);

void tone_map_fxp(float *input, int row_size, int col_size, float *tone_map,
                  float *result);

void tone_map_approx_fxp(float *input, int row_size, int col_size,
                         float *result);

#endif
