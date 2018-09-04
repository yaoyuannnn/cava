#define DEMOSAIC_NN 0x3
#define TRANSFORM 0x4
#define CACHELINE_SIZE 64


#define max(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

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

void demosaic_nn_fxp(float *input, int row_size, int col_size, int chan_size,
                     float *result);

void transform_fxp(float *input, int row_size, int col_size, int chan_size,
                   float *result, float *TsTw_tran);
