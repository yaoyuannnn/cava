#ifndef _COMMON_DEFS_H_
#define _COMMON_DEFS_H_

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#define CACHELINE_SIZE 64

// Debugging message macros.
#if DEBUG_LEVEL >= 1
  #define INFO_MSG(args...) printf(args)

  #if DEBUG_LEVEL >= 2
    #define PRINT_MSG(args...) printf(args)
    #define PRINT_DEBUG(hid, rows, cols, num_cols)                                 \
        print_debug(hid, rows, cols, num_cols)
    #define PRINT_DEBUG4D(hid, rows, cols, height)                                 \
        print_debug4d(hid, rows, cols, height)
    #define PRINT_DEBUG4D_FP16(hid, num, height, rows, cols)                       \
        print_debug4d_fp16(hid, num, height, rows, cols)

    #if DEBUG_LEVEL >= 3
      #define PRINT_DEBUG_V(hid, rows, cols, num_cols)                               \
          print_debug(hid, rows, cols, num_cols)
      #define PRINT_DEBUG4D_V(hid, rows, cols, height)                               \
          print_debug4d(hid, rows, cols, height)
      #define PRINT_MSG_V(args...) printf(args)
    #else
      #define PRINT_DEBUG_V(hid, rows, cols, num_cols)
      #define PRINT_DEBUG4D_V(hid, rows, cols, height)
      #define PRINT_MSG_V(args...)
    #endif
  #else
    #define PRINT_MSG(args...)
    #define PRINT_DEBUG(hid, rows, cols, num_cols)
    #define PRINT_DEBUG4D(hid, rows, cols, height)
    #define PRINT_DEBUG4D_FP16(hid, num, height, rows, cols)
    #define PRINT_DEBUG_V(hid, rows, cols, height)
    #define PRINT_DEBUG4D_V(hid, rows, cols, height)
    #define PRINT_MSG_V(args...)
  #endif
#else
  #define INFO_MSG(args...)
  #define PRINT_DEBUG(hid, rows, cols, num_cols)
  #define PRINT_DEBUG4D(hid, rows, cols, height)
  #define PRINT_DEBUG4D_FP16(hid, num, height, rows, cols)
  #define PRINT_MSG(args...)
  #define PRINT_DEBUG_V(hid, rows, cols, height)
  #define PRINT_DEBUG4D_V(hid, rows, cols, height)
  #define PRINT_MSG_V(args...)
#endif

#define STRING(arg) #arg

// This is to avoid a ton of spurious unused variable warnings when
// we're not building for gem5.
#define UNUSED(x) (void)(x)

// Macros for computing the maximum of a group of elements.
//
// Why macros and not functions (or a loop)? A loop takes O(n) cycles to
// compute the maximum, when it could be done in O(log n) time with a tree
// based implementation. But Aladdin regards function calls as a hard
// dependency that it does not optimize across, so we would not get the
// parallelism we expect from the tree. Thus, these must be macros.
//
// I've only implemented a few of these. These are only meant for the pooling
// layers, and we shouldn't need more than a 3x3 pooling layer anyways.
#define max2(A, B) (((A) > (B)) ? (A) : (B))
#define max3(e0, e1, e2) max2(max2(e0, e1), e2)
#define max4(e0, e1, e2, e3) max2(max2(e0, e1), max2(e2, e3))
#define max8(e0, e1, e2, e3, e4, e5, e6, e7)                                   \
    max2(max4(e0, e1, e2, e3), max4(e4, e5, e6, e7))
#define max9(e0, e1, e2, e3, e4, e5, e6, e7, e8)                               \
    max2(max8(e0, e1, e2, e3, e4, e5, e6, e7), e8)

#define min2(A, B) (((A) < (B)) ? (A) : (B))

#define FRAC_CEIL(A, B) ((A) / (B) + ((A) % (B) != 0))
// Convenience macros to switch between invoking an accelerator (if building a
// binary for gem5) or just calling the kernel function in software.
//
// Usage:
//
//  These macros expand differently based on whether the GEM5_HARNESS macro is
//  defined. If so, then this binary is meant to be run under gem5, invoking
//  accelerators; if not, this binary should run the pure software version of
//  the accelerated kernels.
//
//  If GEM5_HARNESS is defined:
//
//     MAP_ARRAY_TO_ACCEL(myReqCode, myArrayName, myArrayPtr, mySize)
//        ===>   mapArrayToAccelerator(myReqCode, myArrayName, myArrayPtr, mySize)
//
//     INVOKE_KERNEL(myReqCode, kernelFuncName, args...)
//        ===>   invokeAcceleratorAndBlock(myReqCode)
//
//  Otherwise:
//     MAP_ARRAY_TO_ACCEL(myReqCode, myArrayName, myArrayPtr, mySize)
//        expands to nothing
//
//     INVOKE_KERNEL(myReqCode, kernelFuncName, args...)
//        ===>  kernelFuncName(args)
//
#ifdef GEM5_HARNESS

#define MAP_ARRAY_TO_ACCEL(req_code, name, base_addr, size)                    \
    mapArrayToAccelerator(req_code, name, base_addr, size)
#define INVOKE_KERNEL(req_code, kernel_ptr, args...)                           \
    do {                                                                       \
        UNUSED(kernel_ptr);                                                    \
        invokeAcceleratorAndBlock(req_code);                                   \
    } while (0)
#define INVOKE_KERNEL_NOBLOCK(req_code, finish_flag, kernel_ptr, args...)      \
    do {                                                                       \
        UNUSED(kernel_ptr);                                                    \
        invokeAcceleratorAndReturn2(req_code, finish_flag);                    \
    } while (0)

#define INVOKE_DMA_READ_TRAFFIC_GEN(start_addr, size)                          \
    do {                                                                       \
        invokeAladdinTrafficGenAndBlock(start_addr, size, false, false);       \
    } while (0)
#define INVOKE_DMA_WRITE_TRAFFIC_GEN(start_addr, size)                         \
    do {                                                                       \
        invokeAladdinTrafficGenAndBlock(start_addr, size, true, false);        \
    } while (0)
#define INVOKE_ACP_READ_TRAFFIC_GEN(start_addr, size)                          \
    do {                                                                       \
        invokeAladdinTrafficGenAndBlock(start_addr, size, false, true);        \
    } while (0)
#define INVOKE_ACP_WRITE_TRAFFIC_GEN(start_addr, size)                         \
    do {                                                                       \
        invokeAladdinTrafficGenAndBlock(start_addr, size, true, true);         \
    } while (0)

#else

#define MAP_ARRAY_TO_ACCEL(req_code, name, base_addr, size)                    \
    do {                                                                       \
        INFO_MSG("Mapping array %s @ %p, size %d.\n",                          \
                 name, (void*)base_addr, (int)(size));                         \
        UNUSED(req_code);                                                      \
        UNUSED(name);                                                          \
        UNUSED(base_addr);                                                     \
        UNUSED(size);                                                          \
    } while (0)
#define INVOKE_KERNEL(req_code, kernel_ptr, args...) kernel_ptr(args)
#define INVOKE_KERNEL_NOBLOCK(req_code, finish_flag, kernel_ptr, args...)      \
    kernel_ptr(args)
#define INVOKE_DMA_READ_TRAFFIC_GEN(start_addr, size)                          \
    do {                                                                       \
        UNUSED(start_addr);                                                    \
        UNUSED(size);                                                          \
    } while (0)
#define INVOKE_DMA_WRITE_TRAFFIC_GEN(start_addr, size)                         \
    do {                                                                       \
        UNUSED(start_addr);                                                    \
        UNUSED(size);                                                          \
    } while (0)
#define INVOKE_ACP_READ_TRAFFIC_GEN(start_addr, size)                          \
    do {                                                                       \
        UNUSED(start_addr);                                                    \
        UNUSED(size);                                                          \
    } while (0)
#define INVOKE_ACP_WRITE_TRAFFIC_GEN(start_addr, size)                         \
    do {                                                                       \
        UNUSED(start_addr);                                                    \
        UNUSED(size);                                                          \
    } while (0)

#endif

// Simplified version of MAP_ARRAY_TO_ACCEL.
//
// This assumes that the current name of the base pointer is also the name of
// the array in the top level function of the dynamic trace. THIS IS VERY
// IMPORTANT - if the argument passed to a top level function has been renamed in
// the function, then this WILL NOT WORK!
//
// MAP_ARRAY(myReqCode, myArray, mySize)
//    ===>   MAP_ARRAY_TO_ACCEL(myReqCode, "myArray", myArray, mySize)
#define MAP_ARRAY(req_code, name_and_base_addr, size)                          \
    MAP_ARRAY_TO_ACCEL(                                                        \
            req_code, STRING(name_and_base_addr), name_and_base_addr, size)

// Use these convenience macros to cast a raw pointer into a multidimensional
// variable-length array, which lets us use [] notation inside of the ugly
// sub2ind syntax!
//
// Usage:
//   If we have an array like array[5][4]:
//      ARRAY_2D(TYPE, output_name, array, 4);
//
//   If we have an array like array[5][4][3]:
//      ARRAY_3D(TYPE, output_name, array, 4, 3);
//
//   If we have an array like array[5][4][3][2]
//      ARRAY_4D(TYPE, output_name, array, 4, 3, 2);
//
//   And so on...
#define ARRAY_1D(TYPE, output_array_name, input_array_name)                    \
    TYPE* output_array_name = (TYPE*)input_array_name

#define ARRAY_2D(TYPE, output_array_name, input_array_name, DIM_1)             \
    TYPE(*output_array_name)[DIM_1] = (TYPE(*)[DIM_1])input_array_name

#define ARRAY_3D(TYPE, output_array_name, input_array_name, DIM_1, DIM_2)      \
    TYPE(*output_array_name)[DIM_1][DIM_2] =                                   \
        (TYPE(*)[DIM_1][DIM_2])input_array_name

#define ARRAY_4D(                                                              \
    TYPE, output_array_name, input_array_name, DIM_1, DIM_2, DIM_3)            \
        TYPE(*output_array_name)[DIM_1][DIM_2][DIM_3] =                        \
            (TYPE(*)[DIM_1][DIM_2][DIM_3])input_array_name

#define ARRAY_5D(                                                              \
    TYPE, output_array_name, input_array_name, DIM_1, DIM_2, DIM_3, DIM_4)     \
        TYPE(*output_array_name)[DIM_1][DIM_2][DIM_3][DIM_4] =                 \
            (TYPE(*)[DIM_1][DIM_2][DIM_3][DIM_4])input_array_name

#endif
