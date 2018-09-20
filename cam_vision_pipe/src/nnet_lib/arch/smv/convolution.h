#ifndef _ARCH_SMV_CONVOLUTION_H_
#define _ARCH_SMV_CONVOLUTION_H_

#include "arch/common.h"
#include "arch/smv/common.h"
#include "core/smv/params.h"
#include "utility/fp16_utils.h"
#include "utility/utility.h"

typedef struct _smv_convolution_options {
    int img;
    int kern_start;
    int kern_end;
    int total_tile_ofmaps;
    int sampling_upscale_factor;
    bool use_pipelined_dma;
    bool execute;
    bool load_inputs_first;
} smv_convolution_options;

// Prefetch memory range descriptor.
//
// The range is described by a buffer to be prefetched and offset into that
// buffer from where prefetching should begin.
//
// Example:
//    |----------------------------------------|
//    ^             ^++++++++++++++++++++++++++^
//  base_addr   base + prefetch_offset       base_addr + bytes
//
// The memory range indicated by + would be prefetched.
typedef struct _pf_memory_range {
    char* base_addr;
    int bytes;
    int prefetch_offset;
    int prefetch_delay_ns;
} pf_memory_range;

extern const int SMV_CONV_SW_PREFETCH_THRESHOLD;
extern pthread_mutex_t stdout_lock;
void* prefetch_memory_range(void* args);

void smv_convolution_layer_hw_impl(packed_fp16* host_activations,
                                   packed_fp16* host_weights,
                                   packed_fp16* host_results,
                                   float* local_activations,
                                   float* local_weights,
                                   float* local_results,
                                   layer_t curr_layer,
                                   smv_convolution_options* options);

void smv_convolution_layer_hw(packed_fp16* dma_activations,
                              packed_fp16* dma_weights,
                              packed_fp16* dma_results,
                              packed_fp16* cache_activations,
                              packed_fp16* cache_weights,
                              packed_fp16* cache_results,
                              packed_fp16* acp_activations,
                              packed_fp16* acp_weights,
                              packed_fp16* acp_results,
                              float* umem,
                              float* spad0,
                              float* spad1,
                              layer_t curr_layer,
                              access_config* access_config,
                              smv_convolution_options* options);

void run_sampled_hw_pass(layer_t* partial_layer,
                         smv_convolution_options* hw_pass,
                         int img,
                         packed_fp16* temp_results_buf);

void smv_wt_standard_convolution_layer_impl(data_list* host_activations,
                                            data_list* host_weights,
                                            layer_t* layers,
                                            int lnum,
                                            data_list* host_results,
                                            smv_global* g_smv,
                                            device_t* device,
                                            sampling_param_t* sampling_param);
#endif
