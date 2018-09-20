#include <string.h>
#include <pthread.h>

#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/common.h"
#include "arch/smv/load_and_unpack_fp16_data.h"
#include "arch/smv/convolution.h"
#include "core/smv/params.h"
#include "core/smv/smv.h"
#include "core/ref/activation_functions.h"
#include "utility/data_layout_conversion.h"
#include "utility/fp16_utils.h"
#include "utility/thread_pool.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

typedef struct _conv_output_tile {
    int output_dims[5];
    int output_pad;
    int num_ofmaps;
    smv_convolution_options* hw_passes;
    int num_hw_passes;
    int sampling_upscale_factor;
    bool execute;
} conv_output_tile;

typedef struct _conv_input_tile {
    int input_dims[5];
    int input_pad;
    padding pad;
    conv_output_tile* output_tiles;
    int num_output_tiles;
    int sampling_upscale_factor;
    bool execute;
} conv_input_tile;

typedef struct _conv_l2_tile {
    conv_input_tile* input_tiles;
    int num_input_tiles;
    int num_kernels;
    int sampling_upscale_factor;
    bool execute;
} conv_l2_tile;

typedef struct _conv_tiling_cfg {
    conv_l2_tile* l2_tiles;
    int num_l2_tiles;
} conv_tiling_cfg;

// Prefetching usually involves looking ahead some offset X from the base of
// the buffer being prefetched. If that X is greater than the buffer size
// itself, then prefetching has no benefit. This threshold defines the minimum
// threshold X bytes, below which prefetching is not worth the cost.
const int SMV_CONV_SW_PREFETCH_THRESHOLD = 0;
pthread_mutex_t stdout_lock = PTHREAD_MUTEX_INITIALIZER;

void free_conv_tiling_cfg(conv_tiling_cfg* cfg) {
    for (int k = 0; k < cfg->num_l2_tiles; k++) {
        conv_l2_tile* l2_tile = &cfg->l2_tiles[k];
        for (int i = 0; i < l2_tile->num_input_tiles; i++) {
            conv_input_tile* input_tile = &l2_tile->input_tiles[i];
            for (int j = 0; j < input_tile->num_output_tiles; j++) {
                free(input_tile->output_tiles[j].hw_passes);
            }
            free(input_tile->output_tiles);
        }
        free(l2_tile->input_tiles);
    }
    free(cfg->l2_tiles);
}

void print_conv_tiling_cfg(conv_tiling_cfg* cfg, int lnum) {
    INFO_MSG("\nTiling info for layer %d\n", lnum);
    INFO_MSG("\nNumber of L2 tiles %d\n", cfg->num_l2_tiles);
    for (int k = 0; k < cfg->num_l2_tiles; k++) {
        conv_l2_tile* l2_tile = &cfg->l2_tiles[k];
        INFO_MSG("L2 tile %d\n"
                 "  Execute: %s\n"
                 "  Each tile represents: %d L2 tiles\n"
                 "  Input tiles: %d\n",
                 k,
                 bool_to_yesno(l2_tile->execute),
                 l2_tile->sampling_upscale_factor,
                 l2_tile->num_input_tiles);
        for (int i = 0; i < l2_tile->num_input_tiles; i++) {
            conv_input_tile* input_tile =
                    &l2_tile->input_tiles[i];
            INFO_MSG("  + Input tile %d\n"
                     "      Execute: %s\n"
                     "      IFMap size: %d, %d, %d, %d, %d\n"
                     "      zero padding: %d, %d, %d, %d\n"
                     "      input pad: %d\n"
                     "      Each tile represents: %d input tiles\n"
                     "      Output tiles: %d\n",
                     i,
                     bool_to_yesno(input_tile->execute),
                     input_tile->input_dims[0],
                     input_tile->input_dims[1],
                     input_tile->input_dims[2],
                     input_tile->input_dims[3],
                     input_tile->input_dims[4],
                     input_tile->pad.top,
                     input_tile->pad.bottom,
                     input_tile->pad.left,
                     input_tile->pad.right,
                     input_tile->input_pad,
                     input_tile->sampling_upscale_factor,
                     input_tile->num_output_tiles);
            for (int j = 0; j < input_tile->num_output_tiles; j++) {
                conv_output_tile* output_tile =
                        &input_tile->output_tiles[j];
                INFO_MSG("    + Output tile %d:\n"
                         "        Execute: %s\n"
                         "        OFMaps: %d\n"
                         "        OFMap size: %d, %d, %d, %d, %d\n"
                         "        output pad %d\n"
                         "        Each tile represents: %d output tiles\n"
                         "        Num HW passes: %d\n",
                         j,
                         bool_to_yesno(output_tile->execute),
                         output_tile->num_ofmaps,
                         output_tile->output_dims[0],
                         output_tile->output_dims[1],
                         output_tile->output_dims[2],
                         output_tile->output_dims[3],
                         output_tile->output_dims[4],
                         output_tile->output_pad,
                         output_tile->sampling_upscale_factor,
                         output_tile->num_hw_passes);
                for (int k = 0; k < output_tile->num_hw_passes; k++) {
                    smv_convolution_options* hw_options = &output_tile->hw_passes[k];
                    INFO_MSG("        + HW pass %d:\n"
                             "             Execute: %s\n"
                             "             Represents: %d HW passes\n",
                             k,
                             bool_to_yesno(hw_options->execute),
                             hw_options->sampling_upscale_factor);
                }
            }
        }
    }
}

ALWAYS_INLINE
void smv_convolution_layer_hw_impl_weights_load(
        packed_fp16* host_weights,
        float* local_weights,
        layer_t* curr_layer,
        smv_convolution_options* options) {
    // DMA all the weights that we can fit in the current tile (which is
    // specified by this tile's outputs.height).
    // We should only DMA part of the weights.
    int single_weights_elems =
            curr_layer->weights.rows * curr_layer->weights.cols *
            (curr_layer->weights.height + curr_layer->weights.align_pad);
    int num_weights = options->total_tile_ofmaps * single_weights_elems;
    if (curr_layer->weights_req == IO_DMA) {
        setReadyBits(local_weights, num_weights * sizeof(float), 0);
        dma_load_and_unpack_fp16(
                local_weights, host_weights, num_weights, 0, 0);
    } else if (curr_layer->weights_req == IO_ACP) {
        acp_load_and_unpack_fp16(
                local_weights, host_weights, num_weights, 0, 0);
    }
}

ALWAYS_INLINE
void smv_convolution_layer_hw_impl_inputs_load(
        packed_fp16* host_activations,
        float* local_activations,
        layer_t* curr_layer,
        smv_convolution_options* options) {
    int input_height = curr_layer->inputs.height;
    int input_rows = curr_layer->inputs.rows;
    int input_cols = curr_layer->inputs.cols;
    int input_pad = curr_layer->inputs.align_pad;
    ARRAY_4D(packed_fp16, _a, host_activations, input_rows, input_cols,
             input_height + input_pad);
    if (curr_layer->input_req == IO_DMA || curr_layer->input_req == IO_ACP) {
        // Read in ALL channels of the input into the UMEM at once, so that we
        // can reuse them on subsequent output channels.
        size_t num_input_pixels =
                curr_layer->inputs.rows * curr_layer->inputs.cols *
                (curr_layer->inputs.height + curr_layer->inputs.align_pad);
        if (curr_layer->input_req == IO_DMA) {
            setReadyBits(local_activations,
                         num_input_pixels * sizeof(*local_activations), 0);
            dma_load_and_unpack_fp16(local_activations,
                                     &_a[options->img][0][0][0],
                                     num_input_pixels, 0, 0);
        } else {
            int source_offset = &_a[options->img][0][0][0] - &_a[0][0][0][0];
            acp_load_and_unpack_fp16(local_activations,
                                     &_a[options->img][0][0][0],
                                     num_input_pixels, 0, source_offset);
        }
    }
}

void smv_convolution_layer_hw_impl(packed_fp16* host_activations,
                                   packed_fp16* host_weights,
                                   packed_fp16* host_results,
                                   float* local_activations,
                                   float* local_weights,
                                   float* local_results,
                                   layer_t curr_layer,
                                   smv_convolution_options* options) {

    // The main CPU will prefetch either the inputs or weights first, depending
    // on the tiling strategy. If the inputs are prefetched first, we should
    // load the inputs first; if the weights are prefetched first, we should
    // load the weights first.
    if (options->load_inputs_first) {
        smv_convolution_layer_hw_impl_inputs_load(
                host_activations, local_activations, &curr_layer, options);
        smv_convolution_layer_hw_impl_weights_load(
                host_weights, local_weights, &curr_layer, options);
    } else {
        smv_convolution_layer_hw_impl_weights_load(
                host_weights, local_weights, &curr_layer, options);
        smv_convolution_layer_hw_impl_inputs_load(
                host_activations, local_activations, &curr_layer, options);
    }

    // We will invoke the hardware multiple times, but we don't DMA every time.
    // So, we need to read weights from and write outputs to the kern_start'th
    // output channel.
    convolution3d_smv(local_activations, local_weights, curr_layer,
                      options->kern_start, local_results);

    // Run the activation function in-place if applicable on the ofmaps we
    // generated.
    int ofmap_2d_elems =
            curr_layer.outputs.rows *
            (curr_layer.outputs.cols + curr_layer.outputs.align_pad);
    int num_output_pixels =
            (options->kern_end - options->kern_start) * ofmap_2d_elems;
    int start_output_pixel = options->kern_start * ofmap_2d_elems;
    smv_activation_fun_fxp(local_results, NUM_TEST_CASES, num_output_pixels,
                           start_output_pixel, curr_layer.activation);
    if (curr_layer.output_req == IO_DMA) {
        dma_pack_and_store_fp16(host_results,
                                local_results,
                                num_output_pixels,
                                start_output_pixel,
                                start_output_pixel);
    } else if (curr_layer.output_req == IO_ACP) {
        acp_pack_and_store_fp16(host_results,
                                local_results,
                                num_output_pixels,
                                start_output_pixel,
                                start_output_pixel);
    }
}

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
                              smv_convolution_options* options) {
    // XXX: With half-precision data storage, we can't directly access an L1
    // cache, since we can't compute on half precision data without calling the
    // conversion functions on every access. So to prevent such a situation
    // from arising, just disable use of local caches.
    bool use_acp_outputs =
            access_config->outputs == _ACP || access_config->outputs == _Cache;
    bool use_acp_inputs =
            access_config->inputs == _ACP || access_config->inputs == _Cache;
    if (use_acp_outputs) {
        curr_layer.output_req = IO_ACP;
        if (use_acp_inputs) {
            curr_layer.input_req = IO_ACP;
            if (access_config->weights == _DmaOrLocal) {
                smv_convolution_layer_hw_impl(acp_activations, dma_weights,
                    acp_results, umem, spad0, spad1, curr_layer, options);
            } else {
                smv_convolution_layer_hw_impl(acp_activations, acp_weights,
                    acp_results, umem, spad0, spad1, curr_layer, options);
            }
        } else {
            if (curr_layer.input_req != IO_NONE)
                curr_layer.input_req = IO_DMA;
            if (access_config->weights == _DmaOrLocal) {
                smv_convolution_layer_hw_impl(dma_activations, dma_weights,
                    acp_results, umem, spad0, spad1, curr_layer, options);
            } else {
                smv_convolution_layer_hw_impl(dma_activations, acp_weights,
                    acp_results, umem, spad0, spad1, curr_layer, options);
            }
        }
    } else {
        ASSERT((access_config->inputs == _DmaOrLocal &&
                access_config->outputs == _DmaOrLocal) &&
               "IO requirements are inconsistent with DMA fallback!");
        if (access_config->weights == _DmaOrLocal) {
            smv_convolution_layer_hw_impl(dma_activations, dma_weights,
                dma_results, umem, spad0, spad1, curr_layer, options);
        } else {
            smv_convolution_layer_hw_impl(dma_activations, acp_weights,
                dma_results, umem, spad0, spad1, curr_layer, options);
        }
    }
}

void* prefetch_memory_range(void* args) {
    pf_memory_range* range = (pf_memory_range*)args;
    if (range->prefetch_offset >= range->bytes)
        return NULL;
    char* start_addr = range->base_addr + range->prefetch_offset;
    char* end_addr = start_addr + range->bytes - range->prefetch_offset;
#if DEBUG_LEVEL >= 1
    pthread_mutex_lock(&stdout_lock);
    INFO_MSG("Prefetching from %p for %d bytes, delay %d.\n",
             (void*)start_addr, range->bytes, range->prefetch_delay_ns);
    pthread_mutex_unlock(&stdout_lock);
#endif
    int delay_ns = range->prefetch_delay_ns;
    if (delay_ns > 0) {
      // Spin for approximately delay_ns time.
      int start_ts = rdtsc();
      float diff = 0;
      do {
          // Use rdtsc instead of gettimeofday to avoid syscall overhead. It's not
          // a serializing instruction, but that's okay. An approximation will do.
          int end_ts = rdtsc();
          diff = ((float)(end_ts - start_ts)) / 2.5;  // Assuming 2.5GHz clock.
      } while (diff < delay_ns);
    }
    // volatile ensures that the load will not be optimized away.
    for (volatile char* addr = start_addr; addr < end_addr;
         addr += CACHELINE_SIZE) {
        // prefetch instructions seem to be mostly ignored in simulation and
        // have little to no effect on overall performance, so use a "hard"
        // prefetch instead.
        char temp = *addr;
        UNUSED(temp);
    }
    free(range);
    return NULL;
}

// Attempt to prefetch the weights for the **NEXT** output tile.
//
// We won't be able to prefetch weights for the current output tile since
// they'll be cold misses anyways and there isn't enough time to make a
// meaningful difference. But we can prefetch the next weights while the HW is
// running and finished with loading data.
//
// Arguments:
//    input_tile: The current input tile.
//    weights: The base address of the 16-bit weights.
//    weights_dims: The dimensions of the complete weights, padded in NHWC.
//    output_tile_kern_start: The current output tile starts from the N'th
//       output feature map.
//    output_tile_idx: The N'th output tile.
//    use_pf_delay: If true, apply a delay to this prefetch operation.
void try_prefetch_weights(conv_input_tile* input_tile,
                          float16* weights,
                          dims_t* weights_dims,
                          int output_tile_kern_start,
                          int output_tile_idx,
                          bool use_pf_delay) {
#ifdef TRACE_MODE
    return;
#endif
    // The range argument is freed by the worker thread when it is finished.
    pf_memory_range* range = (pf_memory_range*)malloc(sizeof(pf_memory_range));
    conv_output_tile* curr_output_tile =
            &input_tile->output_tiles[output_tile_idx];
    conv_output_tile* next_output_tile =
            &input_tile->output_tiles[output_tile_idx + 1];
    range->bytes = next_output_tile->num_ofmaps * get_dims_size(weights_dims) *
                   sizeof(float16);
    if (range->bytes < SMV_CONV_SW_PREFETCH_THRESHOLD)
        return;
    ARRAY_4D(float16,
             _kernels,
             weights,
             weights_dims->rows,
             weights_dims->cols,
             weights_dims->height + weights_dims->align_pad);
    int next_kern_start = output_tile_kern_start + curr_output_tile->num_ofmaps;
    range->base_addr = (char*)&_kernels[next_kern_start][0][0][0];
    range->prefetch_offset = 0;

    if (use_pf_delay) {
        // To prevent the prefetch from competing for L2 bandwidth with the
        // accelerator when loading the current tile data, delay the prefetch
        // until we think the accelerator is done loading all inputs and
        // weights.
        //
        // Every output tile will need to load the weights, but only the first
        // also needs to load the inputs.
        int current_tile_load_size = curr_output_tile->num_ofmaps *
                                     get_dims_size(weights_dims);
        if (output_tile_idx == 0) {
            current_tile_load_size +=
                    (input_tile->input_dims[0] + input_tile->input_pad) *
                    input_tile->input_dims[1] * input_tile->input_dims[2];
        }
        current_tile_load_size *= sizeof(float16);
        // Conservatively assume an average ACP utilization of 10GB/s (out of
        // 25.6GB/s).
        range->prefetch_delay_ns = (current_tile_load_size) / 10.0;
    } else {
        range->prefetch_delay_ns = 0;
    }
    thread_dispatch(prefetch_memory_range, (void*)range);
}

// Attempt to prefetch the inputs for the **CURRENT** input tile.
//
// Arguments:
//    input_tile: The current input tile.
//    activations: The base address of the 16-bit inputs.
//    img: The N'th input image.
//    input_row_start: The current input tile starts from the N'th row.
void try_prefetch_input_activations(conv_input_tile* input_tile,
                                    float16* activations,
                                    int img,
                                    int input_row_start) {
#ifdef TRACE_MODE
    return;
#endif
    pf_memory_range* range = (pf_memory_range*)malloc(sizeof(pf_memory_range));
    ARRAY_4D(float16,
             _activations,
             activations,
             input_tile->input_dims[2],
             input_tile->input_dims[1],
             input_tile->input_dims[0] + input_tile->input_pad);
    range->base_addr = (char*)&_activations[img][input_row_start][0][0];
    range->bytes = (input_tile->input_dims[0] + input_tile->input_pad) *
                   input_tile->input_dims[1] * input_tile->input_dims[2] *
                   sizeof(float16);
    range->prefetch_offset = 0;
    range->prefetch_delay_ns = 0;
    thread_dispatch(prefetch_memory_range, (void*)range);
}

// Determine whether to use ACP or DMA for the weights for this output tile.
io_req_t get_weights_io_req(layer_t* curr_layer,
                            int total_input_tiles,
                            int num_hw_passes,
                            device_t* device) {
    if (device->weights_load_policy == UseDefaultOffload) {
        return curr_layer->weights_req;
    } else if (device->weights_load_policy == DmaAlways) {
        return IO_DMA;
    } else if (device->weights_load_policy == AcpAlways) {
        return IO_ACP;
    } else if (device->weights_load_policy == DynamicDmaAcp) {
        // If there is only one input tile, there is no reuse of weights across
        // input tiles. However, we can still take advantage of the cache for
        // software-based double buffering if there are multiple hardware
        // passes.
        if (total_input_tiles == 1) {
            if (!device->use_sw_prefetch || num_hw_passes == 1)
                return IO_DMA;
        }
        return IO_ACP;
    } else {
        assert(false && "Unknown data movement load policy!");
        return IO_DMA;
    }
}

io_req_t get_input_io_req(layer_t* curr_layer, device_t* device) {
    if (device->inputs_load_policy == UseDefaultOffload) {
        return curr_layer->input_req;
    } else if (device->inputs_load_policy == AcpAlways) {
        // In this tiling strategy, inputs are never reused, so we only want to
        // use ACP if the user explicitly requested it.
        return IO_ACP;
    }
    return IO_DMA;
}

//=------- Functions to capture cache behavior of sampled tiles --------=//
//
// When we sample, we entirely skip the computation for that tile, but that
// also means we don't touch the caches, so our working set is smaller. To
// model the original working set size, these functions set the relevant
// regions of memory to zero, thus ensuring that those memory regions are
// touched.  We then ignore the cycles in profiling.

// To mimic a HW pass, just set the entire temporary result buffer to zero,
// even though the pixels for each output feature map will interleaved across
// the NUM_PE_INSTS feature maps, because the output tiling code will take care
// of rearranging the zeros into the correct format (as if it was real data).
void run_sampled_hw_pass(layer_t* partial_layer,
                         smv_convolution_options* hw_pass,
                         int img,
                         packed_fp16* temp_results_buf) {
    begin_ignored_profiling(partial_layer->num);
    int result_2d_size =
            partial_layer->outputs.rows *
            (partial_layer->outputs.cols + partial_layer->outputs.align_pad);
    float16* results_buf =
            ((float16*)temp_results_buf) + result_2d_size * hw_pass->kern_start;
    int result_size = sizeof(float16) * result_2d_size *
                      (hw_pass->kern_end - hw_pass->kern_start);
    MAP_ARRAY_TO_ACCEL(g_smv.kConvolutionHw,
                       get_host_results_var_name(partial_layer->output_req),
                       results_buf,
                       result_size);
    if (partial_layer->output_req == IO_DMA) {
        INVOKE_DMA_WRITE_TRAFFIC_GEN(results_buf, result_size);
    } else if (partial_layer->output_req == IO_ACP) {
        INVOKE_ACP_WRITE_TRAFFIC_GEN(results_buf, result_size);
    }
    end_profiling();
}

// To mimic an entire output tile, set the correct slices (rows) of each output
// feature map in this output tile to zero.
void run_sampled_output_tile(layer_t* curr_layer,
                             conv_output_tile* output_tile,
                             int total_input_tiles,
                             int img,
                             int ofmap_start,
                             int ofmap_row_start,
                             device_t* device,
                             float16* kernels,
                             packed_fp16* results_base) {
    begin_ignored_profiling(curr_layer->num);
    ARRAY_4D(float16, _result, results_base, curr_layer->outputs.height,
             curr_layer->outputs.rows,
             curr_layer->outputs.cols + curr_layer->outputs.align_pad);
    int ofmap_size = sizeof(float16) * output_tile->output_dims[1] *
                     (output_tile->output_dims[0] + output_tile->output_pad);
    int weight_pad = calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);
    int single_kernel_size = curr_layer->weights.rows *
                             curr_layer->weights.cols *
                             (curr_layer->weights.height + weight_pad);
    int kernel_size =
            sizeof(float16) * single_kernel_size * output_tile->num_ofmaps;
    float16* kernel_loc = kernels + ofmap_start * single_kernel_size;
    io_req_t weights_req = get_weights_io_req(
            curr_layer, total_input_tiles, output_tile->num_hw_passes, device);
    MAP_ARRAY_TO_ACCEL(g_smv.kConvolutionHw,
                       get_host_weights_var_name(weights_req),
                       kernel_loc,
                       kernel_size);
    if (curr_layer->weights_req == IO_DMA) {
        INVOKE_DMA_READ_TRAFFIC_GEN(kernel_loc, kernel_size);
    } else if (curr_layer->weights_req == IO_ACP) {
        INVOKE_ACP_READ_TRAFFIC_GEN(kernel_loc, kernel_size);
    }
    for (int k = ofmap_start; k < ofmap_start + output_tile->num_ofmaps; k++) {
        float16* dest = &_result[img][k][ofmap_row_start][0];
        MAP_ARRAY_TO_ACCEL(g_smv.kConvolutionHw,
                           get_host_results_var_name(curr_layer->output_req),
                           dest,
                           ofmap_size);
        if (curr_layer->output_req == IO_DMA) {
            INVOKE_DMA_WRITE_TRAFFIC_GEN(dest, ofmap_size);
        } else if (curr_layer->output_req == IO_ACP) {
            INVOKE_ACP_WRITE_TRAFFIC_GEN(dest, ofmap_size);
        }
    }
    end_profiling();
}

// To mimic an entire input tile, set the entire slices of all the output
// feature maps to zero.
void run_sampled_input_tile(layer_t* curr_layer,
                            conv_input_tile* input_tile,
                            int total_input_tiles,
                            int img,
                            int ofmap_row_start,
                            device_t* device,
                            float16* activations,
                            float16* kernels,
                            packed_fp16* results_base) {
    begin_ignored_profiling(curr_layer->num);
    size_t input_tile_size =
            sizeof(float16) * input_tile->input_dims[2] *
            input_tile->input_dims[1] *
            (input_tile->input_dims[0] + input_tile->input_pad);
    io_req_t input_req = get_input_io_req(curr_layer, device);
    MAP_ARRAY_TO_ACCEL(g_smv.kConvolutionHw,
                       get_host_inputs_var_name(input_req),
                       activations,
                       input_tile_size);
    if (input_req == IO_DMA) {
        INVOKE_DMA_READ_TRAFFIC_GEN(activations, input_tile_size);
    } else if (input_req == IO_ACP) {
        INVOKE_ACP_READ_TRAFFIC_GEN(activations, input_tile_size);
    }
    int ofmap_start = 0;
    for (int i = 0; i < input_tile->num_output_tiles; i++) {
        conv_output_tile* output_tile = &input_tile->output_tiles[i];
        run_sampled_output_tile(curr_layer,
                                output_tile,
                                total_input_tiles,
                                img,
                                ofmap_start,
                                ofmap_row_start,
                                device,
                                kernels,
                                results_base);
        ofmap_start += output_tile->num_ofmaps;
    }
    end_profiling();
}

// To mimic an entire L2 tile. set the entire slices of all the output
// feature maps to zero.
void run_sampled_l2_tile(layer_t* curr_layer,
                         conv_l2_tile* l2_tile,
                         int img,
                         int l2_tile_kern_start,
                         device_t* device,
                         packed_fp16* activations_base,
                         packed_fp16* kernels_base,
                         packed_fp16* results_base) {
    begin_ignored_profiling(curr_layer->num);
    ARRAY_4D(float16, _activations, activations_base,
             curr_layer->inputs.rows, curr_layer->inputs.cols,
             curr_layer->inputs.height + curr_layer->inputs.align_pad);
    ARRAY_4D(float16, _kernel, kernels_base, curr_layer->weights.rows,
             curr_layer->weights.cols,
             curr_layer->weights.height + curr_layer->weights.align_pad);
    int halo_rows = curr_layer->weights.rows - curr_layer->stride.rows;
    int input_row_start = 0;
    int ofmap_row_start = 0;
    for (int i = 0; i < l2_tile->num_input_tiles; i++) {
        conv_input_tile* input_tile = &l2_tile->input_tiles[i];
        float16* input_loc = &_activations[img][input_row_start][0][0];
        run_sampled_input_tile(curr_layer,
                               input_tile,
                               l2_tile->num_input_tiles,
                               img,
                               ofmap_row_start,
                               device,
                               input_loc,
                               &_kernel[l2_tile_kern_start][0][0][0],
                               results_base);
        input_row_start += input_tile->input_dims[2] - halo_rows;
        ofmap_row_start += input_tile->output_tiles[0].output_dims[1];
    }
    end_profiling();
}

static layer_t create_partial_layer_from_tile(layer_t* full_layer,
                                              conv_input_tile* input_tile,
                                              conv_output_tile* output_tile) {
    layer_t partial_layer = *full_layer;
    partial_layer.inputs.rows = input_tile->input_dims[2];
    partial_layer.inputs.cols = input_tile->input_dims[1];
    partial_layer.inputs.height = input_tile->input_dims[0];
    partial_layer.inputs.align_pad = input_tile->input_pad;
    partial_layer.outputs.rows = output_tile->output_dims[1];
    partial_layer.outputs.cols = output_tile->output_dims[0];
    partial_layer.outputs.height = output_tile->output_dims[2];
    partial_layer.outputs.align_pad = output_tile->output_pad;
    partial_layer.weights.height = input_tile->input_dims[0];
    partial_layer.weights.align_pad = input_tile->input_pad;
    partial_layer.pad = input_tile->pad;
    return partial_layer;
}

/* Return the largest amount of output pixels produced by an output tile for a
 * given input tile. */
int get_largest_output_tile_size(conv_input_tile* input_tile) {
    int max = 0;
    for (int i = 0; i < input_tile->num_output_tiles; i++) {
        conv_output_tile* output_tile = &input_tile->output_tiles[i];
        max = max2(
                max, (output_tile->output_dims[0] + output_tile->output_pad) *
                             output_tile->output_dims[1] *
                             output_tile->output_dims[2]);
    }
    return max;
}

/* For each L2 tile, input tile, output tile, and HW pass, determine if it
 * should be executed.
 *
 * When we sample, we skip certain output tiles and HW passes based on the
 * appropriate sampling parameter. This function sets the execute bit for each
 * of these tiles depending on whether it should be skipped or not. If a tile /
 * HW pass is not skipped, we also set the appropriate sampling upscaling
 * factor.
 */
static void set_sampling_parameters(conv_tiling_cfg* conv_tiling,
                                    layer_t* curr_layer,
                                    sampling_param_t* sampling_param) {
    // These parameters indicate the number of tiles/hw passes to run *IN
    // ADDITION* to the minimum required amount. At a minimum, we must run the
    // first and last input output tiles and HW passes. So if
    // sampled_outer_tiles = 1, then the total number of executed output tiles
    // is 3.
    int sampled_l2_tiles = sampling_param->smv_conv_l2_tiles;
    int sampled_input_tiles = sampling_param->smv_conv_input_tiles;
    int sampled_output_tiles = sampling_param->smv_conv_output_tiles;
    int sampled_inner_iters = sampling_param->smv_conv_inner_iters;

    bool do_l2_tile_sampling =
            (sampled_l2_tiles > 0 &&
             sampled_input_tiles < conv_tiling->num_l2_tiles - 1);
    int sampling_l2_tiles_upscale_factor;
    if (do_l2_tile_sampling) {
        sampling_l2_tiles_upscale_factor = ceil(
                ((float)conv_tiling->num_l2_tiles - 1) / sampled_l2_tiles);
    } else {
        sampling_l2_tiles_upscale_factor = 1;
    }
    int l2_tiles_remaining = conv_tiling->num_l2_tiles;
    for (int lt = 0; lt < conv_tiling->num_l2_tiles; lt++) {
        conv_l2_tile* l2_tile = &conv_tiling->l2_tiles[lt];
        bool is_last_l2_tile = (lt == conv_tiling->num_l2_tiles - 1);
        if (is_last_l2_tile) {
            l2_tile->execute = true;
            l2_tile->sampling_upscale_factor = 1;
        } else {
            if (l2_tiles_remaining > 1) {
                l2_tile->execute = true;
                l2_tile->sampling_upscale_factor =
                        min2(sampling_l2_tiles_upscale_factor,
                             l2_tiles_remaining - 1);
            } else {
                l2_tile->execute = false;
                l2_tile->sampling_upscale_factor = 0;
            }
        }
        l2_tiles_remaining -= l2_tile->sampling_upscale_factor;

        bool do_input_tile_sampling =
                (sampled_input_tiles > 0 &&
                 sampled_input_tiles < l2_tile->num_input_tiles - 2);
        int sampling_input_tiles_upscale_factor;
        if (do_input_tile_sampling) {
            sampling_input_tiles_upscale_factor =
                    ceil(((float)l2_tile->num_input_tiles - 2) /
                         sampled_input_tiles);
        } else {
            sampling_input_tiles_upscale_factor = 1;
        }
        int input_tiles_remaining = l2_tile->num_input_tiles;
        for (int it = 0; it < l2_tile->num_input_tiles; it++) {
            conv_input_tile* input_tile = &l2_tile->input_tiles[it];
            bool is_first_or_last_input_tile =
                    (it == 0 || it == l2_tile->num_input_tiles - 1);
            if (is_first_or_last_input_tile) {
                input_tile->execute = true;
                input_tile->sampling_upscale_factor = 1;
            } else {
                if (input_tiles_remaining > 1) {
                    input_tile->execute = true;
                    input_tile->sampling_upscale_factor =
                            min2(sampling_input_tiles_upscale_factor,
                                 input_tiles_remaining - 1);
                } else {
                    input_tile->execute = false;
                    input_tile->sampling_upscale_factor = 0;
                }
            }
            input_tiles_remaining -= input_tile->sampling_upscale_factor;

            // The first output tile may need to handle input activation DMA, and
            // the last output tile may be smaller than the rest, so they always
            // need to be executed.
            bool do_output_tile_sampling =
                    (sampled_output_tiles > 0 &&
                     sampled_output_tiles < input_tile->num_output_tiles - 2);

            // If we sample output tiles, how many output tiles does each executed
            // tile represent?
            int sampling_output_tiles_upscale_factor;
            if (do_output_tile_sampling) {
                sampling_output_tiles_upscale_factor =
                        ceil(((float)input_tile->num_output_tiles - 2) /
                             sampled_output_tiles);
            } else {
                sampling_output_tiles_upscale_factor = 1;
            }
            int output_tiles_remaining = input_tile->num_output_tiles;
            for (int ot = 0; ot < input_tile->num_output_tiles; ot++) {
                conv_output_tile* output_tile = &input_tile->output_tiles[ot];
                bool is_first_or_last_output_tile =
                        (ot == 0 || ot == input_tile->num_output_tiles - 1);
                if (is_first_or_last_output_tile || !do_output_tile_sampling) {
                    output_tile->execute = true;
                    output_tile->sampling_upscale_factor = 1;
                } else {
                    // Compare the remaining output tile with one, not zero,
                    // because we always need to execute the last iteration.
                    if (output_tiles_remaining > 1) {
                        output_tile->execute = true;
                        output_tile->sampling_upscale_factor =
                                min2(sampling_output_tiles_upscale_factor,
                                     output_tiles_remaining - 1);
                    } else {
                        output_tile->execute = false;
                        output_tile->sampling_upscale_factor = 0;
                    }
                }
                output_tiles_remaining -= output_tile->sampling_upscale_factor;

                // Each output tile may requires multiple HW passes to execute. We
                // only sample if we must run three or more.
                bool do_hw_pass_sampling =
                        sampled_inner_iters > 0 &&
                        sampled_inner_iters < output_tile->num_hw_passes - 2;
                int remaining_hw_passes = output_tile->num_hw_passes;
                // If we sample HW passes, how many HW passes does each executed
                // pass represent?
                int sampling_hw_passes_upscale_factor;
                if (do_hw_pass_sampling) {
                    sampling_hw_passes_upscale_factor =
                            ceil(((float)output_tile->num_hw_passes - 2) /
                                 sampled_inner_iters);
                } else {
                    sampling_hw_passes_upscale_factor = 1;
                }

                for (int hw_pass = 0; hw_pass < output_tile->num_hw_passes;
                     hw_pass++) {
                    smv_convolution_options* hw_options =
                            &output_tile->hw_passes[hw_pass];
                    bool is_first_or_last_hw_pass =
                            (hw_pass == 0) ||
                            (hw_pass == output_tile->num_hw_passes - 1);
                    if (is_first_or_last_hw_pass || !do_hw_pass_sampling) {
                        hw_options->execute = true;
                        hw_options->sampling_upscale_factor = 1;
                    } else {
                        // Compare the remaining HW pass with one, not zero,
                        // because we always need to execute the last iteration.
                        if (remaining_hw_passes > 1) {
                            hw_options->execute = true;
                            hw_options->sampling_upscale_factor =
                                    min2(sampling_hw_passes_upscale_factor,
                                         remaining_hw_passes);
                        } else {
                            hw_options->execute = false;
                            hw_options->sampling_upscale_factor = 0;
                        }
                    }
                    remaining_hw_passes -= hw_options->sampling_upscale_factor;
                }
            }
        }
    }
}

static conv_tiling_cfg convolution_divide_work(layer_t* curr_layer,
                                               smv_global* g_smv) {
    // Ensure that all the tiling is done using NHWC padding on the inputs (not
    // the outputs - they get written in NCHW!).
    layer_t curr_layer_nhwc_padded = *curr_layer;
    curr_layer_nhwc_padded.weights.align_pad =
            calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);
    curr_layer_nhwc_padded.inputs.align_pad =
            calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);

    conv_tiling_cfg cfg;

    // Decide how many L2 tiles we need.
    // Note that the kernels stored in the L2 cache are packed.
    size_t single_l2_kernel_size =
            get_nhwc_dims_size(&curr_layer_nhwc_padded.weights) *
            sizeof(float) / 2;
    // Make the number of kernerls in an L2 tile multiple of NUM_PE_INSTS.
    int max_kernels_per_l2_tile =
            (g_smv->kL2Size / single_l2_kernel_size / NUM_PE_INSTS) * NUM_PE_INSTS;
    // A simple way to decide how many L2 tiles we need.
    int num_l2_tiles = ceil((float)curr_layer_nhwc_padded.outputs.height /
                            max_kernels_per_l2_tile);

    size_t total_input_bytes =
            get_nhwc_dims_size(&curr_layer_nhwc_padded.inputs) * sizeof(float);
    bool need_input_tiling = (total_input_bytes > g_smv->kUmemSize);
    // The following variables need to be computed for creating tiles.
    int halo_rows;
    int max_rows_per_input_tile;
    int num_input_tiles;
    int first_input_tile_output_rows;
    int inner_input_tile_output_rows;
    int last_input_tile_output_rows;
    size_t first_input_tile_output_2d_size;
    size_t inner_input_tile_output_2d_size;
    size_t last_input_tile_output_2d_size;
    if (!need_input_tiling) {
        // If the whole input can fit on the UMEM, initialize the variables with
        // single input tile setting.
        halo_rows = 0;
        num_input_tiles = 1;
        max_rows_per_input_tile = curr_layer_nhwc_padded.inputs.rows;
        first_input_tile_output_rows = curr_layer_nhwc_padded.outputs.rows;
        inner_input_tile_output_rows = curr_layer_nhwc_padded.outputs.rows;
        last_input_tile_output_rows = curr_layer_nhwc_padded.outputs.rows;
        size_t output_2d_size = curr_layer_nhwc_padded.outputs.rows *
                                (curr_layer_nhwc_padded.outputs.cols +
                                 curr_layer_nhwc_padded.outputs.align_pad) *
                                sizeof(float);
        first_input_tile_output_2d_size = output_2d_size;
        inner_input_tile_output_2d_size = output_2d_size;
        last_input_tile_output_2d_size = output_2d_size;
    } else {
        // If the input can't fit on the UMEM, then we need to do input tiling.
        // The input is tiled based on a strip mining mechanism, the smallest
        // tile is of (K * W * H) layout format, where K is the kernel's length,
        // W is input's width, H is input's height.
        size_t single_row_size = curr_layer_nhwc_padded.inputs.cols *
                                   (curr_layer_nhwc_padded.inputs.height +
                                    curr_layer_nhwc_padded.inputs.align_pad) *
                                   sizeof(float);
        if (single_row_size * curr_layer_nhwc_padded.weights.rows >
            g_smv->kUmemSize) {
            printf("A single strip of the input image exceeds the capacity of "
                   "the UMEM, which is not supported!\n");
            assert(false);
        }
        // Divide up the input over strips.
        halo_rows = curr_layer_nhwc_padded.weights.rows -
                    curr_layer_nhwc_padded.stride.rows;
        max_rows_per_input_tile = g_smv->kUmemSize / single_row_size;
        num_input_tiles =
                ceil((float)(curr_layer_nhwc_padded.inputs.rows - halo_rows) /
                     (max_rows_per_input_tile - halo_rows));

        // Compute the output rows, output 2D feture map size for the first input
        // tile, last input tile and the inner tiles. These will be used for
        // creating output tiles.
        first_input_tile_output_rows =
                (max_rows_per_input_tile - curr_layer_nhwc_padded.weights.rows +
                 curr_layer_nhwc_padded.pad.top) /
                        curr_layer_nhwc_padded.stride.rows +
                1;
        first_input_tile_output_2d_size =
                first_input_tile_output_rows *
                (curr_layer_nhwc_padded.outputs.cols +
                 curr_layer_nhwc_padded.outputs.align_pad) *
                sizeof(float);
        inner_input_tile_output_rows =
                (max_rows_per_input_tile -
                 curr_layer_nhwc_padded.weights.rows) /
                        curr_layer_nhwc_padded.stride.rows +
                1;
        inner_input_tile_output_2d_size =
                inner_input_tile_output_rows *
                (curr_layer_nhwc_padded.outputs.cols +
                 curr_layer_nhwc_padded.outputs.align_pad) *
                sizeof(float);
        int num_rows_last_input_tile =
                curr_layer_nhwc_padded.inputs.rows -
                (max_rows_per_input_tile - halo_rows) * (num_input_tiles - 1);
        last_input_tile_output_rows =
                (num_rows_last_input_tile -
                 curr_layer_nhwc_padded.weights.rows +
                 curr_layer_nhwc_padded.pad.bottom) /
                        curr_layer_nhwc_padded.stride.rows +
                1;
        last_input_tile_output_2d_size =
                last_input_tile_output_rows *
                (curr_layer_nhwc_padded.outputs.cols +
                 curr_layer_nhwc_padded.outputs.align_pad) *
                sizeof(float);
    }

    if (first_input_tile_output_2d_size > g_smv->kSpadSize) {
        fprintf(stderr,
                "A single output channel of the input tile"
                "doesn't fit on the scratchpad! We "
                "don't support this mode of tiling yet!\n");
        assert(false);
    }

    // Divide up the work over output channels.
    // The number of output feature maps we can support at once is determined
    // by how many weights and output feature maps can fit into the two
    // scratchpads.
    int single_kernel_size =
            get_nhwc_dims_size(&curr_layer_nhwc_padded.weights) * sizeof(float);
    int max_kernels_per_output_tile = g_smv->kSpadSize / single_kernel_size;

    // Create tiling configurations.
    cfg.num_l2_tiles = num_l2_tiles;
    cfg.l2_tiles =
            (conv_l2_tile*)malloc(sizeof(conv_l2_tile) * num_l2_tiles);
    int remaining_kernels = curr_layer_nhwc_padded.outputs.height;
    for (int k = 0; k < cfg.num_l2_tiles; k++) {
        conv_l2_tile* l2_tile = &cfg.l2_tiles[k];
        l2_tile->num_input_tiles = num_input_tiles;
        l2_tile->input_tiles = (conv_input_tile*)malloc(
                l2_tile->num_input_tiles * sizeof(conv_input_tile));
        l2_tile->num_kernels = min2(max_kernels_per_l2_tile, remaining_kernels);
        remaining_kernels -= l2_tile->num_kernels;
        int remaining_input_rows = curr_layer_nhwc_padded.inputs.rows;
        for (int i = 0; i < l2_tile->num_input_tiles; i++) {
            bool first_input_tile = (i == 0);
            bool last_input_tile = (i == l2_tile->num_input_tiles - 1);
            conv_input_tile* input_tile = &l2_tile->input_tiles[i];
            input_tile->input_dims[0] = curr_layer_nhwc_padded.inputs.height;
            input_tile->input_dims[1] = curr_layer_nhwc_padded.inputs.cols;
            input_tile->input_dims[2] =
                    min2(remaining_input_rows, max_rows_per_input_tile);
            input_tile->input_dims[3] = NUM_TEST_CASES;
            input_tile->input_dims[4] = 1;
            input_tile->pad = curr_layer_nhwc_padded.pad;
            // If we have more than one input tile, we need to take care of zero
            // padding for all the input tiles. The first tile will have no bottom
            // padding, and the last tile will have no top padding. The rest will
            // have no top and bottom paddings.
            if (l2_tile->num_input_tiles > 1) {
                if (first_input_tile) {
                    input_tile->pad.bottom = 0;
                } else if (last_input_tile) {
                    input_tile->pad.top = 0;
                } else {
                    input_tile->pad.top = 0;
                    input_tile->pad.bottom = 0;
                }
            }
            input_tile->input_pad =
                    calc_padding(input_tile->input_dims[0], DATA_ALIGNMENT);
            // Compute the number of output tiles for this input tile.
            int max_ofmaps_per_output_tile;
            int num_ofmaps_per_output_tile;
            if (first_input_tile) {
                max_ofmaps_per_output_tile =
                        g_smv->kSpadSize / first_input_tile_output_2d_size;
            } else if (last_input_tile) {
                max_ofmaps_per_output_tile =
                        g_smv->kSpadSize / last_input_tile_output_2d_size;
            } else {
                max_ofmaps_per_output_tile =
                        g_smv->kSpadSize / inner_input_tile_output_2d_size;
            }
            num_ofmaps_per_output_tile =
                    min2(max_kernels_per_output_tile, max_ofmaps_per_output_tile);
            // Round down the number of output feature maps to the previous multiple
            // of NUM_PE_INSTS in order to maximine hardware utilization.
            if (num_ofmaps_per_output_tile > NUM_PE_INSTS) {
                num_ofmaps_per_output_tile =
                        (num_ofmaps_per_output_tile / NUM_PE_INSTS) * NUM_PE_INSTS;
            }
            input_tile->num_output_tiles = ceil(((float)l2_tile->num_kernels) /
                                                num_ofmaps_per_output_tile);
            input_tile->output_tiles = (conv_output_tile*)malloc(
                    sizeof(conv_output_tile) * input_tile->num_output_tiles);
            remaining_input_rows -= (max_rows_per_input_tile - halo_rows);
            int remaining_ofmaps = l2_tile->num_kernels;
            for (int j = 0; j < input_tile->num_output_tiles; j++) {
                conv_output_tile* output_tile = &input_tile->output_tiles[j];
                output_tile->num_ofmaps =
                        min2(remaining_ofmaps, num_ofmaps_per_output_tile);
                output_tile->output_dims[2] = output_tile->num_ofmaps;
                output_tile->output_dims[0] = curr_layer_nhwc_padded.outputs.cols;
                // Initialize the number of rows in the output of the tile.
                if (first_input_tile) {
                    output_tile->output_dims[1] = first_input_tile_output_rows;
                } else if (last_input_tile) {
                    output_tile->output_dims[1] = last_input_tile_output_rows;
                } else {
                    output_tile->output_dims[1] = inner_input_tile_output_rows;
                }
                output_tile->output_dims[3] = NUM_TEST_CASES;
                output_tile->output_dims[4] = 1;
                output_tile->output_pad =
                        calc_padding(output_tile->output_dims[0], DATA_ALIGNMENT);
                output_tile->num_hw_passes =
                        ceil((float)output_tile->num_ofmaps / NUM_PE_INSTS);
                output_tile->hw_passes = (smv_convolution_options*)malloc(
                        output_tile->num_hw_passes *
                        sizeof(smv_convolution_options));
                output_tile->hw_passes->load_inputs_first = true;
                remaining_ofmaps -= output_tile->num_ofmaps;
            }
        }
    }
    return cfg;
}

void smv_standard_convolution_layer_impl(data_list* host_activations,
                                         data_list* host_weights,
                                         layer_t* layers,
                                         int lnum,
                                         data_list* host_results,
                                         smv_global* g_smv,
                                         device_t* device,
                                         sampling_param_t* sampling_param) {
    require_data_type(host_weights, 0, UncompressedHalfPrecision);
    require_data_type(host_activations, 0, UncompressedHalfPrecision);

    layer_t curr_layer = layers[lnum];
    const int result_height = curr_layer.outputs.height;
    const int result_rows = curr_layer.outputs.rows;
    const int result_cols = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int input_height = curr_layer.inputs.height;
    const int input_rows = curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int k_rows = curr_layer.weights.rows;
    const int k_cols = curr_layer.weights.cols;

    data_list* nhwc_activations = init_data_list(1);
    begin_ignored_profiling(lnum);
    dims_t activations_nhwc = convert_nchw_to_nhwc(host_activations,
                                                   0,
                                                   NUM_TEST_CASES,
                                                   curr_layer.inputs,
                                                   DATA_ALIGNMENT,
                                                   nhwc_activations);
    end_profiling();
    packed_fp16* activations = nhwc_activations->data[0].dense_hp->d;
    ARRAY_4D(float16,
             _activations,
             activations,
             input_rows,
             input_cols,
             input_height + activations_nhwc.align_pad);
    dims_t nhwc_weights_dims = { curr_layer.weights.rows,
                                 curr_layer.weights.cols,
                                 curr_layer.weights.height,
                                 calc_padding(curr_layer.weights.height,
                                              DATA_ALIGNMENT) };
    // host_weights is half-precision, so it only occupies half the space that
    // the logical dimensions would indicate.
    ARRAY_4D(float16, _kernels, host_weights->data[0].dense_hp->d, k_rows,
             k_cols, input_height + nhwc_weights_dims.align_pad);
    ARRAY_4D(float16, _result, host_results->data[0].dense_hp->d,
             result_height, result_rows, result_cols + result_pad);

    conv_tiling_cfg tiling = convolution_divide_work(&curr_layer, g_smv);
    int64_t num_input_tiles = tiling.l2_tiles[0].num_input_tiles;
    int64_t num_output_tiles = tiling.l2_tiles[0].input_tiles[0].num_output_tiles;
    int64_t kernel_size = k_rows * k_cols *
                      (input_height + nhwc_weights_dims.align_pad);
    int64_t weight_size = kernel_size * result_height * sizeof(float16);
    int64_t ofmap_size =  result_rows * (result_cols + result_pad);
    int64_t input_size = input_rows * input_cols *
                     (input_height + activations_nhwc.align_pad) *
                     sizeof(float16);
    int64_t lat_dram = 140;
    int64_t lat_l2 = curr_layer.input_req == IO_ACP ? 40 : 140;
    int64_t orig_lat = (weight_size * num_input_tiles + input_size) * lat_dram;
    int64_t new_lat = (weight_size + input_size) * lat_dram;
    if (num_input_tiles > 1)
        new_lat += (num_output_tiles - 1) * input_size * lat_l2;
    if (new_lat < orig_lat) {
        if (kernel_size < ofmap_size) {
            printf("This is layer %d. The weight priority does potentially "
                   "give better performance, but our current implementation "
                   "doesn't support this case. It assumes that a single kernel "
                   "size is larger than a single channel of the output feature "
                   "maps.\n",
                   lnum);
        } else {
            INFO_MSG("Use new tiling for layer %d.\n", lnum);
            smv_wt_standard_convolution_layer_impl(host_activations,
                                                   host_weights,
                                                   layers,
                                                   lnum,
                                                   host_results,
                                                   g_smv,
                                                   device,
                                                   sampling_param);
            return;
        }
    }
    set_sampling_parameters(&tiling, &curr_layer, sampling_param);
    print_conv_tiling_cfg(&tiling, lnum);

    bool do_hw_activation = device->use_hw_activation_func &&
                            smiv_is_supported_activation_func(
                                    curr_layer.type, curr_layer.activation);
    bool use_pipelined_dma = device->use_pipelined_dma;
    begin_ignored_profiling(lnum);
    if (curr_layer.input_req == IO_DMA) {
        flush_cache_range(
                host_activations->data[0].dense_hp->d,
                host_activations->data[0].dense_hp->size * sizeof(float16));
        flush_cache_range(
                host_weights->data[0].dense_hp->d,
                host_weights->data[0].dense_hp->size * sizeof(float16));
    }
    if (do_hw_activation || curr_layer.output_req == IO_DMA) {
        flush_cache_range(
                host_results->data[0].dense_hp->d,
                host_results->data[0].dense_hp->size * sizeof(float16));
    }
    end_profiling();

    volatile int finish_flag;
#ifdef GEM5_HARNESS
    finish_flag = NOT_COMPLETED;
#else
    finish_flag = 0;
#endif
    // Outermost loop for batching.
    int halo_rows = curr_layer.weights.rows - curr_layer.stride.rows;
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        // Outermost loop for L2 tiling. One L2 tile blocks a suitable number
        // of kernels in the L2 cache, making reloading the weights more cache
        // friendly. The compromise of this L2 tiling is every L2 tile needs to
        // reload the inputs, hence, decrease the number of L2 tiles if the
        // reloading overheads of the inputs can't be offset by getting cheaper
        // weights reloading.
        int l2_tile_kern_start = 0;
        for (int lt = 0; lt < tiling.num_l2_tiles; lt++) {
            conv_l2_tile* l2_tile = &tiling.l2_tiles[lt];
            if (!l2_tile->execute) {
                run_sampled_l2_tile(
                        &curr_layer,
                        l2_tile,
                        img,
                        l2_tile_kern_start,
                        device,
                        nhwc_activations->data[0].dense_hp->d,
                        host_weights->data[0].dense_hp->d,
                        host_results->data[0].dense_hp->d);
                l2_tile_kern_start += l2_tile->num_kernels;
                continue;
            }
            begin_profiling("standard_convolution_layer_smv_l2_tile", lnum);
            if (l2_tile->sampling_upscale_factor > 1) {
                set_profiling_type_sampled(1, l2_tile->sampling_upscale_factor);
            }
            // Outer loop for input tiling. The input is tiled along rows, and the
            // output of an input tile is furtur tiled along output channels.
            // We need to track the start row number of the current input tile and
            // its result in the original input and result buffers, respectively.
            int input_row_start = 0;
            int result_row_start = 0;
            for (int it = 0; it < l2_tile->num_input_tiles; it++) {
                conv_input_tile* input_tile = &l2_tile->input_tiles[it];
                if (!input_tile->execute) {
                    run_sampled_input_tile(
                            &curr_layer,
                            input_tile,
                            l2_tile->num_input_tiles,
                            img,
                            result_row_start,
                            device,
                            &_activations[img][input_row_start][0][0],
                            &_kernels[l2_tile_kern_start][0][0][0],
                            host_results->data[0].dense_hp->d);
                    input_row_start += input_tile->input_dims[2] - halo_rows;
                    // All output tiles within an input tile produce the same
                    // number of output rows.
                    result_row_start += input_tile->output_tiles[0].output_dims[1];
                    continue;
                }
                // Prefetch the current input tile.
                if (device->use_sw_prefetch && curr_layer.input_req == IO_ACP) {
                    try_prefetch_input_activations(input_tile,
                                                   &_activations[0][0][0][0],
                                                   img,
                                                   input_row_start);
                }

                int output_tile_kern_start = l2_tile_kern_start;
                layer_t partial_layer;
                // Output tile sampling operates over the following loop. We always
                // execute the last output tile, because it has a different number
                // of output feature maps. We only run up to sampled_output_tiles
                // iterations.
                begin_profiling("standard_convolution_layer_smv_input_tile", lnum);
                if (input_tile->sampling_upscale_factor > 1) {
                    set_profiling_type_sampled(
                            1, input_tile->sampling_upscale_factor);
                }

                // Store all results from the accelerator in this temporary buffer.
                int result_buf_size = get_largest_output_tile_size(input_tile);
                fp16array_t* temp_result = init_fp16array(result_buf_size, true);

                // Inner loop for output tiling of an input tile.
                for (int ot = 0; ot < input_tile->num_output_tiles; ot++) {
                    conv_output_tile* output_tile = &input_tile->output_tiles[ot];
                    // NOTE: partial_layer's inputs are in NHWC format. So use
                    // get_nhwc_dims_size() to get the input size instead of
                    // get_dims_size().
                    partial_layer = create_partial_layer_from_tile(
                            &curr_layer, input_tile, output_tile);
                    // Only the first output tile will copy inputs data.
                    partial_layer.input_req =
                            (ot == 0) ? get_input_io_req(&partial_layer, device)
                                      : IO_NONE;
                    partial_layer.weights_req =
                            get_weights_io_req(&curr_layer,
                                               l2_tile->num_input_tiles,
                                               output_tile->num_hw_passes,
                                               device);
                    INFO_MSG("input req: %d, weights req: %d\n",
                             partial_layer.input_req,
                             partial_layer.weights_req);

                    // If there is only one HW pass, we can't delay the
                    // prefetch until the next HW pass, so we run the prefetch
                    // with a delay.
                    bool use_pf_delay = output_tile->num_hw_passes == 1;
                    if (!output_tile->execute) {
                        if (device->use_sw_prefetch &&
                            partial_layer.weights_req == IO_ACP) {
                            try_prefetch_weights(input_tile,
                                                 &_kernels[0][0][0][0],
                                                 &nhwc_weights_dims,
                                                 output_tile_kern_start,
                                                 ot,
                                                 use_pf_delay);
                            // So we don't start too many threads during
                            // sampling, wait for all threads to join.
                            thread_pool_join();
                        }
                        run_sampled_output_tile(
                                &curr_layer,
                                output_tile,
                                l2_tile->num_input_tiles,
                                img,
                                output_tile_kern_start,
                                result_row_start,
                                device,
                                &_kernels[0][0][0][0],
                                host_results->data[0].dense_hp->d);
                        output_tile_kern_start += output_tile->num_ofmaps;
                        continue;
                    }
                    begin_profiling(
                            "standard_convolution_layer_smv_output_tile", lnum);
                    if (output_tile->sampling_upscale_factor > 1) {
                        set_profiling_type_sampled(
                                1, output_tile->sampling_upscale_factor);
                    }

                    // Set up input location and mappings.
                    float16* activations_loc =
                            &_activations[img][input_row_start][0][0];
                    if (partial_layer.input_req != IO_NONE) {
                        MAP_ARRAY_TO_ACCEL(
                                g_smv->kConvolutionHw,
                                get_host_inputs_var_name(
                                        partial_layer.input_req),
                                activations_loc,
                                get_nhwc_dims_size(&partial_layer.inputs) *
                                        sizeof(float16));
                    }

                    // Convert weights to NHWC and set up mappings.
                    float16* weights_loc =
                            &_kernels[output_tile_kern_start][0][0][0];
                    MAP_ARRAY_TO_ACCEL(
                            g_smv->kConvolutionHw,
                            get_host_weights_var_name(
                                    partial_layer.weights_req),
                            weights_loc,
                            output_tile->num_ofmaps *
                                    get_nhwc_dims_size(&nhwc_weights_dims) *
                                    sizeof(float16));

                    // Sampling operates on iterations of the following loop over
                    // num_hw_iters. We always execute the first and last
                    // iterations, because those iterations are responsible for
                    // handling data movement. Between those two iterations, we only
                    // run up to sampled_inner_iter iterations. The remaining
                    // iterations have their outputs set to zero.
                    for (int iter = 0; iter < output_tile->num_hw_passes; iter++) {
                        // Prefetch weights for the next output tile.
                        //
                        // We can do this either only on the second hardware
                        // pass, or on the first hardware pass if there is only
                        // HW pass. It's generally better to prefetch on the second
                        // HW pass so we don't cause contention for L2
                        // bandwidth when the accelerator is loading data too.
                        bool do_prefetch =
                                device->use_sw_prefetch &&
                                (partial_layer.weights_req == IO_ACP) &&
                                (ot != input_tile->num_output_tiles - 1) &&
                                ((iter == 1) ||
                                 (output_tile->num_hw_passes == 1));
                        if (do_prefetch) {
                            try_prefetch_weights(input_tile,
                                                 &_kernels[0][0][0][0],
                                                 &nhwc_weights_dims,
                                                 output_tile_kern_start,
                                                 ot,
                                                 use_pf_delay);
                        }
                        smv_convolution_options* options =
                                &output_tile->hw_passes[iter];
                        options->img = img;
                        // This kern_start is with respect to the current set of
                        // output fmaps in the tile.
                        options->kern_start = iter * NUM_PE_INSTS;
                        int num_kerns_this_iter =
                                min2(output_tile->num_ofmaps - options->kern_start,
                                     (int)NUM_PE_INSTS);
                        options->kern_end = options->kern_start + num_kerns_this_iter;
                        // This is required to DMA the correct number of weights and
                        // outputs back from the accelerator at the beginning and
                        // end.
                        options->total_tile_ofmaps = output_tile->num_ofmaps;
                        options->use_pipelined_dma = use_pipelined_dma;
                        if (!options->execute) {
                            run_sampled_hw_pass(&partial_layer,
                                                options,
                                                img,
                                                temp_result->d);
                            continue;
                        }

                        // Map the result array to the accelerator. We have to
                        // do this per hw iteration because the sampled passes
                        // will overwrite the mapping for this array.
                        MAP_ARRAY_TO_ACCEL(
                                g_smv->kConvolutionHw,
                                get_host_results_var_name(
                                        partial_layer.output_req),
                                temp_result->d,
                                temp_result->size * sizeof(packed_fp16));

                        // Only copy weights and inputs on the first iteration.
                        if (iter > 0) {
                            if (partial_layer.weights_req == IO_DMA ||
                                partial_layer.weights_req == IO_ACP)
                                partial_layer.weights_req = IO_NONE;
                            if (partial_layer.input_req == IO_DMA ||
                                partial_layer.input_req == IO_ACP)
                                partial_layer.input_req = IO_NONE;
                        }
                        // Only run the activation function on the last iteration.
                        partial_layer.activation = (do_hw_activation)
                                                           ? curr_layer.activation
                                                           : NO_ACTIVATION;
                        access_config access_cfg =
                                layer_to_access_config(&partial_layer);
                        INFO_MSG("Layer %d: l2 tile %d, input tile %d, "
                                 "output_tile %d, hw_pass %d: input_req = %d, "
                                 "output_req = %d, weights_req = %d\n",
                                 partial_layer.num, lt, it, ot, iter,
                                 partial_layer.input_req,
                                 partial_layer.output_req,
                                 partial_layer.weights_req);
                        INVOKE_KERNEL_SAMPLED(g_smv->kConvolutionHw,
                                              lnum,
                                              options->sampling_upscale_factor,
                                              smv_convolution_layer_hw,
                                              // DMA
                                              (packed_fp16*)activations_loc,
                                              (packed_fp16*)weights_loc,
                                              temp_result->d,
                                              // Cache
                                              (packed_fp16*)activations_loc,
                                              (packed_fp16*)weights_loc,
                                              temp_result->d,
                                              // ACP
                                              (packed_fp16*)activations_loc,
                                              (packed_fp16*)weights_loc,
                                              temp_result->d,
                                              g_smv->umem,
                                              g_smv->spad0,
                                              g_smv->spad1,
                                              partial_layer,
                                              &access_cfg,
                                              options);
                    }

                    // Reorganize the temporary results into the host result buffer.
                    int partial_result_2d_size =
                            partial_layer.outputs.rows *
                            (partial_layer.outputs.cols +
                             partial_layer.outputs.align_pad);
                    begin_ignored_profiling(curr_layer.num);
                    for (int k = 0; k < output_tile->num_ofmaps; k++) {
                        memcpy(&_result[img][k + output_tile_kern_start]
                                       [result_row_start][0],
                               temp_result->d +
                                       (partial_result_2d_size * k) / 2,
                               partial_result_2d_size * sizeof(float16));
                    }
                    end_profiling();

                    output_tile_kern_start += output_tile->num_ofmaps;
                    end_profiling();  // standard_convolution_layer_smv_output_tile
                }
                free_fp16array(temp_result);
                end_profiling(); // standard_convolution_layer_smv_input_tile
                INFO_MSG("Finished an input tile.\n");
                input_row_start += (partial_layer.inputs.rows - halo_rows);
                result_row_start += partial_layer.outputs.rows;
            }
            end_profiling();  // standard_convolution_layer_smv_l2_tile
            l2_tile_kern_start += l2_tile->num_kernels;
        }
    }
    free_data_list(nhwc_activations);
    free_conv_tiling_cfg(&tiling);
}
