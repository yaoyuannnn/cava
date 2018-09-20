#include <string.h>

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

typedef struct _conv_wt_input_tile {
    int input_dims[5];
    int output_dims[5];
    int input_pad;
    padding pad;
    smv_convolution_options* hw_passes;
    int num_hw_passes;
    int sampling_upscale_factor;
    bool execute;
} conv_wt_input_tile;

typedef struct _conv_wt_output_tile {
    int output_pad;
    int num_ofmaps;
    conv_wt_input_tile* input_tiles;
    int num_input_tiles;
    int sampling_upscale_factor;
    bool execute;
} conv_wt_output_tile;

typedef struct _conv_wt_tiling_cfg {
    conv_wt_output_tile* output_tiles;
    int num_output_tiles;
} conv_wt_tiling_cfg;

void free_conv_wt_tiling_cfg(conv_wt_tiling_cfg* cfg) {
    for (int i = 0; i < cfg->num_output_tiles; i++) {
        conv_wt_output_tile* output_tile = &cfg->output_tiles[i];
        for (int j = 0; j < output_tile->num_input_tiles; j++) {
            free(output_tile->input_tiles[j].hw_passes);
        }
        free(output_tile->input_tiles);
    }
    free(cfg->output_tiles);
}

void print_conv_wt_tiling_cfg(conv_wt_tiling_cfg* cfg, int lnum) {
    INFO_MSG("\nTiling info for layer %d\n", lnum);
    INFO_MSG("\nNumber of output tiles %d\n", cfg->num_output_tiles);
    for (int i = 0; i < cfg->num_output_tiles; i++) {
        conv_wt_output_tile* output_tile = &cfg->output_tiles[i];
        INFO_MSG("  + Output tile %d\n"
                 "      Execute: %s\n"
                 "      OFMaps: %d\n"
                 "      output pad: %d\n"
                 "      Each tile represents: %d output tiles\n"
                 "      Input tiles: %d\n",
                 i,
                 bool_to_yesno(output_tile->execute),
                 output_tile->num_ofmaps,
                 output_tile->output_pad,
                 output_tile->sampling_upscale_factor,
                 output_tile->num_input_tiles);
        for (int j = 0; j < output_tile->num_input_tiles; j++) {
            conv_wt_input_tile* input_tile =
                    &output_tile->input_tiles[j];
            INFO_MSG("    + Input tile %d:\n"
                     "        Execute: %s\n"
                     "        IFMap size: %d, %d, %d, %d, %d\n"
                     "        OFMap size: %d, %d, %d, %d, %d\n"
                     "        zero padding: %d, %d, %d, %d\n"
                     "        input pad %d\n"
                     "        Each tile represents: %d input tiles\n"
                     "        Num HW passes: %d\n",
                     j,
                     bool_to_yesno(input_tile->execute),
                     input_tile->input_dims[0],
                     input_tile->input_dims[1],
                     input_tile->input_dims[2],
                     input_tile->input_dims[3],
                     input_tile->input_dims[4],
                     input_tile->output_dims[0],
                     input_tile->output_dims[1],
                     input_tile->output_dims[2],
                     input_tile->output_dims[3],
                     input_tile->output_dims[4],
                     input_tile->pad.top,
                     input_tile->pad.bottom,
                     input_tile->pad.left,
                     input_tile->pad.right,
                     input_tile->input_pad,
                     input_tile->sampling_upscale_factor,
                     input_tile->num_hw_passes);
            for (int k = 0; k < input_tile->num_hw_passes; k++) {
                smv_convolution_options* hw_options =
                        &input_tile->hw_passes[k];
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

// Determine whether to use ACP or DMA for the weights for this output tile.
io_req_t get_wt_weights_io_req(layer_t* curr_layer, device_t* device) {
    if (device->weights_load_policy == UseDefaultOffload) {
        return curr_layer->weights_req;
    } else if (device->weights_load_policy == AcpAlways) {
        // In this tiling strategy, weights are never reused, so we only want to
        // use ACP if the user explicitly requested it.
        return IO_ACP;
    }
    return IO_DMA;
}

// Determine whether to use ACP or DMA for the weights for this output tile.
io_req_t get_wt_inputs_io_req(layer_t* curr_layer,
                              int total_output_tiles,
                              int num_hw_passes,
                              device_t* device) {
    if (device->inputs_load_policy == UseDefaultOffload) {
        return curr_layer->input_req;
    } else if (device->inputs_load_policy == DmaAlways) {
        return IO_DMA;
    } else if (device->inputs_load_policy == AcpAlways) {
        return IO_ACP;
    } else if (device->inputs_load_policy == DynamicDmaAcp) {
        // If there is only one output tile, there is no reuse of inputs across
        // output tiles. However, we can still take advantage of the cache for
        // software-based double buffering if there are multiple hardware
        // passes per input tile.
        if (total_output_tiles == 1) {
            if (!device->use_sw_prefetch || num_hw_passes == 1)
                return IO_DMA;
        }
        return IO_ACP;
    } else {
        assert(false && "Unknown data movement load policy!");
        return IO_DMA;
    }
}

// Attempt to prefetch the next input tile.
//
// Arguments:
//    output_tile: The current output tile.
//    inputs: The base address of the complete input tensor.
//    input_dims: The dimensions of the complete inputs, padded in NHWC.
//    weights_dims: The dimensions of the complete weights, padded in NHWC. We
//       need this so we can correctly compute the delay that should be applied
//       to the prefetch operation if needed.
//    img: The N'th input image.
//    input_tile_idx: The N'th input tile.
//    input_row_start: The CURRENT input tile's starting row.
//    halo_rows: The size of the row halo region.
//    use_pf_delay: Apply a delay to this prefetch operation to avoid creating
//       contention for L2 bandwidth.
void wt_try_prefetch_input_activations(conv_wt_output_tile* output_tile,
                                       float16* inputs,
                                       dims_t* inputs_dims,
                                       dims_t* weights_dims,
                                       int img,
                                       int input_tile_idx,
                                       int input_row_start,
                                       int halo_rows,
                                       bool use_pf_delay) {
#ifdef TRACE_MODE
    return;
#endif
    // The range argument is freed by the worker thread when it is finished.
    pf_memory_range* range = (pf_memory_range*)malloc(sizeof(pf_memory_range));
    conv_wt_input_tile* curr_input_tile =
            &output_tile->input_tiles[input_tile_idx];
    conv_wt_input_tile* next_input_tile =
            &output_tile->input_tiles[input_tile_idx + 1];
    range->bytes =
            (next_input_tile->input_dims[0] + next_input_tile->input_pad) *
            next_input_tile->input_dims[1] * next_input_tile->input_dims[2] *
            sizeof(float16);
    if (range->bytes < SMV_CONV_SW_PREFETCH_THRESHOLD)
        return;
    ARRAY_4D(float16,
             _inputs,
             inputs,
             inputs_dims->rows,
             inputs_dims->cols,
             inputs_dims->height + inputs_dims->align_pad);
    int next_row_start =
            input_row_start + curr_input_tile->input_dims[2] - halo_rows;
    range->base_addr = (char*)&_inputs[img][next_row_start][0][0];
    range->prefetch_offset = 0;

    if (use_pf_delay) {
        // To prevent the prefetch from competing for L2 bandwidth with the
        // accelerator when loading the current tile data, delay the prefetch
        // until we think the accelerator is done loading all inputs and
        // weights.
        //
        // Every output tile will need to load new inputs, but only the first
        // also needs to load new weights.
        int current_tile_load_size =
                (curr_input_tile->input_dims[0] + curr_input_tile->input_pad) *
                curr_input_tile->input_dims[1] * curr_input_tile->input_dims[2];
        if (input_tile_idx == 0) {
            current_tile_load_size +=
                    output_tile->num_ofmaps * get_nhwc_dims_size(weights_dims);
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

// Attempt to prefetch the weights for the **CURRENT** output tile.
//
// Arguments:
//    weights: The base address of the 16-bit weights.
//    weights_dims: The weights dimensions with NHWC padding.
//    ofmap_start: Which ofmap to start from.
//    num_ofmaps: THe number of kernels to fetch.
void wt_try_prefetch_weights(float16* weights,
                             dims_t* weights_dims,
                             int ofmap_start,
                             int num_ofmaps) {
#ifdef TRACE_MODE
    return;
#endif
    pf_memory_range* range = (pf_memory_range*)malloc(sizeof(pf_memory_range));
    ARRAY_4D(float16,
             _weights,
             weights,
             weights_dims->rows,
             weights_dims->cols,
             weights_dims->height + weights_dims->align_pad);
    range->base_addr = (char*)&_weights[ofmap_start][0][0][0];
    range->bytes =
            get_nhwc_dims_size(weights_dims) * num_ofmaps * sizeof(float16);
    range->prefetch_offset = 0;
    range->prefetch_delay_ns = 0;
    thread_dispatch(prefetch_memory_range, (void*)range);
}

//=------- Functions to capture cache behavior of sampled tiles --------=//
//
// When we sample, we entirely skip the computation for that tile, but that
// also means we don't touch the caches, so our working set is smaller. To
// model the original working set size, these functions set the relevant
// regions of memory to zero, thus ensuring that those memory regions are
// touched.  We then ignore the cycles in profiling.

// To mimic an entire input tile, set the correct slices (rows) of each output
// feature map in this input tile to zero.
void run_wt_sampled_input_tile(layer_t* curr_layer,
                               conv_wt_output_tile* output_tile,
                               conv_wt_input_tile* input_tile,
                               int total_output_tiles,
                               int img,
                               int ofmap_start,
                               int ofmap_row_start,
                               device_t* device,
                               float16* activations,
                               packed_fp16* results_base) {
    begin_ignored_profiling(curr_layer->num);
    ARRAY_4D(float16, _result, results_base, curr_layer->outputs.height,
             curr_layer->outputs.rows,
             curr_layer->outputs.cols + curr_layer->outputs.align_pad);
    size_t input_tile_size =
            sizeof(float16) * input_tile->input_dims[2] *
            input_tile->input_dims[1] *
            (input_tile->input_dims[0] + input_tile->input_pad);
    io_req_t input_req = get_wt_inputs_io_req(
            curr_layer, total_output_tiles, input_tile->num_hw_passes, device);
    MAP_ARRAY_TO_ACCEL(g_smv.kConvolutionHw,
                       get_host_inputs_var_name(input_req),
                       activations,
                       input_tile_size);
    if (input_req == IO_DMA) {
        INVOKE_DMA_READ_TRAFFIC_GEN(activations, input_tile_size);
    } else if (input_req == IO_ACP) {
        INVOKE_ACP_READ_TRAFFIC_GEN(activations, input_tile_size);
    }
    int ofmap_size = sizeof(float16) * input_tile->output_dims[1] *
                     (input_tile->output_dims[0] + output_tile->output_pad);
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

// To mimic an entire output tile, set the entire slices of all the output
// feature maps to zero.
void run_wt_sampled_output_tile(layer_t* curr_layer,
                                conv_wt_output_tile* output_tile,
                                int num_output_tiles,
                                int img,
                                int output_tile_kern_start,
                                device_t* device,
                                float16* activations,
                                float16* kernels,
                                packed_fp16* results_base) {
    begin_ignored_profiling(curr_layer->num);
    int input_pad = calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);
    ARRAY_4D(float16,
             _activations,
             activations,
             curr_layer->inputs.rows,
             curr_layer->inputs.cols,
             curr_layer->inputs.height + input_pad);
    int weight_pad = calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);
    int single_kernel_size = curr_layer->weights.rows *
                             curr_layer->weights.cols *
                             (curr_layer->weights.height + weight_pad);
    int kernel_size =
            sizeof(float16) * single_kernel_size * output_tile->num_ofmaps;
    io_req_t weights_req = get_wt_weights_io_req(curr_layer, device);
    MAP_ARRAY_TO_ACCEL(g_smv.kConvolutionHw,
                       get_host_weights_var_name(weights_req),
                       kernels,
                       kernel_size);
    if (weights_req == IO_DMA) {
        INVOKE_DMA_READ_TRAFFIC_GEN(kernels, kernel_size);
    } else if (weights_req == IO_ACP) {
        INVOKE_ACP_READ_TRAFFIC_GEN(kernels, kernel_size);
    }
    int halo_rows = curr_layer->weights.rows - curr_layer->stride.rows;
    int input_row_start = 0;
    int result_row_start = 0;
    for (int i = 0; i < output_tile->num_input_tiles; i++) {
        conv_wt_input_tile* input_tile = &output_tile->input_tiles[i];
        run_wt_sampled_input_tile(curr_layer,
                                  output_tile,
                                  input_tile,
                                  num_output_tiles,
                                  img,
                                  output_tile_kern_start,
                                  result_row_start,
                                  device,
                                  &_activations[img][input_row_start][0][0],
                                  results_base);
        input_row_start += input_tile->input_dims[2] - halo_rows;
        result_row_start += input_tile->output_dims[1];
    }
    end_profiling();
}

static layer_t create_wt_partial_layer_from_tile(
        layer_t* full_layer,
        conv_wt_input_tile* input_tile,
        conv_wt_output_tile* output_tile) {
    layer_t partial_layer = *full_layer;
    partial_layer.inputs.rows = input_tile->input_dims[2];
    partial_layer.inputs.cols = input_tile->input_dims[1];
    partial_layer.inputs.height = input_tile->input_dims[0];
    partial_layer.inputs.align_pad = input_tile->input_pad;
    partial_layer.outputs.rows = input_tile->output_dims[1];
    partial_layer.outputs.cols = input_tile->output_dims[0];
    partial_layer.outputs.height = input_tile->output_dims[2];
    partial_layer.outputs.align_pad = output_tile->output_pad;
    partial_layer.weights.height = input_tile->input_dims[0];
    partial_layer.weights.align_pad = input_tile->input_pad;
    partial_layer.pad = input_tile->pad;
    return partial_layer;
}

/* Return the largest amount of output pixels produced by an input tile for a
 * given output tile. */
int get_largest_wt_output_tile_size(conv_wt_output_tile* output_tile) {
    int max = 0;
    for (int i = 0; i < output_tile->num_input_tiles; i++) {
        conv_wt_input_tile* input_tile = &output_tile->input_tiles[i];
        max = max2(max,
                   (input_tile->output_dims[0] + output_tile->output_pad) *
                           input_tile->output_dims[1] *
                           input_tile->output_dims[2]);
    }
    return max;
}

/* For input tile, output tile, and HW pass, determine if it should be executed.
 *
 * When we sample, we skip certain output tiles and HW passes based on the
 * appropriate sampling parameter. This function sets the execute bit for each
 * of these tiles depending on whether it should be skipped or not. If a tile /
 * HW pass is not skipped, we also set the appropriate sampling upscaling
 * factor.
 */
static void set_wt_sampling_parameters(conv_wt_tiling_cfg* conv_tiling,
                                    layer_t* curr_layer,
                                    sampling_param_t* sampling_param) {
    // These parameters indicate the number of tiles/hw passes to run *IN
    // ADDITION* to the minimum required amount. At a minimum, we must run the
    // first and last input output tiles and HW passes. So if
    // sampled_outer_tiles = 1, then the total number of executed output tiles
    // is 3.
    int sampled_input_tiles = sampling_param->smv_conv_input_tiles;
    int sampled_output_tiles = sampling_param->smv_conv_output_tiles;
    int sampled_inner_iters = sampling_param->smv_conv_inner_iters;

    bool do_output_tile_sampling =
            (sampled_output_tiles > 0 &&
             sampled_output_tiles < conv_tiling->num_output_tiles - 2);
    int sampling_output_tiles_upscale_factor;
    if (do_output_tile_sampling) {
        sampling_output_tiles_upscale_factor =
                ceil(((float)conv_tiling->num_output_tiles - 2) /
                     sampled_output_tiles);
    } else {
        sampling_output_tiles_upscale_factor = 1;
    }
    int output_tiles_remaining = conv_tiling->num_output_tiles;
    for (int ot = 0; ot < conv_tiling->num_output_tiles; ot++) {
        conv_wt_output_tile* output_tile = &conv_tiling->output_tiles[ot];
        bool is_first_or_last_output_tile =
                (ot == 0 || ot == conv_tiling->num_output_tiles - 1);
        if (is_first_or_last_output_tile) {
            output_tile->execute = true;
            output_tile->sampling_upscale_factor = 1;
        } else {
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

        // The first input tile may need to handle input activation DMA, and
        // the last input tile may be smaller than the rest, so they always
        // need to be executed.
        bool do_input_tile_sampling =
                (sampled_input_tiles > 0 &&
                 sampled_input_tiles < output_tile->num_input_tiles - 2);

        // If we sample input tiles, how many input tiles does each executed
        // tile represent?
        int sampling_input_tiles_upscale_factor;
        if (do_input_tile_sampling) {
            sampling_input_tiles_upscale_factor =
                    ceil(((float)output_tile->num_input_tiles - 2) /
                         sampled_input_tiles);
        } else {
            sampling_input_tiles_upscale_factor = 1;
        }
        int input_tiles_remaining = output_tile->num_input_tiles;
        for (int it = 0; it < output_tile->num_input_tiles; it++) {
            conv_wt_input_tile* input_tile = &output_tile->input_tiles[it];
            bool is_first_or_last_input_tile =
                    (it == 0 || it == output_tile->num_input_tiles - 1);
            if (is_first_or_last_input_tile || !do_input_tile_sampling) {
                input_tile->execute = true;
                input_tile->sampling_upscale_factor = 1;
            } else {
                // Compare the remaining input tile with one, not zero,
                // because we always need to execute the last iteration.
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

            // Each input tile may requires multiple HW passes to execute. We
            // only sample if we must run three or more.
            bool do_hw_pass_sampling =
                    sampled_inner_iters > 0 &&
                    sampled_inner_iters < input_tile->num_hw_passes - 2;
            int remaining_hw_passes = input_tile->num_hw_passes;
            // If we sample HW passes, how many HW passes does each executed
            // pass represent?
            int sampling_hw_passes_upscale_factor;
            if (do_hw_pass_sampling) {
                sampling_hw_passes_upscale_factor =
                        ceil(((float)input_tile->num_hw_passes - 2) /
                             sampled_inner_iters);
            } else {
                sampling_hw_passes_upscale_factor = 1;
            }

            for (int hw_pass = 0; hw_pass < input_tile->num_hw_passes;
                 hw_pass++) {
                smv_convolution_options* hw_options =
                        &input_tile->hw_passes[hw_pass];
                bool is_first_or_last_hw_pass =
                        (hw_pass == 0) ||
                        (hw_pass == input_tile->num_hw_passes - 1);
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

static conv_wt_tiling_cfg convolution_wt_divide_work(layer_t* curr_layer,
                                               smv_global* g_smv) {
    // Ensure that all the tiling is done using NHWC padding on the inputs (not
    // the outputs - they get written in NCHW!).
    layer_t curr_layer_nhwc_padded = *curr_layer;
    curr_layer_nhwc_padded.weights.align_pad =
            calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);
    curr_layer_nhwc_padded.inputs.align_pad =
            calc_padding(curr_layer->inputs.height, DATA_ALIGNMENT);

    conv_wt_tiling_cfg cfg;

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
                (max_rows_per_input_tile - curr_layer_nhwc_padded.weights.rows) /
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
                (num_rows_last_input_tile - curr_layer_nhwc_padded.weights.rows +
                 curr_layer_nhwc_padded.pad.bottom) /
                        curr_layer_nhwc_padded.stride.rows +
                1;
        last_input_tile_output_2d_size =
                last_input_tile_output_rows *
                (curr_layer_nhwc_padded.outputs.cols +
                 curr_layer_nhwc_padded.outputs.align_pad) *
                sizeof(float);

        if (first_input_tile_output_2d_size > g_smv->kSpadSize) {
            fprintf(stderr,
                    "A single output channel of the input tile"
                    "doesn't fit on the scratchpad! We "
                    "don't support this mode of tiling yet!\n");
            assert(false);
        }
    }

    // Divide up the work over output channels.
    // The number of output feature maps we can support at once is determined
    // by how many weights and output feature maps can fit into the two
    // scratchpads.
    int single_kernel_size =
            get_nhwc_dims_size(&curr_layer_nhwc_padded.weights) * sizeof(float);
    int max_kernels_per_output_tile = g_smv->kSpadSize / single_kernel_size;
    // Round down the number of output feature maps to the previous multiple
    // of NUM_PE_INSTS in order to maximine hardware utilization.
    if (max_kernels_per_output_tile > NUM_PE_INSTS) {
        max_kernels_per_output_tile =
                (max_kernels_per_output_tile / NUM_PE_INSTS) * NUM_PE_INSTS;
    }

    // Create tiling configurations.
    cfg.num_output_tiles = ceil(((float)curr_layer_nhwc_padded.outputs.height) /
                                max_kernels_per_output_tile);
    cfg.output_tiles = (conv_wt_output_tile*)malloc(sizeof(conv_wt_output_tile) *
                                                 cfg.num_output_tiles);
    int remaining_ofmaps = curr_layer_nhwc_padded.outputs.height;
    for (int k = 0; k < cfg.num_output_tiles; k++) {
        conv_wt_output_tile* output_tile = &cfg.output_tiles[k];
        output_tile->num_input_tiles = num_input_tiles;
        output_tile->input_tiles = (conv_wt_input_tile*)malloc(
                output_tile->num_input_tiles * sizeof(conv_wt_input_tile));
        output_tile->num_ofmaps =
                min2(remaining_ofmaps, max_kernels_per_output_tile);
        output_tile->output_pad = calc_padding(
                curr_layer_nhwc_padded.outputs.cols, DATA_ALIGNMENT);

        int remaining_input_rows = curr_layer_nhwc_padded.inputs.rows;
        for (int i = 0; i < output_tile->num_input_tiles; i++) {
            bool first_input_tile = (i == 0);
            bool last_input_tile = (i == output_tile->num_input_tiles - 1);
            conv_wt_input_tile* input_tile = &output_tile->input_tiles[i];

            int max_ofmaps_per_output_tile;
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
            assert(max_kernels_per_output_tile <= max_ofmaps_per_output_tile);

            input_tile->output_dims[2] = output_tile->num_ofmaps;
            if (first_input_tile) {
                input_tile->output_dims[1] = first_input_tile_output_rows;
            } else if (last_input_tile) {
                input_tile->output_dims[1] = last_input_tile_output_rows;
            } else {
                input_tile->output_dims[1] = inner_input_tile_output_rows;
            }
            input_tile->output_dims[0] = curr_layer_nhwc_padded.outputs.cols;
            input_tile->output_dims[3] = NUM_TEST_CASES;
            input_tile->output_dims[4] = 1;

            input_tile->input_dims[0] = curr_layer_nhwc_padded.inputs.height;
            input_tile->input_dims[1] = curr_layer_nhwc_padded.inputs.cols;
            input_tile->input_dims[2] =
                    min2(remaining_input_rows, max_rows_per_input_tile);
            input_tile->input_dims[3] = NUM_TEST_CASES;
            input_tile->input_dims[4] = 1;
            input_tile->pad = curr_layer_nhwc_padded.pad;
            // We need to take care of zero padding for all the input tiles. The
            // first tile will have no bottom padding, and the last tile will have
            // no top padding. The rest will have no top and bottom paddings.
            if (output_tile->num_input_tiles > 1) {
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

            remaining_input_rows -= (max_rows_per_input_tile - halo_rows);
            input_tile->num_hw_passes =
                    ceil((float)output_tile->num_ofmaps / NUM_PE_INSTS);
            input_tile->hw_passes = (smv_convolution_options*)malloc(
                    input_tile->num_hw_passes *
                    sizeof(smv_convolution_options));
            input_tile->hw_passes->load_inputs_first = false;
        }
        remaining_ofmaps -= output_tile->num_ofmaps;
    }
    return cfg;
}

void smv_wt_standard_convolution_layer_impl(data_list* host_activations,
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
    convert_nchw_to_nhwc(host_activations,
                         0,
                         NUM_TEST_CASES,
                         curr_layer.inputs,
                         DATA_ALIGNMENT,
                         nhwc_activations);
    dims_t nhwc_activations_dims = { input_rows,
                                input_cols,
                                input_height,
                                calc_padding(input_height, DATA_ALIGNMENT) };
    end_profiling();
    packed_fp16* activations = nhwc_activations->data[0].dense_hp->d;
    ARRAY_4D(float16,
             _activations,
             activations,
             input_rows,
             input_cols,
             input_height + nhwc_activations_dims.align_pad);
    // TODO: Add metadata to indicate the size of elements contained inside
    // DataFormat.

    // XXX: host_weights arrives in NHWC format, but layer.weights is still in
    // NCHW dimension format.
    dims_t nhwc_weights_dims =
            nchw_to_nhwc_dims(&curr_layer.weights, DATA_ALIGNMENT);
    // host_weights is half-precision, so it only occupies half the space that
    // the logical dimensions would indicate.
    ARRAY_4D(float16, _kernels, host_weights->data[0].dense_hp->d, k_rows,
             k_cols, input_height + nhwc_weights_dims.align_pad);
    ARRAY_4D(float16, _result, host_results->data[0].dense_hp->d,
             result_height, result_rows, result_cols + result_pad);

    conv_wt_tiling_cfg tiling = convolution_wt_divide_work(&curr_layer, g_smv);
    set_wt_sampling_parameters(&tiling, &curr_layer, sampling_param);
    print_conv_wt_tiling_cfg(&tiling, lnum);

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

    // Outermost loop for batching.
    int halo_rows = curr_layer.weights.rows - curr_layer.stride.rows;
    for (int img = 0; img < NUM_TEST_CASES; img++) {
        int output_tile_kern_start = 0;
        // Outer loop for output tiling.
        for (int ot = 0; ot < tiling.num_output_tiles; ot++) {
            conv_wt_output_tile* output_tile = &tiling.output_tiles[ot];
            if (!output_tile->execute) {
                run_wt_sampled_output_tile(
                        &curr_layer,
                        output_tile,
                        tiling.num_output_tiles,
                        img,
                        output_tile_kern_start,
                        device,
                        &_activations[0][0][0][0],
                        &_kernels[output_tile_kern_start][0][0][0],
                        host_results->data[0].dense_hp->d);
                output_tile_kern_start += output_tile->num_ofmaps;
                continue;
            }

            if (device->use_sw_prefetch && curr_layer.input_req == IO_ACP) {
                wt_try_prefetch_weights(
                        &_kernels[0][0][0][0],
                        &nhwc_weights_dims,
                        output_tile_kern_start,
                        output_tile->num_ofmaps);
            }

            layer_t partial_layer;
            // Output tile sampling operates over the following loop. We always
            // execute the last output tile, because it has a different number
            // of output feature maps. We only run up to sampled_output_tiles
            // iterations.
            begin_profiling("standard_convolution_layer_smv_output_tile", lnum);
            if (output_tile->sampling_upscale_factor > 1) {
                set_profiling_type_sampled(
                        1, output_tile->sampling_upscale_factor);
            }

            // Set up the results buffer and mappings.
            int result_buf_size = get_largest_wt_output_tile_size(output_tile);
            fp16array_t* temp_result = init_fp16array(result_buf_size, true);

            // Inner loop for input tiling. The input is tiled along rows, and
            // the output of an input tile is furtur tiled along output channels.
            // We need to track the start row number of the current input tile
            // and its result in the original input and result buffers,
            // respectively.
            int input_row_start = 0;
            int result_row_start = 0;
            for (int it = 0; it < output_tile->num_input_tiles; it++) {
                conv_wt_input_tile* input_tile = &output_tile->input_tiles[it];
                // NOTE: partial_layer's inputs are in NHWC format. So use
                // get_nhwc_dims_size() to get the input size instead of
                // get_dims_size().
                partial_layer = create_wt_partial_layer_from_tile(
                        &curr_layer, input_tile, output_tile);
                // Only the first output tile will copy inputs data.
                partial_layer.input_req =
                        get_wt_inputs_io_req(&partial_layer,
                                             tiling.num_output_tiles,
                                             input_tile->num_hw_passes,
                                             device);
                partial_layer.weights_req =
                        (it == 0) ? get_wt_weights_io_req(&curr_layer, device)
                                  : IO_NONE;
                INFO_MSG("weights req: %d, input req: %d\n",
                         partial_layer.weights_req,
                         partial_layer.input_req);
                bool use_pf_delay = input_tile->num_hw_passes == 1;
                if (!input_tile->execute) {
                    if (device->use_sw_prefetch &&
                        curr_layer.input_req == IO_ACP) {
                        wt_try_prefetch_input_activations(
                                output_tile,
                                &_activations[0][0][0][0],
                                &nhwc_activations_dims,
                                &nhwc_weights_dims,
                                img,
                                it,
                                input_row_start,
                                halo_rows,
                                use_pf_delay);
                    }
                    run_wt_sampled_input_tile(
                            &curr_layer,
                            output_tile,
                            input_tile,
                            tiling.num_output_tiles,
                            img,
                            output_tile_kern_start,
                            result_row_start,
                            device,
                            &_activations[img][input_row_start][0][0],
                            host_results->data[0].dense_hp->d);
                    input_row_start += (partial_layer.inputs.rows - halo_rows);
                    result_row_start += partial_layer.outputs.rows;
                    continue;
                }

                begin_profiling(
                        "standard_convolution_layer_smv_input_tile", lnum);
                if (input_tile->sampling_upscale_factor > 1) {
                    set_profiling_type_sampled(
                            1, input_tile->sampling_upscale_factor);
                }

                // Set up input location and mappings.
                float16* activations_loc =
                        &_activations[img][input_row_start][0][0];
                if (partial_layer.input_req != IO_NONE) {
                    MAP_ARRAY_TO_ACCEL(
                            g_smv->kConvolutionHw,
                            get_host_inputs_var_name(partial_layer.input_req),
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
                                get_dims_size(&nhwc_weights_dims) *
                                sizeof(float16));

                // Sampling operates on iterations of the following loop over
                // num_hw_iters. We always execute the first and last
                // iterations, because those iterations are responsible for
                // handling data movement. Between those two iterations, we only
                // run up to sampled_inner_iter iterations. The remaining
                // iterations have their outputs set to zero.

                for (int iter = 0; iter < input_tile->num_hw_passes; iter++) {
                    // Prefetch weights for the next output tile.
                    //
                    // We can do this either only on the second hardware
                    // pass, or on the first hardware pass if there is only
                    // HW pass. It's generally better to prefetch on the second
                    // HW pass so we don't cause contention for L2
                    // bandwidth when the accelerator is loading data too.
                    bool do_prefetch =
                            device->use_sw_prefetch &&
                            (partial_layer.input_req == IO_ACP) &&
                            (it != output_tile->num_input_tiles - 1) &&
                            ((iter == 1) || (input_tile->num_hw_passes == 1));
                    if (do_prefetch) {
                        wt_try_prefetch_input_activations(
                                output_tile,
                                &_activations[0][0][0][0],
                                &nhwc_activations_dims,
                                &nhwc_weights_dims,
                                img,
                                it,
                                input_row_start,
                                halo_rows,
                                use_pf_delay);
                    }
                    smv_convolution_options* options =
                            &input_tile->hw_passes[iter];
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
                        run_sampled_hw_pass(
                                &partial_layer, options, img, temp_result->d);
                        continue;
                    }

                    // Map the result array to the accelerator. We have to
                    // do this per hw iteration because the sampled passes
                    // will overwrite the mapping for this array.
                    MAP_ARRAY_TO_ACCEL(
                            g_smv->kConvolutionHw,
                            get_host_results_var_name(curr_layer.output_req),
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
                    INFO_MSG("Layer %d: output tile %d, "
                             "input_tile %d, hw_pass %d: input_req = %d, "
                             "output_req = %d, weights_req = %d\n",
                             partial_layer.num, ot, it, iter,
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

                end_profiling();  // standard_convolution_layer_smv_input_tile
                input_row_start += (partial_layer.inputs.rows - halo_rows);
                result_row_start += partial_layer.outputs.rows;
            }
            free_fp16array(temp_result);
            end_profiling(); // standard_convolution_layer_smv_output_tile
            INFO_MSG("Finished an output tile.\n");
            output_tile_kern_start += output_tile->num_ofmaps;
        }
    }
    free_data_list(nhwc_activations);
    free_conv_wt_tiling_cfg(&tiling);
}
