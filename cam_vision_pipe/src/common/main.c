#include <argp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "common/utility.h"

#include "cam_pipe/utility/cam_pipe_utility.h"
#include "cam_pipe/kernels/pipe_stages.h"
#include "cam_pipe/cam_pipe.h"

#include "arch/common.h"
#include "arch/interface.h"
#include "core/ref/lookup_tables.h"
#include "nnet_lib/utility/compression.h"
#include "nnet_lib/utility/data_archive.h"
#include "nnet_lib/utility/data_archive_bin.h"
#include "nnet_lib/utility/init_data.h"
#include "nnet_lib/utility/profiling.h"
#include "nnet_lib/utility/read_model_conf.h"
#include "nnet_lib/utility/utility.h"

int NUM_TEST_CASES;
int NUM_CLASSES;
int INPUT_DIM;
int NUM_WORKER_THREADS;
float* sigmoid_table;
float* exp_table;
sigmoid_impl_t SIGMOID_IMPL;

typedef enum _argnum {
    RAW_IMAGE_BIN,
    OUTPUT_IMAGE_BIN,
    NETWORK_CONFIG,
    NUM_REQUIRED_ARGS,
    DATA_FILE = NUM_REQUIRED_ARGS,
    NUM_ARGS,
} argnum;

typedef struct _arguments {
    char* args[NUM_ARGS];
    int num_inputs;
    int num_threads;
    data_init_mode data_mode;
    sigmoid_impl_t sigmoid_impl;
} arguments;

static char prog_doc[] = "\nCamera vision pipeline on gem5-Aladdin.\n";
static char args_doc[] = "path/to/raw-image-binary path/to/output-image-binary "
                         "path/to/model-config-file";
static struct argp_option options[] = {
    { "num-inputs", 'n', "N", 0, "Number of input images" },
    { "data-init-mode", 'd', "D", 0,
      "Data and weights generation mode (FIXED, FAST_FIXED, RANDOM, "
      "READ_FILE)." },
    { "data-file", 'f', "F", 0,
      "File to read data and weights from (if data-init-mode == READ_FILE or "
      "save-params is true). *.txt files are decoded as text files, while "
      "*.bin files are decoded as binary files." },
        { "sigmoid-impl", 'm', "IMPL", 0,
      "Sigmoid implementation: exp-unit (default), centered-lut, or "
      "noncentered-lut." },
    { "num-threads", 't', "THREADS", 0,
      "Number of worker threads in the thread pool." },
    { 0 },
};

// Convert a string to a data initialization mode.
//
// If the string was a valid choice, this updates @mode and returns 0;
// otherwise, returns 1.
int str2mode(char* str, data_init_mode* mode) {
    if (strncmp(str, "RANDOM", 7) == 0) {
        *mode = RANDOM;
        return 0;
    } else if (strncmp(str, "FIXED", 6) == 0) {
        *mode = FIXED;
        return 0;
    } else if (strncmp(str, "FAST_FIXED", 11) == 0) {
        *mode = FAST_FIXED;
        return 0;
    } else if (strncmp(str, "READ_FILE", 10) == 0) {
        *mode = READ_FILE;
        return 0;
    }
    return 1;
}

// Convert a string to a sigmoid implementation mode.
//
// If the string was a valid choice, this updates @impl and returns 0;
// otherwise, returns 1.
int str2sigmoidimpl(char* str, sigmoid_impl_t* impl) {
    if (strncmp(str, "exp-unit", 9) == 0) {
        *impl = ExpUnit;
        return 0;
    } else if (strncmp(str, "centered-lut", 13) == 0) {
        *impl = CenteredLUT;
        return 0;
    } else if (strncmp(str, "noncentered-lut", 16) == 0) {
        *impl = NoncenteredLUT;
        return 0;
    }
    return 1;
}

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
    arguments* args = (arguments*)(state->input);
    switch (key) {
        case 'n': {
            args->num_inputs = strtol(arg, NULL, 10);
            break;
        }
        case 'd': {
            if (str2mode(arg, &args->data_mode))
                argp_usage(state);
            break;
        }
        case 'm': {
            if (str2sigmoidimpl(arg, &args->sigmoid_impl))
                argp_usage(state);
            break;
        }
        case 'f': {
            args->args[DATA_FILE] = arg;
            break;
        }
        case 't': {
            args->num_threads = strtol(arg, NULL, 10);
            break;
        }
        case ARGP_KEY_ARG: {
            if (state->arg_num >= NUM_REQUIRED_ARGS)
                argp_usage(state);
            args->args[state->arg_num] = arg;
            break;
        }
        case ARGP_KEY_END: {
            if (state->arg_num < NUM_REQUIRED_ARGS) {
                fprintf(stderr,
                        "Not enough arguments! Got %d, require %d.\n",
                        state->arg_num,
                        NUM_REQUIRED_ARGS);
                argp_usage(state);
            }
            break;
        }
        case ARGP_KEY_FINI: {
            if (args->data_mode == READ_FILE && !args->args[DATA_FILE]) {
                fprintf(stderr,
                        "[ERROR]: You must specify a data file to read "
                        "parameters "
                        "from.\n");
                argp_usage(state);
            }
            break;
        }
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

void set_default_args(arguments* args) {
    args->num_inputs = 1;
    args->num_threads = 0;
    args->data_mode = RANDOM;
    args->sigmoid_impl = ExpUnit;
    for (int i = 0; i < NUM_ARGS; i++) {
        args->args[i] = NULL;
    }
}

data_list* pack_compress_colmajor_weights(float* weights,
                                          dims_t* orig_dims,
                                          dims_t* bias_dims) {
    // Compress the weights without the bias row first.
    // Swap the rows and columns.
    dims_t transposed_dims = *orig_dims;
    int rows = transposed_dims.rows;
    transposed_dims.rows = transposed_dims.cols;
    transposed_dims.cols = rows;
    transposed_dims.align_pad =
            calc_padding(transposed_dims.rows, DATA_ALIGNMENT);
    csr_array_t* weights_csr = compress_dense_data_csr(weights, &transposed_dims);
    packed_csr_array_t* packed_weights_csr =
            pack_csr_array_vec8_f16(weights_csr, &transposed_dims);

    // Just store biases as an uncompressed buffer.
    float* bias_loc = weights + get_dims_size(&transposed_dims);
    farray_t* biases_storage = init_farray(bias_dims->cols, false);
    memcpy(biases_storage->d, bias_loc, biases_storage->size * sizeof(float));

    data_list* list = init_data_list(2);
    list->data[0].packed = packed_weights_csr;
    list->data[1].dense = biases_storage;
    list->type[0] = PackedCSR;
    list->type[1] = Uncompressed;
    free_csr_array_t(weights_csr);

    return list;
}

// If any of the layers in the network can use compressed weights storage, then
// compress their (currently) dense weights and update the layer's weight
// storage type accordingly.
void process_compressed_weights(network_t* network,
                                farray_t* weights,
                                iarray_t* compress_mask) {
    for (int i = 1; i < network->depth; i++) {
        layer_t* layer = &network->layers[i];
        assert(compress_mask->d[i] < NumDataStorageTypes &&
               "Invalid value of compress type found!");
        data_storage_t storage_type = (data_storage_t)compress_mask->d[i];
        float* weights_loc =
                (weights->d + get_weights_loc_for_layer(network->layers, i));
        if (storage_type == Uncompressed) {
            layer->host_weights = init_data_list(1);
            // Don't do a memcpy - this gets extremely expensive in simulation
            // with large models. Instead, use the pointer directly and mark
            // the array as un-freeable.
            farray_t* layer_weights = init_farray(0, false);
            layer_weights->d = weights_loc;
            layer_weights->size = get_num_weights_layer(layer, 0);
            layer->host_weights->data[0].dense = layer_weights;
            layer->host_weights->type[0] = Uncompressed;
        } else if (storage_type == CSR) {
            dims_t dims_with_bias = layer->weights;
            dims_with_bias.rows += layer->biases.rows;
            csr_array_t* csr =
                    compress_dense_data_csr(weights_loc, &dims_with_bias);
            layer->host_weights = init_data_list(1);
            layer->host_weights->data[0].csr = csr;
            layer->host_weights->type[0] = CSR;
        } else if (storage_type == PackedCSR) {
#if TRANSPOSE_WEIGHTS == 1
            layer->host_weights = pack_compress_colmajor_weights(
                    weights_loc, &layer->weights, &layer->biases);
#else
            dims_t dims_with_bias = layer->weights;
            dims_with_bias.rows += layer->biases.rows;
            csr_array_t* csr =
                    compress_dense_data_csr(weights_loc, &dims_with_bias);
            packed_csr_array_t* packed_csr =
                    pack_csr_array_vec8_f16(csr, &dims_with_bias);
            layer->host_weights = init_data_list(1);
            layer->host_weights->data[0].packed = packed_csr;
            layer->host_weights->type[0] = PackedCSR;
            free_csr_array_t(csr);
#endif
        }
    }
}

// Free weights used in the network.
void free_network_weights(network_t* network) {
    for (int i = 0; i < network->depth; i++) {
        layer_t* layer = &network->layers[i];
        if (!layer->host_weights)
            continue;
        free_data_list(layer->host_weights);
    }
}

static struct argp parser = { options, parse_opt, args_doc, prog_doc };

int main(int argc, char* argv[]) {
    // Parse the arguments.
    arguments args;
    set_default_args(&args);
    argp_parse(&parser, argc, argv, 0, 0, &args);

    //////////////////////////////////////////////////////////////////////////
    //
    // Invoke the camera pipeline!
    //
    // Invoke the front end camera pipeline. We use raw images to model the
    // inputs from the camera sensor.
    //////////////////////////////////////////////////////////////////////////

    uint8_t* host_input = NULL;
    uint8_t* host_input_nwc = NULL;
    uint8_t* host_result = NULL;
    uint8_t* host_result_nwc = NULL;
    int row_size, col_size;

    // Read a raw image.
    printf("Reading a raw image from %s\n", args.args[RAW_IMAGE_BIN]);
    host_input_nwc = read_image_from_binary(
            args.args[RAW_IMAGE_BIN], &row_size, &col_size);
    printf("Raw image shape: %d x %d x %d\n", row_size, col_size, CHAN_SIZE);
    // The input image is stored in HWC format. To make it more efficient for
    // future optimization (e.g., vectorization), I expect we would transform it
    // to CHW format at some point.
    convert_hwc_to_chw(host_input_nwc, row_size, col_size, &host_input);

    // Allocate a buffer for storing the output image data.
    host_result =
            malloc_aligned(sizeof(uint8_t) * row_size * col_size * CHAN_SIZE);

    // Invoke the camera pipeline
    cam_pipe(host_input, host_result, row_size, col_size);

    // Transform the output image back to HWC format.
    convert_chw_to_hwc(host_result, row_size, col_size, &host_result_nwc);

    // Output the image
    printf("Writing output image to %s\n", args.args[OUTPUT_IMAGE_BIN]);
    write_image_to_binary(
            args.args[OUTPUT_IMAGE_BIN], host_result_nwc, row_size, col_size);

    //////////////////////////////////////////////////////////////////////////
    //
    // Invoke SMAUG!
    //
    // The output images from the camera pipeline is fed to SMAUG, completing
    // the "vision" part of the camera vision pipeline.
    //////////////////////////////////////////////////////////////////////////

    NUM_TEST_CASES = args.num_inputs;
    NUM_WORKER_THREADS = args.num_threads;
    SIGMOID_IMPL = args.sigmoid_impl;

    network_t network;
    device_t* device;
    sampling_param_t* sampling_param;
    network.depth = configure_network_from_file(args.args[NETWORK_CONFIG],
                                                &network.layers,
                                                &device,
                                                &sampling_param);

    // Sanity check on the dimensionality of the input layer. It should match
    // the image generated from the camera pipeline.
    if (row_size != network.layers[0].inputs.rows ||
        col_size != network.layers[0].inputs.cols ||
        CHAN_SIZE != network.layers[0].inputs.height) {
        fprintf(stderr,
                "Input layer shape: %d x %d x %d. Does not match the image "
                "from the camera pipeline! Image shape: %d x %d x %d\n",
                network.layers[0].inputs.rows,
                network.layers[0].inputs.cols,
                network.layers[0].inputs.height,
                row_size,
                col_size,
                CHAN_SIZE);
        exit(1);
    }

    // Initialize weights, data, and labels.
    // This is just a container for the global set of weights in the entire
    // network, which we need for now in case we want to populate it with data
    // from an archive.
    data_list* inputs = init_data_list(1);
    data_list* outputs = init_data_list(1);
    layer_t input_layer = network.layers[0];
    data_list* global_weights = init_data_list(1);
    int total_weight_size =
            get_total_num_weights(network.layers, network.depth);
    printf("  Total weights: %d elements\n", total_weight_size);
    iarray_t labels = { NULL, 0 };
    labels.size = NUM_TEST_CASES;
    labels.d = (int*)malloc_aligned(labels.size * sizeof(float));
    memset(labels.d, 0, labels.size * sizeof(float));

    // This stores a binary mask for each layer, specifying whether its weights
    // can be compressed or not.
    iarray_t compress_type = { NULL, (size_t)network.depth };
    compress_type.d = (int*)malloc_aligned(compress_type.size * sizeof(int));
    memset(compress_type.d, 0, compress_type.size * sizeof(int));

    mmapped_file model_file = init_mmapped_file();
    if (args.data_mode == READ_FILE) {
        // Read all except input from file, since the camera pipeline will
        // produce the input.
        model_file = read_all_except_input_from_file(
                args.args[DATA_FILE], &network, &global_weights->data[0].dense,
                &labels, &compress_type);
        inputs->data[0].dense = init_farray(
                NUM_TEST_CASES * get_dims_size(&input_layer.inputs), true);
        init_data_from_image(inputs->data[0].dense->d,
                             &network,
                             NUM_TEST_CASES,
                             host_result);
    } else {
        global_weights->data[0].dense = init_farray(total_weight_size, false);
        inputs->data[0].dense = init_farray(
                NUM_TEST_CASES * get_dims_size(&input_layer.inputs), true);
        init_weights(global_weights->data[0].dense->d, network.layers,
                     network.depth, args.data_mode, TRANSPOSE_WEIGHTS);
        //init_data(inputs->data[0].dense->d, &network, NUM_TEST_CASES,
        //          args.data_mode);
        init_data_from_image(inputs->data[0].dense->d,
                             &network,
                             NUM_TEST_CASES,
                             host_result);
        init_labels(labels.d, NUM_TEST_CASES, args.data_mode);
    }
    inputs->type[0] = Uncompressed;
    global_weights->type[0] = Uncompressed;

    init_sigmoid_table(&sigmoid_table);
    init_exp_table(&exp_table);
    process_compressed_weights(
            &network, global_weights->data[0].dense, &compress_type);
    fflush(stdout);

    // Run a forward pass through the neural net
    printf("Running forward pass\n");
    init_profiling_log();
    nnet_fwd(inputs, global_weights, outputs, &network, device, sampling_param);
    dump_profiling_log();
    close_profiling_log();

    // Compute the classification error rate
    float* result = network.layers[network.depth - 1].result_in_temp
                            ? outputs->data[0].dense->d
                            : inputs->data[0].dense->d;

    float error_fraction =
            compute_errors(result, labels.d, NUM_TEST_CASES, NUM_CLASSES);
    write_output_labels("output_labels.out",
                        result,
                        NUM_TEST_CASES,
                        NUM_CLASSES,
                        network.layers[network.depth - 1].outputs.align_pad);

    printf("Fraction incorrect (over %d cases) = %f\n", NUM_TEST_CASES,
           error_fraction);

    // Free up the allocated memories.
    if (sigmoid_table)
        free(sigmoid_table);
    if (exp_table)
        free(exp_table);
    if (model_file.addr != NULL) {
        close_bin_data_file(&model_file);
    }
    free_data_list(inputs);
    free_data_list(global_weights);
    free_network_weights(&network);
    free_data_list(outputs);
    free(labels.d);
    free(compress_type.d);
    free(network.layers);
    free(device);
    free(sampling_param);

    free(host_input);
    free(host_input_nwc);
    free(host_result);
    free(host_result_nwc);

    M5_EXIT(0);

    return 0;
}

