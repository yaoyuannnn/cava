#include <argp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "common/utility.h"

#include "cam_pipe/utility/cam_pipe_utility.h"
#include "cam_pipe/kernels/pipe_stages.h"
#include "cam_pipe/cam_pipe.h"

#include "nnet_lib/utility/compression.h"
#include "nnet_lib/utility/data_archive.h"
#include "nnet_lib/utility/data_archive_bin.h"
#include "nnet_lib/utility/init_data.h"
#include "nnet_lib/utility/profiling.h"
#include "nnet_lib/utility/read_model_conf.h"
#include "nnet_lib/utility/utility.h"
#include "nnet_lib/nnet_fwd.h"
#include "nnet_lib/arch/interface.h"

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
} arguments;

static char prog_doc[] = "\nCamera vision pipeline on gem5-Aladdin.\n";
static char args_doc[] = "path/to/raw-image-binary path/to/output-image-binary "
                         "path/to/model-config-file";
static struct argp_option options[] = {
    { "num-inputs", 'n', "N", 0, "Number of input images" }, { 0 },
    { "data-file", 'f', "F", 0,
      "File to read data and weights from (if data-init-mode == READ_FILE or "
      "save-params is true). *.txt files are decoded as text files, while "
      "*.bin files are decoded as binary files." },
};

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
    arguments* args = (arguments*)(state->input);
    switch (key) {
        case 'n': {
            args->num_inputs = strtol(arg, NULL, 10);
            break;
        }
        case 'f': {
            args->args[DATA_FILE] = arg;
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
                printf("Not enough arguments! Got %d, require %d.\n",
                       state->arg_num,
                       NUM_REQUIRED_ARGS);
                argp_usage(state);
            }
            break;
        }
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
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
        printf("Input layer shape: %d x %d x %d. Does not match the image from "
               "the camera pipeline! Image shape: %d x %d x %d\n",
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
    model_file = read_all_from_file(args.args[DATA_FILE],
                                    &network,
                                    &global_weights->data[0].dense,
                                    &inputs->data[0].dense,
                                    &labels,
                                    &compress_type);

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

