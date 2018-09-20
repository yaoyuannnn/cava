#include <argp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utility/utility.h"
#include "kernels/pipe_stages.h"
#include "camera_pipe.h"

typedef enum _argnum {
    RAW_IMAGE_BIN,
    OUTPUT_IMAGE_BIN,
    NUM_REQUIRED_ARGS,
    NUM_ARGS = NUM_REQUIRED_ARGS,
} argnum;

typedef struct _arguments {
  char *args[NUM_ARGS];
} arguments;

static char prog_doc[] = "\nCamera vision pipeline on gem5-Aladdin.\n";
static char args_doc[] =
    "path/to/raw-image-binary path/to/output-image-binary";
static struct argp_option options[] = {
  { 0 },
};

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
  arguments *args = (arguments*)(state->input);
  switch (key) {
    case ARGP_KEY_ARG: {
      if (state->arg_num >= NUM_REQUIRED_ARGS)
        argp_usage(state);
      args->args[state->arg_num] = arg;
      break;
    }
    case ARGP_KEY_END: {
        if (state->arg_num < NUM_REQUIRED_ARGS) {
          printf("Not enough arguments! Got %d, require %d.\n", state->arg_num,
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

static struct argp parser = {options, parse_opt, args_doc, prog_doc};

int main(int argc, char* argv[]) {
  // Parse the arguments.
  arguments args;
  argp_parse(&parser, argc, argv, 0, 0, &args);

  uint8_t *host_input = NULL;
  uint8_t *host_input_nwc = NULL;
  uint8_t *host_result = NULL;
  uint8_t *host_result_nwc = NULL;
  int row_size, col_size;

  // Read a raw image.
  printf("Reading a raw image from %s\n", args.args[RAW_IMAGE_BIN]);
  host_input_nwc =
      read_image_from_binary(args.args[RAW_IMAGE_BIN], &row_size, &col_size);
  printf("Raw image shape: %d x %d x %d\n", row_size, col_size,
         CHAN_SIZE);
  // The input image is stored in HWC format. To make it more efficient for
  // future optimization (e.g., vectorization), I expect we would transform it
  // to CHW format at some point.
  convert_hwc_to_chw(host_input_nwc, row_size, col_size, &host_input);

  // Allocate a buffer for storing the output image data.
  host_result = malloc_aligned(sizeof(uint8_t) * row_size * col_size * CHAN_SIZE);

  // Invoke the camera pipeline
  camera_pipe(host_input, host_result, row_size, col_size);

  // Transform the output image back to HWC format.
  convert_chw_to_hwc(host_result, row_size, col_size, &host_result_nwc);

  // Output the image
  printf("Writing output image to %s\n", args.args[OUTPUT_IMAGE_BIN]);
  write_image_to_binary(args.args[OUTPUT_IMAGE_BIN], host_result_nwc, row_size,
                        col_size);

  free(host_input);
  free(host_input_nwc);
  free(host_result);
  free(host_result_nwc);
  return 0;
}

