#!/usr/bin/env bash

# Runs camera/vision pipeline on a target raw image.
# Requires a RAW image as input (not a PNG/JPG/etc.).


top_level=`git rev-parse --show-toplevel`
bin_path=${top_level}/build/cam-vision-native
load_and_convert=${top_level}/scripts/load_and_convert.py 

# We are using a preprocessed raw input image.
#raw_input_image=N/A
output_image_name=result
binary_input_image=raw_32x32.bin
binary_output_image=${output_image_name}.bin

# Convert raw image to binary.
#${load_and_convert} -r ${raw_input_image}

# Run camera pipeline on binary, then DNN using specified configuration.
${bin_path} ${binary_input_image} ${binary_output_image} test.conf

# From binary back to raw, for user display.
${load_and_convert} -b ${binary_output_image}
