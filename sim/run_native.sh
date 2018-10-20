#!/usr/bin/env bash

# Runs camera/vision pipeline on a target image file.
# Converts input.png to result.png.

input_image=input.png

top_level=`git rev-parse --show-toplevel`
bin_path=${top_level}/build/cam-vision-native
load_and_convert=${top_level}/scripts/load_and_convert.py 

output_image_name=result
raw_input_image=${input_image}
binary_input_image=input.bin
binary_output_image=${output_image_name}.bin

# Convert raw image to binary.
${load_and_convert} -r ${raw_input_image}

# Run camera pipeline on binary, then DNN using specified configuration.
${bin_path} ${binary_input_image} ${binary_output_image} test.conf

# From binary back to raw, for user display.
${load_and_convert} -b ${binary_output_image}
