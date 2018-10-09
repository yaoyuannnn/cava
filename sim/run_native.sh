#!/usr/bin/env bash

top_level=`git rev-parse --show-toplevel`
bin_path=${top_level}/build/cam-vision-native

${bin_path} raw_32x32.bin result.bin test.conf
