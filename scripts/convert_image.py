#!/usr/bin/env python

import numpy as np
import argparse
import imageio
import struct
import math
import random

def convert_raw_to_binary(raw_name):
    im = imageio.imread(raw_name)
    print(im.shape)

    # Write the raw image to a binary file
    bin_name = raw_name.replace('png', 'bin')
    with open(bin_name, 'w') as bin_file:
      bin_file = open(bin_name, 'w')
      bin_file.write(struct.pack('i', im.shape[0]))
      bin_file.write(struct.pack('i', im.shape[1]))
      bin_file.write(struct.pack('i', im.shape[2]))
      bin_file.write(im)

def convert_binary_to_image(bin_name):
    with open(bin_name, "r") as bin_file:
      shape = struct.unpack("iii", bin_file.read(12))
      im = struct.unpack("B" * np.prod(shape), bin_file.read())
      im = np.asarray(im, dtype=np.uint8).reshape(shape)

    # Write to the image file
    im_name = bin_name.replace('bin', 'png')
    imageio.imwrite(im_name, im)

#------------------------------------------------------------
# Functions for reverting an image to a raw image
#------------------------------------------------------------
def scale(input_im, output_im):
  for row in xrange(input_im.shape[0]):
    for col in xrange(input_im.shape[1]):
      for chan in xrange(3):
        output_im[row][col][chan] = float(input_im[row][col][chan]) / 255

def descale(input_im, output_im):
  for row in xrange(input_im.shape[0]):
    for col in xrange(input_im.shape[1]):
      for chan in xrange(3):
        output_im[row][col][chan] = input_im[row][col][chan] * 255

def remosaic(input_im, output_im):
  for y in xrange(input_im.shape[0]):
    for x in xrange(input_im.shape[1]):
      # If an even row
      if y % 2 == 0:
        # If an even column
        if x % 2 == 0:
          # Green pixel, remove blue and red
          output_im[y][x][0] = 0
          output_im[y][x][2] = 0
          # Also divide the green by half to account
          # for interpolation reversal
          output_im[y][x][1] = input_im[y][x][1] / 2
        # If an odd column
        else:
          # Red pixel, remove blue and green
          output_im[y][x][0] = 0
          output_im[y][x][1] = 0
      # If an odd row
      else:
        # If an even column
        if x % 2 == 0:
          # Blue pixel, remove red and green
          output_im[y][x][2] = 0
          output_im[y][x][1] = 0
        # If an odd column
        else:
          # Green pixel, remove blue and red
          output_im[y][x][0] = 0
          output_im[y][x][2] = 0
          # Also divide the green by half to account
          # for interpolation reversal
          output_im[y][x][1] = input_im[y][x][1] / 2


def renoise(input_im, output_im):
  temp_im = np.ndarray(shape=input_im.shape, dtype=np.uint8)
  descale(input_im, temp_im)
  row,col,ch= input_im.shape
  mean = 0
  var = 0.1
  sigma = var**0.5
  gauss = np.random.normal(mean,sigma,(row,col,ch))
  gauss = gauss.reshape(row,col,ch)
  temp_im = temp_im + gauss
  scale(temp_im, output_im)

def renoise1(input_im, output_im):
  red_a   =  0.1460
  red_b   =  7.6876
  green_a =  0.1352
  green_b =  5.0834
  blue_a  =  0.1709
  blue_b  = 12.3381

  for y in xrange(input_im.shape[0]):
    for x in xrange(input_im.shape[1]):
      # Compute the channel noise standard deviation
      red_std = math.sqrt(red_a * input_im[y][x][2] + red_b)
      green_std = math.sqrt(green_a * input_im[y][x][1] + green_b)
      blue_std = math.sqrt(blue_a * input_im[y][x][0] + blue_b)

      # Blue channel
      print red_std, green_std, blue_std
      output_im[y][x][0] = input_im[y][x][0] + np.random.normal(0, blue_std)
      ## Green channel
      output_im[y][x][1] = input_im[y][x][1] + np.random.normal(0, green_std)
      ## Red channel
      output_im[y][x][2] = input_im[y][x][2] + np.random.normal(0, red_std)
      print output_im[y][x][0] - input_im[y][x][0],
      print output_im[y][x][1] - input_im[y][x][1],
      print output_im[y][x][2] - input_im[y][x][2]
      exit(0)

def reverse_color_transform(input_im, wb_index, output_im):
  tr = np.ndarray(shape=(3, 3), dtype=float)
  with open("../cam_vision_pipe/cam_models/NikonD7000/jpg2raw_transform.txt", "r") as tr_file:
    wb_base = 5 + 5*(wb_index-1)
    tr_data = tr_file.read().splitlines(True)[wb_base:wb_base+3]
    for i,line in enumerate(tr_data):
      tr[i] = line.split()
  tr_inv = np.linalg.inv(tr)

  for row in xrange(input_im.shape[0]):
    for col in xrange(input_im.shape[1]):
      for chan in xrange(3):
        output_im[row][col][chan] = \
            max(input_im[row][col][0] * tr_inv[chan][0] + \
                input_im[row][col][1] * tr_inv[chan][1] + \
                input_im[row][col][2] * tr_inv[chan][2], \
                0);

def reverse_gamut_map(input_im, num_cps, output_im):
  gm_cp = np.ndarray(shape=(num_cps, 3), dtype=float)
  gm_weight = np.ndarray(shape=(num_cps, 3), dtype=float)
  gm_coef = np.ndarray(shape=(4, 3), dtype=float)
  with open("../cam_vision_pipe/cam_models/NikonD7000/jpg2raw_ctrlPoints.txt", "r") as gm_cp_file:
    gm_cp_data = gm_cp_file.read().splitlines(True)[1:]
    for i,line in enumerate(gm_cp_data):
      gm_cp[i] = line.split()

  with open("../cam_vision_pipe/cam_models/NikonD7000/jpg2raw_coefs.txt", "r") as gm_coef_file:
    gm_coef_data = gm_coef_file.read().splitlines(True)[1:]
    for i, line in enumerate(gm_coef_data):
      if i < num_cps:
        gm_weight[i] = line.split()
      else:
        gm_coef[i-num_cps] = line.split()

  l2_dist = np.zeros(num_cps)
  for row in xrange(input_im.shape[0]):
    print "row =",
    print row
    for col in xrange(input_im.shape[1]):
      for cp in xrange(num_cps):
        l2_dist[cp] = \
            math.sqrt((input_im[row][col][0] - gm_cp[cp][0]) * \
                     (input_im[row][col][0] - gm_cp[cp][0]) + \
                 (input_im[row][col][1] - gm_cp[cp][1]) * \
                     (input_im[row][col][1] - gm_cp[cp][1]) + \
                 (input_im[row][col][2] - gm_cp[cp][2]) * \
                     (input_im[row][col][2] - gm_cp[cp][2]));

      for chan in xrange(3):
        output_im[row][col][chan] = 0.0;
        for cp in xrange(num_cps):
          output_im[row][col][chan] += \
              l2_dist[cp] * gm_weight[cp][chan];
        output_im[row][col][chan] += gm_coef[0][chan] + \
                                   gm_coef[1][chan] * input_im[row][col][0] + \
                                   gm_coef[2][chan] * input_im[row][col][1] + \
                                   gm_coef[3][chan] * input_im[row][col][2];


def reverse_tone_map(input_im, output_im):
  tm_resp_func = np.ndarray(shape=(256,3), dtype=float)
  with open("../cam_vision_pipe/cam_models/NikonD7000/jpg2raw_respFcns.txt", "r") as tm_file:
    tm_data = tm_file.read().splitlines(True)[1:]
    for i,line in enumerate(tm_data):
      tm_resp_func[i] = line.split()

  for chan in xrange(3):
    for row in xrange(input_im.shape[0]):
      for col in xrange(input_im.shape[1]):
        x = int(input_im[row][col][chan] * 255)
        output_im[row][col][chan] = tm_resp_func[x][chan]

def convert_image_to_raw(image_name):
  wb_index = 6
  num_cps = 3702
  orig_im = imageio.imread(image_name)
  print("Input image shape:"), orig_im.shape
  result_im = np.ndarray(shape=orig_im.shape, dtype=np.uint8)

  # These two arrays will used as ping-poing buffers for
  # input/output of every kernel.
  input_im = np.ndarray(shape=orig_im.shape, dtype=float)
  output_im = np.ndarray(shape=orig_im.shape, dtype=float)

  scale(orig_im, input_im)
  reverse_tone_map(input_im, output_im)
  reverse_gamut_map(output_im, num_cps, input_im)
  reverse_color_transform(input_im, wb_index, output_im)
  renoise(output_im, input_im)
  remosaic(input_im, output_im)
  descale(output_im, result_im)

  # Write to the image file
  output_name = "raw_" + image_name
  imageio.imwrite(output_name, result_im)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", "-r",
      help="Convert a raw image to a binary file")
  parser.add_argument("--binary", "-b",
      help="Convert a binary file to a image")
  parser.add_argument("--image", "-i",
      help="Convert an image file to a raw image")
  args = parser.parse_args()

  if args.raw != None:
    convert_raw_to_binary(args.raw)
  elif args.binary != None:
    convert_binary_to_image(args.binary)
  elif args.image != None:
    convert_image_to_raw(args.image)

if __name__ == "__main__":
  main()
