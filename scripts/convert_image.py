#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
from scipy.stats import rankdata
import argparse
import imageio
import struct
import math
import random
import cmodule
from PIL import Image

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
  np.copyto(output_im, np.divide(input_im, 255.0))

def descale(input_im, output_im):
  np.copyto(output_im, np.multiply(input_im, 255).astype(np.uint8))

def remosaic(input_im, output_im):
  print "Remosaicing."
  for y in xrange(input_im.shape[0]):
    for x in xrange(input_im.shape[1]):
      # If an even row
      if y % 2 == 0:
        # If an even column
        if x % 2 == 0:
          # Green pixel, remove blue and red
          output_im[y][x][2] = 0
          output_im[y][x][0] = 0
          # Also divide the green by half to account
          # for interpolation reversal
          output_im[y][x][1] = input_im[y][x][1] / 2
        # If an odd column
        else:
          # Red pixel, remove blue and green
          output_im[y][x][2] = 0
          output_im[y][x][1] = 0
          output_im[y][x][0] = input_im[y][x][0]
      # If an odd row
      else:
        # If an even column
        if x % 2 == 0:
          # Blue pixel, remove red and green
          output_im[y][x][0] = 0
          output_im[y][x][1] = 0
          output_im[y][x][2] = input_im[y][x][2]
        # If an odd column
        else:
          # Green pixel, remove blue and red
          output_im[y][x][2] = 0
          output_im[y][x][0] = 0
          # Also divide the green by half to account
          # for interpolation reversal
          output_im[y][x][1] = input_im[y][x][1] / 2

def denoise(image_name):
  print "Denoising."
  origin_im = imageio.imread(image_name)
  print("Input image shape:"), origin_im.shape
  result_im = np.ndarray(shape=origin_im.shape, dtype=np.uint8)

  input_im = np.ndarray(shape=origin_im.shape, dtype=np.float32)
  output_im = np.ndarray(shape=origin_im.shape, dtype=np.float32)
  scale(origin_im, input_im)

  row_size,col_size,ch_size= input_im.shape
  for row in xrange(row_size):
    for col in xrange(col_size):
      for ch in xrange(ch_size):
        if row > 0 and row < row_size - 1 and \
           col > 0 and col < col_size - 1:
          window = np.array([input_im[row-1][col-1][ch], input_im[row-1][col][ch], \
                             input_im[row-1][col+1][ch], input_im[row][col-1][ch], \
                             input_im[row][col][ch], input_im[row][col+1][ch],\
                             input_im[row+1][col-1][ch], input_im[row+1][col][ch], \
                             input_im[row+1][col+1][ch]])
          sort = np.sort(window)
          output_im[row][col][ch] = sort[4]
        else:
          output_im[row][col][ch] = input_im[row][col][ch]

  descale(output_im, result_im)

  # Write to the image file
  output_name = "denoised_" + image_name
  imageio.imwrite(output_name, result_im)

def renoise(input_im, output_im):
  print "Renoising."
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

def reverse_color_transform(input_im, wb_index, output_im):
  print "Reverse color mapping."
  tr = np.ndarray(shape=(3, 3), dtype=np.float32)
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

def gamut_map(input_im, num_cps, output_im, reverse=True):
  if reverse:
    print "Reverse",
  print "gamut mapping."
  gm_cp = np.ndarray(shape=(num_cps, 3), dtype=np.float32)
  gm_weight = np.ndarray(shape=(num_cps, 3), dtype=np.float32)
  gm_coef = np.ndarray(shape=(4, 3), dtype=np.float32)

  if reverse:
    gm_cp_fname = "../cam_vision_pipe/cam_models/NikonD7000/jpg2raw_ctrlPoints.txt"
    gm_coef_fname = "../cam_vision_pipe/cam_models/NikonD7000/jpg2raw_coefs.txt"
  else:
    gm_cp_fname = "../cam_vision_pipe/cam_models/NikonD7000/raw2jpg_ctrlPoints.txt"
    gm_coef_fname = "../cam_vision_pipe/cam_models/NikonD7000/raw2jpg_coefs.txt"

  with open(gm_cp_fname, "r") as gm_cp_file:
    gm_cp_data = gm_cp_file.read().splitlines(True)[1:]
    for i,line in enumerate(gm_cp_data):
      gm_cp[i] = line.split()

  with open(gm_coef_fname, "r") as gm_coef_file:
    gm_coef_data = gm_coef_file.read().splitlines(True)[1:]
    for i, line in enumerate(gm_coef_data):
      if i < num_cps:
        gm_weight[i] = line.split()
      else:
        gm_coef[i-num_cps] = line.split()

  # Use the external C++ implementation for better performance.
  cmodule.gamut_map(input_im, output_im, gm_cp, gm_weight, gm_coef)

def reverse_tone_map(input_im, output_im):
  print "Reverse tone mapping."
  tm_resp_func = np.ndarray(shape=(256,3), dtype=np.float32)
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
  input_im = np.ndarray(shape=orig_im.shape, dtype=np.float32)
  output_im = np.ndarray(shape=orig_im.shape, dtype=np.float32)

  scale(orig_im, input_im)
  reverse_tone_map(input_im, output_im)
  gamut_map(output_im, num_cps, input_im, True)
  reverse_color_transform(input_im, wb_index, output_im)
  #renoise(output_im, input_im)
  remosaic(output_im, input_im)
  descale(input_im, result_im)

  # Write to the image file
  output_name = "raw_" + image_name
  imageio.imwrite(output_name, result_im)

def convert_image_to_grayscale(image_name):
  img = Image.open(image_name).convert('LA')
  output_name = "grayscale_" + image_name
  img.save(output_name)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", "-r",
      help="Convert a raw image to a binary file.")
  parser.add_argument("--binary", "-b",
      help="Convert a binary file to a image.")
  parser.add_argument("--backward", "-B",
      help="Convert an image file to a raw image.")
  parser.add_argument("--denoise", "-d",
      help="Apply denoising to the image.")
  parser.add_argument("--grayscale", "-g",
      help="Convert an RGB image into grayscale.")
  args = parser.parse_args()

  if args.raw != None:
    convert_raw_to_binary(args.raw)
  elif args.binary != None:
    convert_binary_to_image(args.binary)
  elif args.backward != None:
    convert_image_to_raw(args.backward)
  elif args.grayscale != None:
    convert_image_to_grayscale(args.grayscale)
  elif args.denoise != None:
    denoise(args.denoise)

if __name__ == "__main__":
  main()
