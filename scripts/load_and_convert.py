#!/usr/bin/env python

import numpy as np
import argparse
import imageio
import struct

def convert_raw_to_binary(raw_name):
    im = imageio.imread(raw_name)
    print im.shape
    for i in xrange(im.shape[0]):
      for j in xrange(im.shape[1]):
          print '(',
          print im[i][j][0], im[i][j][1], im[i][j][2],
          print ')',
      print
    exit(0)

    # Write the raw image to a binary file
    bin_name = raw_name.split('.')[:-1]+'.bin'
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

    #for i in xrange(im.shape[0]):
    #  for j in xrange(im.shape[1]):
    #    for k in xrange(im.shape[2]):
    #      print im[i][j][k],
    #  print
    #exit(0)

    # Write to the image file
    im_name = bin_name.replace('bin', 'png')
    imageio.imwrite(im_name, im)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", "-r",
      help="Convert raw image to a binary file")
  parser.add_argument("--binary", "-b",
      help="Convert binary file to a image")
  args = parser.parse_args()

  if args.raw != None:
    convert_raw_to_binary(args.raw)
  elif args.binary != None:
    convert_binary_to_image(args.binary)

if __name__ == "__main__":
  main()
