# CAVA: Camera Vision Pipeline on gem5-Aladdin
==================================================================

CAVA is a library for building and simulating camera vision pipelines, written
to work with gem5-Aladdin. It consists of two parts: an ISP and a DNN framework
(SMAUG).

In SMAUG, several reference implementations are provided, along with a
model of an actual SoC containing multiple DNN accelerators.

## Getting started ##

To build and run a camera vision pipeline:

  ```bash
  make native
  cd sim
  sh run_native.sh
  ```

## CAVA frontend — an ISP model ##
An Image Signal Processor (ISP) converts the raw pixels produced by camera sensors to useful images. The current ISP kernel is modeled after the Nikon-D7000 camera. It contains a five-stage camera pipeline. Namely, demosaicing, denoising, color space conversion/ white balancing, gamut mapping and tone mapping. The purpose and implementation of every pipeline stage is discussed as follows.

### Demosaicing ###

### Denosing ###

### Color space transform/ White balancing ###

### Gamut mapping ###

### Tone mapping ###

## CAVA backend — a computer vision framework: SMAUG ##


## A walk through CAVA ##
The input for CAVA is a raw image. 
