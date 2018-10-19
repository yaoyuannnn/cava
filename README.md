# CAVA: Camera Vision Pipeline on gem5-Aladdin
==================================================================

CAVA is a library for building and simulating camera vision pipelines, written
to work with gem5-Aladdin. It consists of two parts: an ISP and a DNN framework
(SMAUG).

In SMAUG, several reference implementations are provided, along with a
model of an actual SoC containing multiple DNN accelerators.

## Getting started ##

### Clone CAVA repository

These example commands happen to use HTTPS, not SSH, but either is fine.

  ```bash
  git clone https://github.com/yaoyuannnn/cava.git
  ```

### Install gem5-aladdin

In the same directory that you cloned CAVA, clone the gem5-aladdin repository.

  ```bash
  # recursively clone aladdin and xenon dependencies
  git clone --recursive https://github.com/harvard-acc/gem5-aladdin.git
  ```

After the aladdin repository has been recursively cloned into the
`gem5-aladdin/src` subdirectory, set your `$ALADDIN_HOME` environment variable
to the path `gem5-aladdin/src/aladdin` within gem5-aladdin. This environment
variable determines the paths in the build files, so you will see some errors
when building if you forget to set it.

### Install libconfuse

CAVA depends on (`libconfuse`)[https://github.com/martinh/libconfuse] for
reading its configuration files. For example, you can install it on Ubuntu
with:

  ```bash
  apt-get install libconfuse-dev
  ```

### Build and Run

First, make sure you are using `gcc` by setting your `$CC` environment variable
to `gcc`, which is used in the build files. (If you use Clang, you will likely
see a bunch of unrecognized warning flags and run into issues with unrecognized
`.func` and `.endfunc` directives which are used in `-gstabs` debugging.)

To build and run the default camera vision pipeline:

  ```bash
  make native
  cd sim
  sh run_native.sh
  ```

## CAVA frontend — an ISP model ##
An Image Signal Processor (ISP) converts the raw pixels produced by camera sensors to useful images. The current ISP kernel is modeled after the [Nikon-D7000 camera](https://en.wikipedia.org/wiki/Nikon_D7000). It contains a five-stage camera pipeline. Namely, demosaicing, denoising, color space conversion / white balancing, gamut mapping and tone mapping. The purpose and implementation of every pipeline stage is discussed as follows.

### Demosaicing ###

### Denosing ###

### Color space transform/ White balancing ###

### Gamut mapping ###

### Tone mapping ###

## CAVA backend — a computer vision framework: SMAUG ##


## A walk through CAVA ##
The input for CAVA is a raw image. 
