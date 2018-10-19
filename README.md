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
An *Image Signal Processor (ISP)* converts the raw pixels produced by camera
sensors to useful images. 

The default ISP kernel is modeled after the [Nikon-D7000
camera](https://en.wikipedia.org/wiki/Nikon_D7000). It contains a five-stage
camera pipeline: 

1. Demosaicing
2. Denoising
3. Color space conversion / white balancing
4. Gamut mapping
5. Tone mapping

The purpose and implementation of every pipeline stage is discussed as follows.

See `cam_vision_pipe/src/cam_pipe/kernels/pipe_stages.c` for implementation
details.

### Demosaicing ###

Filters using a _color filter array_ (CFA) over each photosite of a sensor to
interpolate local undersampled colors into a true color at the pixel. A common
CFA is known as the _Bayer filter_, which contains more green than red and blue
pixels due to the imbalance in human perception. The filter operation yields a
"mosaic" of RGB pixels with intensities.

Also known as _debayering_, _CFA interpolation_, or _color reconstruction_.

### Denoising ###

There are many algorithms for denoising, which aims to reduce the level of
noise in the image. The default ISP kernel implements a local nonlinear
interpolation.

### Color Space Transform / White Balancing ###

To perform color balancing, we multiply the RGB color value at each point with
a 3x3 diagonal matrix whose values are configurable.

### Gamut Mapping ###

A _gamut_ is the set of colors which fully represents some scenario, whether an
image, a color space, or the capability of a particular output device. 

For example, preparing an image for printing requires gamut mapping. This is
because the image is often specified in RGB, whereas the printer expects the
CMYK color space. Gamut mapping performs this transformation from RGB to CMYK
so that the image is most faithfully realized in print.

The gamut mapping stage here computes the L2-norm to a set of control points,
weights them, and adds bias for a radial basis function (RBF).

### Tone Mapping ###

Tone mapping approximates images with a higher dynamic range than the output
device. The process must preserve colors and other aspects of the original
image while seeking to squeeze the presumably stronger contrast of the original
image into the feasible range of the output device.

For example, an HDR image may be the result of capturing multiple exposures
which together approximate the luminance of the original scene. The tone
mapping operator then squeezes this into the lower dynamic range of an output
device such as a monitor. Although such approximations may produce unusual
artifacts, they preserve image features and often retain a pleasant balance
between global contrast and local contrast.

There are a variety of availability _tone mapping operators_ (TMOs), which may
be either global or local.

This is sometimes called _color reproduction_ or _color processing_.

## CAVA backend — a computer vision framework: SMAUG ##

SMAUG is a framework for deep neural networks (DNNs), together with reference
implementations that include a model of an SoC with multiple DNN accelerators.

## A walk through CAVA ##
The input for CAVA is a raw image. 
