# CAVA: Camera Vision Pipeline on gem5-Aladdin
==================================================================

CAVA is a library for building and simulating camera vision pipelines, written
to work with gem5-Aladdin. It consists of two parts: an ISP and a DNN framework
(SMAUG).

In SMAUG, several reference implementations are provided, along with a
model of an actual SoC containing multiple DNN accelerators.

# Overview #

## Getting started ##

To build and run a camera vision pipeline:

  ```bash
  make native
  cd sim
  sh run_native.sh
  ```
