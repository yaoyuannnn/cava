#!/usr/bin/env bash

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
top_level=`git rev-parse --show-toplevel`
bin_path=${top_level}/build/cam-vision-gem5-accel

${gem5_dir}/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --env=env.txt \
  --num-cpus=1 \
  --mem-size=4GB \
  --mem-type=LPDDR4_3200_2x16  \
  --sys-clock=1.25GHz \
  --cpu-clock=2.5GHz \
  --cpu-type=DerivO3CPU \
  --ruby \
  --access-backing-store \
  --l2_size=2097152 \
  --enable_prefetchers \
  --cacheline_size=64 \
  --accel_cfg_file=gem5.cfg \
  -c ${bin_path} -o "raw_32x32.bin result.bin test.conf" \
   > stdout 2> stderr
