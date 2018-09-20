#ifndef _CORE_M5_OPS_H_
#define _CORE_M5_OPS_H_

#ifdef GEM5_HARNESS
#include "gem5/m5ops.h"
#define M5_SWITCH_CPU() m5_switch_cpu()
#define M5_RESET_STATS() m5_reset_stats(0, 0)
#define M5_DUMP_STATS() m5_dump_stats(0, 0)
#define M5_DUMP_RESET_STATS() m5_dump_reset_stats(0, 0)
#define M5_QUIESCE() m5_quiesce()
#define M5_WAKE_CPU(id) m5_wake_cpu(id)
#define M5_EXIT(ns) m5_exit(ns)
#define M5_GET_CPUID() m5_get_cpuid()
#else
#define M5_SWITCH_CPU()
#define M5_RESET_STATS()
#define M5_DUMP_STATS()
#define M5_DUMP_RESET_STATS()
#define M5_QUIESCE()
#define M5_WAKE_CPU(id) (-1)
#define M5_EXIT(ns)
#define M5_GET_CPUID() (0)
#endif

#endif
