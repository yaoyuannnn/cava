.PHONY: all help native dma-trace-binary \
                          gem5-cpu gem5-accel \
                             run clean-trace clean-gem5 clean-native clean gem5

native:
	@$(MAKE) -f common/Makefile.common native

debug:
	@$(MAKE) -f common/Makefile.common debug

dma-trace-binary:
	@$(MAKE) -f common/Makefile.tracer dma-trace-binary

gem5-cpu:
	@$(MAKE) -f common/Makefile.gem5 gem5-cpu

gem5-accel:
	@$(MAKE) -f common/Makefile.gem5 gem5-accel

clean-gem5:
	@$(MAKE) -f common/Makefile.gem5 clean-gem5

clean-native:
	@$(MAKE) -f common/Makefile.common clean-native

clean-trace:
	@$(MAKE) -f common/Makefile.tracer clean-trace

run:
	./build/camera-pipe-native raw.bin result.bin
	./scripts/load_and_convert.py --binary result.bin

gem5: gem5-cpu gem5-accel

clean: clean-trace clean-gem5 clean-native

