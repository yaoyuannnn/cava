.PHONY: all help native debug dma-trace-binary \
        gem5-cpu gem5-accel run \
        clean-trace clean-gem5 clean-native clean gem5

native:
	@$(MAKE) -f common/Makefile.native native

debug:
	@$(MAKE) -f common/Makefile.native debug

run:
	@$(MAKE) -f common/Makefile.native run

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

gem5: gem5-cpu gem5-accel

clean: clean-trace clean-gem5 clean-native

