.PHONY: all help native dma-trace-binary \
                          gem5-cpu gem5-accel \
                                clean-trace clean-gem5 clean-native clean gem5

native:
	@$(MAKE) -f common/Makefile.common --no-print-directory native

dma-trace-binary:
	@$(MAKE) -f common/Makefile.common --no-print-directory dma-trace-binary

gem5-cpu:
	@$(MAKE) -f common/Makefile.common --no-print-directory gem5-cpu

gem5-accel:
	@$(MAKE) -f common/Makefile.common --no-print-directory gem5-accel

clean-gem5:
	@$(MAKE) -f common/Makefile.common --no-print-directory clean-gem5

clean-native:
	@$(MAKE) -f common/Makefile.common --no-print-directory clean-native

gem5: gem5-cpu gem5-accel

clean: clean-trace clean-gem5 clean-native

