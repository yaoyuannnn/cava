#ifndef _COMMON_UTILITY_H_
#define _COMMON_UTILITY_H_

#include <stddef.h>

void *malloc_aligned(size_t size);

void swap_pointers(void** ptr1, void** ptr2);
#define SWAP_PTRS(a_ptr, b_ptr)                                                \
    swap_pointers((void**)&(a_ptr), (void**)&(b_ptr))

#endif
