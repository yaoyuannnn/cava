#include <stdlib.h>
#include <assert.h>
#include "common/defs.h"
#include "common/utility.h"

void *malloc_aligned(size_t size) {
  void *ptr = NULL;
  int err = posix_memalign((void **)&ptr, CACHELINE_SIZE, size);
  assert(err == 0 && "Failed to allocate memory!");
  return ptr;
}

// Swap the pointers stored in ptr1 and ptr2.
void swap_pointers(void** ptr1, void** ptr2) {
    void* temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;
}
