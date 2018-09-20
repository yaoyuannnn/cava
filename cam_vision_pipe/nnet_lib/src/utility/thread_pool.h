#ifndef _ARCH_THREAD_POOL_H_
#define _ARCH_THREAD_POOL_H_

#include <stdbool.h>

typedef enum _thread_status {
    UNINITIALIZED,
    IDLE,
    RUNNING
} thread_status;

typedef void *(*thread_worker_func)(void*);

// A worker thread descriptor.
//
// This descriptor informs the worker thread whether there is work to be done
// and what work to be done, and provides information to the outside world on
// its current status.
typedef struct _thread_work_t {
    thread_worker_func func;
    void* args;

    // The gem5 ID of the CPU running this worker thread. This should not be
    // changed after the thread is initialized.
    int cpuid;
    // This mutex protects all of the subsequent fields of this struct. Any
    // modification of these fields must first acquire status_mutex.
    pthread_mutex_t status_mutex;
    // Set to true to inform the worker thread to terminate.
    bool exit;
    // Set to true if the thread_worker_func func and args are valid and need
    // to be executed.
    bool valid;
    // IDLE or RUNNING.
    thread_status status;
    // The main thread signals this condition variable to wake up the thread
    // and have it check for work (indicated by valid = true).
    pthread_cond_t wakeup_cond;
    // The worker thread signals this condition variable to inform the main
    // thread of a change in status (usually from RUNNING -> IDLE).
    pthread_cond_t status_cond;
} thread_work_t;

// Represents a pool of threads to which work can be dispatched.
typedef struct _thread_pool_t {
    // pthread handles.
    pthread_t* threads;
    // Worker descriptors.
    thread_work_t* work;
    // Total number of threads.
    int num_threads;
} thread_pool_t;

extern thread_pool_t thread_pool;

int thread_dispatch(thread_worker_func func, void* args);

// Initialize the thread pool with nthreads. If this is called twice in a row,
// it will trigger an assertion failure.
void init_thread_pool(int nthreads);

// Shutdown the thread pool and free all resources.
void destroy_thread_pool();

// Wait for all threads in the pool to return to idle state.
void thread_pool_join();

#endif
