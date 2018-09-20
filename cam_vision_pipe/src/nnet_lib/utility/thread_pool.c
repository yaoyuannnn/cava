#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/m5ops.h"
#include "utility/thread_pool.h"

thread_pool_t thread_pool;

// This struct is only used within this file to initialize the worker threads.
// The rest of the user interface uses thread_work_t.
typedef struct _thread_init_args {
    thread_work_t* work;
    pthread_mutex_t cpuid_mutex;
    pthread_cond_t cpuid_cond;
    int cpuid;
} thread_init_args;

typedef struct _work_queue_entry {
    thread_worker_func func;
    void* args;
    struct _work_queue_entry* next;
} work_queue_entry;

typedef struct _work_queue {
    work_queue_entry *head;
    work_queue_entry *tail;
    pthread_mutex_t mutex;
} work_queue_t;

work_queue_t work_queue;

void push_work_queue(work_queue_t* queue, work_queue_entry* entry) {
    if (queue->head == NULL) {
        queue->head = entry;
        queue->tail = entry;
    } else {
        queue->tail->next = entry;
        queue->tail = entry;
    }
}

work_queue_entry* pop_work_queue(work_queue_t* queue) {
    if (!queue->head)
        return NULL;
    work_queue_entry* head = queue->head;
    queue->head = queue->head->next;
    if (!queue->head)
        queue->tail = NULL;
    return head;
}

work_queue_entry* peek_work_queue(work_queue_t* queue) {
    return queue->head;
}

static int try_dispatch_to_thread(thread_worker_func func,
                                  void* args,
                                  thread_work_t* worker) {
    pthread_mutex_lock(&worker->status_mutex);
    if (worker->status == IDLE && !worker->valid) {
        worker->func = func;
        worker->args = args;
        worker->valid = true;
        M5_WAKE_CPU(worker->cpuid);
        pthread_cond_signal(&worker->wakeup_cond);
        pthread_mutex_unlock(&worker->status_mutex);
        return worker->cpuid;
    }
    pthread_mutex_unlock(&worker->status_mutex);
    return -1;
}

// Dispatches the specified work to the thread pool.
//
// If the thread pool is all busy, then this queues it on the work queue. The
// work queue lock MUST be held by the caller before calling this function.
static int thread_dispatch_to_pool(thread_worker_func func, void* args) {
    for (int i = 0; i < thread_pool.num_threads; ++i) {
        int ret = try_dispatch_to_thread(func, args, &thread_pool.work[i]);
        if (ret != -1)
            return ret;
    }
    work_queue_entry* entry =
            (work_queue_entry*)malloc(sizeof(work_queue_entry));
    entry->func = func;
    entry->args = args;
    entry->next = NULL;
    push_work_queue(&work_queue, entry);
    return -1;
}

// User-facing API to dispatch work to the thread pool.
int thread_dispatch(thread_worker_func func, void* args) {
    pthread_mutex_lock(&work_queue.mutex);
    int ret = thread_dispatch_to_pool(func, args);
    pthread_mutex_unlock(&work_queue.mutex);
    return ret;
}

// Checks the work queue for more work and dispatches it if possible.
//
// By default, we'll first try to dispatch it to the current worker thread
// (since this function is only called when it is done). If this thread has
// already been assigned more work, however, we'll then try to dispatch it any
// thread in the pool. Should that also fail, we'll leave it on the work
// queue.
void thread_finish_check_for_work(thread_work_t* worker) {
    pthread_mutex_lock(&work_queue.mutex);
    work_queue_entry* next_work = peek_work_queue(&work_queue);
    if (next_work) {
        int cpuid = try_dispatch_to_thread(
                next_work->func, next_work->args, worker);
        if (cpuid == -1)
            cpuid = thread_dispatch_to_pool(next_work->func, next_work->args);
        if (cpuid != -1)
            pop_work_queue(&work_queue);
    }
    pthread_mutex_unlock(&work_queue.mutex);
}

static void* thread_spinloop(void* args) {
    thread_init_args* init_args = (thread_init_args*)args;
    thread_work_t* work = init_args->work;
    // Notify the main thread about this thread's cpuid. This can only be done
    // after the thread context is created.
    pthread_mutex_lock(&init_args->cpuid_mutex);
    init_args->cpuid = M5_GET_CPUID();
    work->status = IDLE;
    pthread_cond_signal(&init_args->cpuid_cond);
    pthread_mutex_unlock(&init_args->cpuid_mutex);

    do {
        if (!work->valid && !work->exit) {
            M5_QUIESCE();
        }
        pthread_mutex_lock(&work->status_mutex);
        while (!work->valid && !work->exit)
            pthread_cond_wait(&work->wakeup_cond, &work->status_mutex);
        if (work->valid) {
            work->status = RUNNING;
            pthread_mutex_unlock(&work->status_mutex);

            // Run the function.
            work->func(work->args);

            pthread_mutex_lock(&work->status_mutex);
            work->status = IDLE;
            work->valid = false;
            pthread_cond_signal(&work->status_cond);
        }
        bool exit_thread = work->exit;
        pthread_mutex_unlock(&work->status_mutex);
        if (exit_thread)
            break;
        // Check the queue if we have any more work to do. We cannot be holding
        // any locks when this function is called.
        thread_finish_check_for_work(work);
    } while (true);

    pthread_exit(NULL);
}

void init_thread_pool(int nthreads) {
    if (nthreads <= 0)
        return;
    thread_pool.threads = (pthread_t*)malloc(sizeof(pthread_t) * nthreads);
    thread_pool.work =
            (thread_work_t*)malloc(sizeof(thread_work_t) * nthreads);
    thread_pool.num_threads = nthreads;
    work_queue.head = NULL;
    work_queue.tail = NULL;

    thread_init_args* init_args =
            (thread_init_args*)malloc(sizeof(thread_init_args) * nthreads);
    // Initialize the work descriptors for each thread and start the threads.
    for (int i = 0; i < nthreads; i++) {
        thread_pool.work[i].func = NULL;
        thread_pool.work[i].args = NULL;
        thread_pool.work[i].exit = false;
        thread_pool.work[i].valid = false;
        thread_pool.work[i].status = UNINITIALIZED;
        pthread_mutex_init(&thread_pool.work[i].status_mutex, NULL);
        pthread_cond_init(&thread_pool.work[i].wakeup_cond, NULL);
        pthread_cond_init(&thread_pool.work[i].status_cond, NULL);

        init_args[i].work = &thread_pool.work[i];
        pthread_mutex_init(&init_args[i].cpuid_mutex, NULL);
        pthread_cond_init(&init_args[i].cpuid_cond, NULL);
        init_args[i].cpuid = -1;
        pthread_create(
                &thread_pool.threads[i], NULL, &thread_spinloop, &init_args[i]);

        // Fill in the cpuid of the worker thread.
        pthread_mutex_lock(&init_args[i].cpuid_mutex);
        while (init_args[i].cpuid == -1 ||
               thread_pool.work[i].status == UNINITIALIZED)
            pthread_cond_wait(&init_args[i].cpuid_cond, &init_args[i].cpuid_mutex);
        thread_pool.work[i].cpuid = init_args[i].cpuid;
        pthread_mutex_unlock(&init_args[i].cpuid_mutex);
        assert(thread_pool.work[i].status != UNINITIALIZED &&
               "Worker thread did not succcessfully initialize!");
    }
    free(init_args);
}

void thread_pool_join() {
    // There is no need to call M5_WAKE_CPU() here. If the CPU is quiesced, then
    // it cannot possibly be running anything, so its status will be IDLE, and
    // this will move on to the next CPU.
    for (int i = 0; i < thread_pool.num_threads; ++i) {
        thread_work_t* worker = &thread_pool.work[i];
        pthread_mutex_lock(&worker->status_mutex);
        while (worker->status == RUNNING || worker->valid == true) {
            pthread_cond_wait(&worker->status_cond, &worker->status_mutex);
        }
        pthread_mutex_unlock(&worker->status_mutex);
    }
}

void destroy_thread_pool() {
    if (thread_pool.num_threads <= 0)
        return;
    for (int i = 0; i < thread_pool.num_threads; ++i) {
        M5_WAKE_CPU(thread_pool.work[i].cpuid);
        pthread_mutex_lock(&thread_pool.work[i].status_mutex);
        thread_pool.work[i].exit = true;
        pthread_cond_signal(&thread_pool.work[i].wakeup_cond);
        pthread_mutex_unlock(&thread_pool.work[i].status_mutex);
    }
    for (int i = 0; i < thread_pool.num_threads; ++i) {
        pthread_join(thread_pool.threads[i], NULL);
        pthread_mutex_destroy(&thread_pool.work[i].status_mutex);
        pthread_cond_destroy(&thread_pool.work[i].wakeup_cond);
        pthread_cond_destroy(&thread_pool.work[i].status_cond);
    }
    free(thread_pool.threads);
    free(thread_pool.work);
    thread_pool.threads = NULL;
    thread_pool.work = NULL;
    thread_pool.num_threads = 0;
}
