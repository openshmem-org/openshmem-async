/*
 *
 * Copyright (c) 2011 - 2016
 *   University of Houston System and UT-Battelle, LLC.
 * Copyright (c) 2009 - 2016
 *   Silicon Graphics International Corp.  SHMEM is copyrighted
 *   by Silicon Graphics International Corp. (SGI) The OpenSHMEM API
 *   (shmem) is released by Open Source Software Solutions, Inc., under an
 *   agreement with Silicon Graphics International Corp. (SGI).
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * o Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * o Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * o Neither the name of the University of Houston System,
 *   UT-Battelle, LLC. nor the names of its contributors may be used to
 *   endorse or promote products derived from this software without specific
 *   prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


/*
 * These are compatibility routines for older SGI architectures.  They
 * are now defined in OpenSHMEM to do nothing.
 */

#include "utils.h"
#include "trace.h"

/*
 * Compatibility no-op cache routines
 */

#ifdef HAVE_FEATURE_PSHMEM
#include "pshmem.h"
#else
/*
 * TODO: We using this include just to get the definition of hclib specific structs
 * declared inside shmem.h (e.g. shmem_promise_t, etc.). If there is a better way,
 * then this can be avoided.
 */
#include "shmem.h" 
#endif /* HAVE_FEATURE_PSHMEM */

#include "api.h"

#ifdef HAVE_FEATURE_HCLIB

void shmem_task_scope_begin() {
  hclib_start_finish(); 
}

void shmem_task_scope_end() {
  hclib_end_finish();
}

void shmem_task_nbi (void (*body)(void *), void *user_data, shmem_future_t **optional_future)
{
  hclib_async(body, user_data, optional_future, NULL, NULL, 0);
}

void shmem_parallel_for_nbi(void (*body)(void *), void *user_data, shmem_future_t **optional_future, 
                              int lowBound, int highBound, int stride, int tile_size, 
                              int loop_dimension, int loop_mode) {
  
  loop_domain_t info = { lowBound, highBound, stride, tile_size};
  hclib_forasync(body, user_data, optional_future, loop_dimension, &info, loop_mode);
}

int shmem_n_workers() {
  return hclib_num_workers();
}

int shmem_my_worker() {
  return get_current_worker();
}

#ifndef HCLIB_COMM_WORKER_FIXED
// this would go away once hclib fibre support is fixed for communication worker
void temporary_wrapper(void* entrypoint) {
  /*
   * In HC-OpenSHMEM, there is no start_finish equivalent call.
   * The end_finish is called everytime user will call shmem_fence/shmem_barrier etc.
   * Once the end_finish (implicitely) is called from HC-OpenSHMEM,
   * a new start_finish scope is started to pair with
   * the hclib_end_finish implicitely called at the end of user_main.
   */
  hclib_start_finish();
  asyncFct_t funcPtr = (asyncFct_t) entrypoint;
  funcPtr(NULL);
  hclib_end_finish();
}

void shmem_workers_init(void* entrypoint, void * arg) {
  assert(arg==NULL && "temporarily we are not allowing passing argument to the entrypoint function");
  hclib_launch(temporary_wrapper, entrypoint);
}
#else 
void shmem_workers_init() {
  hclib_init();
  /*
   * In HC-OpenSHMEM, there is no start_finish equivalent call. 
   * The end_finish is called everytime user will call shmem_fence/shmem_barrier etc.
   * Once the end_finish (implicitely) is called from HC-OpenSHMEM, 
   * a new start_finish scope is started to pair with
   * the hclib_end_finish implicitely called at the end of user_main.
   */
  hclib_start_finish();
}

void shmem_workers_finalize() {
  hclib_end_finish();
  hclib_finalize();
}
#endif

void shmem_hclib_end_finish() {
  hclib_end_finish();
  /*
   * In HC-OpenSHMEM, there is no start_finish equivalent call. 
   * The end_finish is called everytime user will call shmem_fence/shmem_barrier etc.
   * Once the end_finish (implicitely) is called from HC-OpenSHMEM, 
   * a new start_finish scope is started to pair with
   * the hclib_end_finish implicitely called at the end of user_main.
   */
  hclib_start_finish();
}

shmem_promise_t *shmem_malloc_promise() {
  return ((shmem_promise_t *) hclib_promise_create());
}

shmem_promise_t **shmem_malloc_promises(int npromises) {
  return ((shmem_promise_t **) hclib_promise_create_n(npromises, 1)); //TODO: Do we really have to force null-terminated ?
}

void shmem_satisfy_promise(shmem_promise_t *promise, void* datum) {
  hclib_promise_put((hclib_promise_t *) promise, datum);
}

void* shmem_future_wait(shmem_future_t *future) {
  return hclib_future_wait((hclib_future_t*) future);
}

#else // !HAVE_FEATURE_HCLIB --> unsupported case

typedef void (*asyncFct_t)(void * arg);
void shmem_parallel_for_nbi(void (*body)(void *), void *user_data, shmem_future_t **optional_future, 
                              int lowBound, int highBound, int stride, int tile_size, 
                              int loop_dimension, int loop_mode) {
}
void shmem_task_nbi (void (*body)(void *), void *user_data, shmem_future_t **optional_future) { 
  body(user_data);
}
int shmem_n_workers() { return -1; }
int shmem_my_worker() { return -1; }
void shmem_workers_init(int *argc, char **argv, void* entrypoint, void * arg) { 
  asyncFct_t funcPtr = (asyncFct_t) entrypoint;
  funcPtr(arg);
}
void shmem_hclib_end_finish(){}
shmem_promise_t *shmem_malloc_promise() { }
shmem_promise_t **shmem_malloc_promises(int npromises) { }
void shmem_satisfy_promise(shmem_promise_t *promise, void* datum) { }
void* shmem_future_wait(shmem_future_t *future) { return NULL; }

#endif
