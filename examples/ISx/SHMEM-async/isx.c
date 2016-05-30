/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
    * Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/
#include <shmem.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <unistd.h> // sleep()
#include <sys/stat.h>
#include <stdint.h>
#include "params.h"
#include "isx.h"
#include "timer.h"
#include "pcg_basic.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#define ROOT_PE 0

// Needed for shmem collective operations
int pWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
double dWrk[_SHMEM_REDUCE_SYNC_SIZE];
long long int llWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long pSync[_SHMEM_REDUCE_SYNC_SIZE];

uint64_t NUM_PES; // Number of parallel workers
uint64_t TOTAL_KEYS; // Total number of keys across all PEs
uint64_t NUM_KEYS_PER_PE; // Number of keys generated on each PE
uint64_t NUM_BUCKETS; // The number of buckets in the bucket sort
uint64_t BUCKET_WIDTH; // The size of each bucket
uint64_t MAX_KEY_VAL; // The maximum possible generated key value
char* log_file;

/*
 * This variable sets the maximum number of workers allowed
 * to participate in computation per pe. In actual
 * when the computation is run, there can be 1-n number
 * of workers (n<=WORKERS_PER_PE).
 *
 * In AsyncSHMEM model, these workers are the hclib workers.
 */
int WORKERS_PER_PE=1;
#if defined (_SHMEM_WORKERS) || defined (_OPENMP)
#define MAX_CHUNKS_ALLOWED 512
#define GET_VIRTUAL_RANK(rank, wid) ((rank * WORKERS_PER_PE) + (wid))
#define GET_REAL_RANK(vrank) ((int)(vrank / WORKERS_PER_PE))
#define PARALLEL_FOR_MODE SHMEM_PARALLEL_FOR_RECURSIVE_MODE
//#define PARALLEL_FOR_MODE SHMEM_PARALLEL_FOR_FLAT_MODE
#else
#define MAX_CHUNKS_ALLOWED 1
#define GET_VIRTUAL_RANK(rank, wid) (rank)
#define GET_REAL_RANK(vrank) (vrank)
#endif
uint64_t NUM_KEYS_PER_WORKERS;
int actual_num_workers;
// This is done due to current limitation that entrypoint function
// cannot accept arguments. This will be resolved in future version of 
// AsyncSHMEM
int m_argc;
char** m_argv;

volatile int whose_turn;

long long int* receive_offset;
long long int* my_bucket_size;

/*
 * In case of WORKERS_PER_PE>1, setting KEY_BUFFER_SIZE_PER_PE=(1uLL<<28uLL) 
 * would cause compilation error (in gasnet) with the declaration of:
 * "KEY_TYPE my_bucket_keys[WORKERS_PER_PE][KEY_BUFFER_SIZE_PER_PE];"
 * I am not sure why that is causing the problem. But decreasing
 * the KEY_BUFFER_SIZE_PER_PE as this removes that compilation bug.
 *
 * As an alternative, I left KEY_BUFFER_SIZE_PER_PE=(1uLL<<28uLL)
 * and used shmem_malloc to allocate:
 * "KEY_TYPE **my_bucket_keys"
 * This resolved the compilation error but writing to this
 * array my_bucket_keys using shmem_int_put was causing 
 * SEGFAULT.
 */
#define KEY_BUFFER_SIZE_PER_PE ((int)(1uLL<<28uLL))

// The receive array for the All2All exchange
KEY_TYPE* my_bucket_keys[MAX_CHUNKS_ALLOWED];

#ifdef PERMUTE
int * permute_array;
#endif

void entrypoint(void *arg) {

  char * log_file = parse_params(m_argc, m_argv);

  init_shmem_sync_array(pSync); 

  bucket_sort();

  log_times(log_file);

  //return err;
}

int main (int argc, char ** argv) {
  shmem_init ();
  m_argc = argc;
  m_argv = argv;

#if defined(_SHMEM_WORKERS)
  shmem_workers_init(entrypoint, NULL);
#else
  entrypoint(NULL);
#endif

  shmem_finalize ();
  return 0;
}

// Parses all of the command line input and definitions in params.h
// to set all necessary runtime values and options
static char * parse_params(const int argc, char ** argv)
{
  if(argc != 3)
  {
    if( shmem_my_pe() == 0){
      printf("Usage:  \n");
      printf("  ./%s <total num keys(strong) | keys per pe(weak)> <log_file>\n",argv[0]);
    }

    shmem_finalize();
    exit(1);
  }

#if defined(_OPENMP)
  const char* chunks_env = getenv("ISX_PE_CHUNKS");
  WORKERS_PER_PE = chunks_env ? atoi(chunks_env) : 1;
#pragma omp parallel
  actual_num_workers = omp_get_num_threads();
#elif defined(_SHMEM_WORKERS)
  const char* chunks_env = getenv("ISX_PE_CHUNKS");
  WORKERS_PER_PE = chunks_env ? atoi(chunks_env) : 1;
  actual_num_workers = shmem_n_workers();
#else
  actual_num_workers = 1;
  WORKERS_PER_PE = 1;
#endif
  assert(WORKERS_PER_PE <= MAX_CHUNKS_ALLOWED);
  NUM_PES = (uint64_t) shmem_n_pes();
  MAX_KEY_VAL = DEFAULT_MAX_KEY;
  NUM_BUCKETS = NUM_PES*WORKERS_PER_PE;
  BUCKET_WIDTH = (uint64_t) ceil((double)MAX_KEY_VAL/NUM_BUCKETS);
  char * log_file = argv[2];
  char scaling_msg[64];

  switch(SCALING_OPTION){
    case STRONG:
      {
        TOTAL_KEYS = (uint64_t) atoi(argv[1]);
        NUM_KEYS_PER_WORKERS = (uint64_t) ceil((double)TOTAL_KEYS/(NUM_PES * WORKERS_PER_PE));
        NUM_KEYS_PER_PE = NUM_KEYS_PER_WORKERS * WORKERS_PER_PE;
        sprintf(scaling_msg,"STRONG");
        break;
      }

    case WEAK:
      {
        // When comparing N PEs with 0 workers to X PEs with Y workers (where, N = X*Y)
        // we need to ensure we have same number of total keys across 
        // both the implementations.
        // Hence, for case N PEs with 0 workers, TOTAL_KEYS = N * NUM_KEYS_PER_PE
        // and for case X PEs with Y workers, TOTAL_KEYS = X * Y * NUM_KEYS_PER_PE
        NUM_KEYS_PER_PE = ((uint64_t) (atoi(argv[1]))) * actual_num_workers;
        assert(NUM_KEYS_PER_PE%WORKERS_PER_PE == 0); // if this is not satisfied, change the input
        NUM_KEYS_PER_WORKERS = NUM_KEYS_PER_PE / WORKERS_PER_PE;
        sprintf(scaling_msg,"WEAK");
        break;
      }

    case WEAK_ISOBUCKET:
      {
        assert("Broken currently due to WORKERS_PER_PE" && 0);
        NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1]));
        BUCKET_WIDTH = ISO_BUCKET_WIDTH; 
        MAX_KEY_VAL = (uint64_t) (NUM_PES * BUCKET_WIDTH);
        sprintf(scaling_msg,"WEAK_ISOBUCKET");
        break;
      }

    default:
      {
        if(shmem_my_pe() == 0){
          printf("Invalid scaling option! See params.h to define the scaling option.\n");
        }

        shmem_finalize();
        exit(1);
        break;
      }
  }

  assert(MAX_KEY_VAL > 0);
  assert(NUM_KEYS_PER_PE > 0);
  assert(NUM_PES > 0);
  assert(MAX_KEY_VAL > NUM_PES);
  assert(NUM_BUCKETS > 0);
  assert(BUCKET_WIDTH > 0);
  
  if(shmem_my_pe() == 0){
    printf("ISx v%1d.%1d\n",MAJOR_VERSION_NUMBER,MINOR_VERSION_NUMBER);
#ifdef PERMUTE
    printf("Random Permute Used in ATA.\n");
#endif
    printf("  Number of Keys per PE: %" PRIu64 "\n", NUM_KEYS_PER_PE);
#if defined(_OPENMP)
    printf("  OpenMP Version, total workers: %d\n",actual_num_workers); 
    printf("  Number of Keys per Chunk: %" PRIu64 "\n", NUM_KEYS_PER_WORKERS);
    printf("  Number of Chunks per PE (ISX_PE_CHUNKS): %d\n",WORKERS_PER_PE);
#elif defined(_SHMEM_WORKERS)
    printf("  AsyncSHMEM Version, total workers: %d\n",actual_num_workers);
    printf("  Number of Keys per Chunk: %" PRIu64 "\n", NUM_KEYS_PER_WORKERS);
    printf("  Number of Chunks per PE: %d\n",WORKERS_PER_PE);
#else
    printf("  AsyncSHMEM Sequential version\n");
#endif
    printf("  Max Key Value: %" PRIu64 "\n", MAX_KEY_VAL);
    printf("  Bucket Width: %" PRIu64 "\n", BUCKET_WIDTH);
    printf("  Number of Iterations: %u\n", NUM_ITERATIONS);
    printf("  Number of PEs: %" PRIu64 "\n", NUM_PES);
    printf("  %s Scaling!\n",scaling_msg);
    }

  return log_file;
}


/*
 * The primary compute function for the bucket sort
 * Executes the sum of NUM_ITERATIONS + BURN_IN iterations, as defined in params.h
 * Only iterations after the BURN_IN iterations are timed
 * Only the final iteration calls the verification function
 */
static int bucket_sort(void)
{
  int err = 0;

  init_timers(NUM_ITERATIONS);

#ifdef PERMUTE
  create_permutation_array();
#endif

  receive_offset = (long long int*) shmem_malloc(sizeof(long long int) * WORKERS_PER_PE);
  my_bucket_size = (long long int*) shmem_malloc(sizeof(long long int) * WORKERS_PER_PE);
  for(int i=0; i<WORKERS_PER_PE; i++) my_bucket_keys[i] = (KEY_TYPE *) shmem_malloc(sizeof(KEY_TYPE) 
                                                                  * ((int)(KEY_BUFFER_SIZE_PER_PE/WORKERS_PER_PE)));
  for(uint64_t i = 0; i < (NUM_ITERATIONS + BURN_IN); ++i)
  {
    // Reset offsets used in exchange_keys
    memset(receive_offset, 0x00, WORKERS_PER_PE * sizeof(long long int));
    memset(my_bucket_size, 0x00, WORKERS_PER_PE * sizeof(long long int));

    // Reset timers after burn in 
    if(i == BURN_IN){ init_timers(NUM_ITERATIONS); } 

    shmem_barrier_all();

    timer_start(&timers[TIMER_TOTAL]);

// Time phase = 1703.453
    KEY_TYPE ** my_keys = make_input();

// Time phase = 1029.996
    int ** local_bucket_sizes = count_local_bucket_sizes(my_keys);

    int ** send_offsets;
// Time phase = 0.002
    int ** local_bucket_offsets = compute_local_bucket_offsets(local_bucket_sizes,
                                                                   &send_offsets);

// Time phase = 1699.666
    KEY_TYPE ** my_local_bucketed_keys =  bucketize_local_keys(my_keys, local_bucket_offsets);

// Time phase = 121.137
    exchange_keys(send_offsets, 
                  local_bucket_sizes,
                  my_local_bucketed_keys);

    memcpy(my_bucket_size, receive_offset, WORKERS_PER_PE*sizeof(long long int)); 
// Time phase = 2790.646
    int** my_local_key_counts = count_local_keys();

    shmem_barrier_all();

    timer_stop(&timers[TIMER_TOTAL]);

    // Only the last iteration is verified
    if(i == NUM_ITERATIONS) { 
      err = verify_results(my_local_key_counts);
    }

#if 0
    for(int wid=0; wid<WORKERS_PER_PE; wid++) {
      free(my_local_bucketed_keys[wid]);
      free(my_keys[wid]);
      free(local_bucket_sizes[wid]);
      free(local_bucket_offsets[wid]);
      free(send_offsets[wid]);
      free(my_local_key_counts[wid]);
    }
    free(my_local_bucketed_keys);
    free(my_keys);
    free(local_bucket_sizes);
    free(local_bucket_offsets);
    free(send_offsets);
    free(my_local_key_counts);
#endif

    shmem_barrier_all();
  }

  return err;
}

#if defined(_SHMEM_WORKERS)
void make_input_async(void *args, int wid) {
  KEY_TYPE ** restrict my_keys = *((KEY_TYPE *** restrict) args);

  pcg32_random_t rng = seed_my_worker(wid);
  for(uint64_t i = 0; i < NUM_KEYS_PER_WORKERS; ++i) {
    my_keys[wid][i] = pcg32_boundedrand_r(&rng, MAX_KEY_VAL);
  }
}
#endif

/*
 * Generates uniformly random keys [0, MAX_KEY_VAL] on each rank using the time and rank
 * number as a seed
 */
static KEY_TYPE ** make_input(void)
{
  timer_start(&timers[TIMER_INPUT]);

  KEY_TYPE ** restrict const my_keys = (KEY_TYPE**) shmem_malloc(WORKERS_PER_PE * sizeof(KEY_TYPE*));
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    my_keys[wid] = (KEY_TYPE*) shmem_malloc(NUM_KEYS_PER_WORKERS * sizeof(KEY_TYPE));
  }
 
  int wid; 
#if defined(_SHMEM_WORKERS)
  int lowBound = 0;
  int highBound = WORKERS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(make_input_async, (void*)(&my_keys), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(wid) schedule (dynamic,1) 
#endif
  // parallel block
  for(wid=0; wid<WORKERS_PER_PE; wid++) {
    pcg32_random_t rng = seed_my_worker(wid);
    for(uint64_t i = 0; i < NUM_KEYS_PER_WORKERS; ++i) {
      my_keys[wid][i] = pcg32_boundedrand_r(&rng, MAX_KEY_VAL);
    }
  }
#endif

  timer_stop(&timers[TIMER_INPUT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
    sprintf(msg,"V_Rank %d: Initial Keys: ", v_rank);
    for(uint64_t i = 0; i < NUM_KEYS_PER_WORKERS; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", my_keys[wid][i]);
    }
    sprintf(msg + strlen(msg),"\n");
    printf("%s",msg);
  }
  fflush(stdout);
  my_turn_complete();
#endif
  return my_keys;
}

#if defined(_SHMEM_WORKERS)
typedef struct count_local_bucket_sizes_t {
  int *** restrict local_bucket_sizes;
  KEY_TYPE ** const restrict const my_keys;
} count_local_bucket_sizes_t;

void count_local_bucket_sizes_async(void* arg, int wid) {
  count_local_bucket_sizes_t* args = (count_local_bucket_sizes_t*) arg;
  int ** restrict local_bucket_sizes = (*(args->local_bucket_sizes));
  KEY_TYPE ** const restrict const my_keys = args->my_keys;

  init_array(local_bucket_sizes[wid] , NUM_BUCKETS); // doing memset 0x00
  for(uint64_t i = 0; i < NUM_KEYS_PER_WORKERS; ++i){
    const uint32_t bucket_index = my_keys[wid][i]/BUCKET_WIDTH;
    local_bucket_sizes[wid][bucket_index]++;
  }
}
#endif

/*
 * Computes the size of each bucket by iterating all keys and incrementing
 * their corresponding bucket's size
 */
static inline int ** count_local_bucket_sizes(KEY_TYPE const ** restrict const my_keys)
{
  int ** restrict const local_bucket_sizes = (int**) shmem_malloc(WORKERS_PER_PE * sizeof(int*));
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    local_bucket_sizes[wid] = (int*) shmem_malloc(NUM_BUCKETS * sizeof(int));
  }

  timer_start(&timers[TIMER_BCOUNT]);

  int wid;
  // parallel block
#if defined(_SHMEM_WORKERS)
  count_local_bucket_sizes_t args = { &local_bucket_sizes, my_keys };
  int lowBound = 0;
  int highBound = WORKERS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(count_local_bucket_sizes_async, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(wid) schedule (dynamic,1) 
#endif
  for(wid=0; wid<WORKERS_PER_PE; wid++) {
    init_array(local_bucket_sizes[wid] , NUM_BUCKETS); // doing memset 0x00
    for(uint64_t i = 0; i < NUM_KEYS_PER_WORKERS; ++i){
      const uint32_t bucket_index = my_keys[wid][i]/BUCKET_WIDTH;
      local_bucket_sizes[wid][bucket_index]++;
    }
  }
#endif

  timer_stop(&timers[TIMER_BCOUNT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
    sprintf(msg,"V_Rank %d: local bucket sizes: ", v_rank);
    for(uint64_t i = 0; i < NUM_BUCKETS; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", local_bucket_sizes[wid][i]);
    }  
    sprintf(msg + strlen(msg),"\n");
    printf("%s",msg);   
  }

  fflush(stdout);
  my_turn_complete();
#endif

  return local_bucket_sizes;
}

#if defined(_SHMEM_WORKERS)
typedef struct compute_local_bucket_offsets_t {
  int *** restrict local_bucket_offsets;
  int const ** restrict const local_bucket_sizes;
  int *** restrict send_offsets;
} compute_local_bucket_offsets_t;

void compute_local_bucket_offsets_async(void *arg, int wid) {
  compute_local_bucket_offsets_t *args = (compute_local_bucket_offsets_t*) arg;
  int **restrict local_bucket_offsets = *(args->local_bucket_offsets);
  int const ** restrict const local_bucket_sizes = args->local_bucket_sizes;
  int *** restrict send_offsets = args->send_offsets;

  local_bucket_offsets[wid][0] = 0;
  (*send_offsets)[wid][0] = 0;
  int temp = 0;
  for(uint64_t i = 1; i < NUM_BUCKETS; i++){
    temp = local_bucket_offsets[wid][i-1] + local_bucket_sizes[wid][i-1];
    local_bucket_offsets[wid][i] = temp;
    (*send_offsets)[wid][i] = temp;
  } 
}
#endif

/*
 * Computes the prefix scan of the bucket sizes to determine the starting locations
 * of each bucket in the local bucketed array
 * Stores a copy of the bucket offsets for use in exchanging keys because the
 * original bucket_offsets array is modified in the bucketize function
 */
static inline int ** compute_local_bucket_offsets(int const ** restrict const local_bucket_sizes,
                                                 int *** restrict send_offsets)
{
  int ** restrict const local_bucket_offsets = (int**) shmem_malloc(WORKERS_PER_PE * sizeof(int*));
  (*send_offsets) = (int**) shmem_malloc(WORKERS_PER_PE * sizeof(int*));
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    local_bucket_offsets[wid] = (int*) shmem_malloc(NUM_BUCKETS * sizeof(int));
    (*send_offsets)[wid] = (int*) shmem_malloc(NUM_BUCKETS * sizeof(int));
  }

  timer_start(&timers[TIMER_BOFFSET]);

  int wid;
  // parallel block
#if defined(_SHMEM_WORKERS)
  compute_local_bucket_offsets_t args = { &local_bucket_offsets, local_bucket_sizes, send_offsets };
  int lowBound = 0;
  int highBound = WORKERS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(compute_local_bucket_offsets_async, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(wid) schedule (dynamic,1) 
#endif
  for(wid=0; wid<WORKERS_PER_PE; wid++) {
    local_bucket_offsets[wid][0] = 0;
    (*send_offsets)[wid][0] = 0;
    int temp = 0;
    for(uint64_t i = 1; i < NUM_BUCKETS; i++){
      temp = local_bucket_offsets[wid][i-1] + local_bucket_sizes[wid][i-1];
      local_bucket_offsets[wid][i] = temp;
      (*send_offsets)[wid][i] = temp;
    } 
  }
#endif

  timer_stop(&timers[TIMER_BOFFSET]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
    sprintf(msg,"V_Rank %d: local bucket offsets: ", v_rank);
    for(uint64_t i = 0; i < NUM_BUCKETS; ++i){
      if(i < PRINT_MAX)
        sprintf(msg + strlen(msg),"%d ", local_bucket_offsets[wid][i]);
    }
    sprintf(msg + strlen(msg),"\n");
    printf("%s",msg);
  }
  fflush(stdout);
  my_turn_complete();
#endif
  return local_bucket_offsets;
}

#if defined(_SHMEM_WORKERS)
typedef struct bucketize_local_keys_t {
  KEY_TYPE *** restrict my_local_bucketed_keys;
  KEY_TYPE const ** restrict const my_keys;
  int ** restrict const local_bucket_offsets;  
} bucketize_local_keys_t;

void bucketize_local_keys_async(void* arg, int wid) {
  bucketize_local_keys_t* args = (bucketize_local_keys_t*) arg;
  KEY_TYPE ** restrict my_local_bucketed_keys = *(args->my_local_bucketed_keys);
  KEY_TYPE const ** restrict const my_keys = args->my_keys;
  int ** restrict const local_bucket_offsets = args->local_bucket_offsets;

  for(uint64_t i = 0; i < NUM_KEYS_PER_WORKERS; ++i){
    const KEY_TYPE key = my_keys[wid][i];
    const uint32_t bucket_index = key / BUCKET_WIDTH; 
    uint32_t index;
    assert(local_bucket_offsets[wid][bucket_index] >= 0);
    index = local_bucket_offsets[wid][bucket_index]++;
    assert(index < NUM_KEYS_PER_WORKERS);
    my_local_bucketed_keys[wid][index] = key;
  }
}
#endif

/*
 * Places local keys into their corresponding local bucket.
 * The contents of each bucket are not sorted.
 */
static inline KEY_TYPE ** bucketize_local_keys(KEY_TYPE const ** restrict const my_keys,
                                              int ** restrict const local_bucket_offsets)
{
  KEY_TYPE ** restrict const my_local_bucketed_keys = (KEY_TYPE**) shmem_malloc(WORKERS_PER_PE * sizeof(KEY_TYPE*));
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    my_local_bucketed_keys[wid] = (KEY_TYPE*) shmem_malloc(NUM_KEYS_PER_WORKERS * sizeof(KEY_TYPE));
  }

  timer_start(&timers[TIMER_BUCKETIZE]);

  int wid;
  // parallel block
#if defined(_SHMEM_WORKERS)
  bucketize_local_keys_t args = { &my_local_bucketed_keys, my_keys, local_bucket_offsets };
  int lowBound = 0;
  int highBound = WORKERS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(bucketize_local_keys_async, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(wid) schedule (dynamic,1) 
#endif
  for(wid=0; wid<WORKERS_PER_PE; wid++) {
    for(uint64_t i = 0; i < NUM_KEYS_PER_WORKERS; ++i){
      const KEY_TYPE key = my_keys[wid][i];
      const uint32_t bucket_index = key / BUCKET_WIDTH; 
      uint32_t index;
      assert(local_bucket_offsets[wid][bucket_index] >= 0);
      index = local_bucket_offsets[wid][bucket_index]++;
      assert(index < NUM_KEYS_PER_WORKERS);
      my_local_bucketed_keys[wid][index] = key;
    }
  }
#endif

  timer_stop(&timers[TIMER_BUCKETIZE]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
    sprintf(msg,"V_Rank %d: local bucketed keys: ", v_rank);
    for(uint64_t i = 0; i < NUM_KEYS_PER_WORKERS; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", my_local_bucketed_keys[wid][i]);
    }
    sprintf(msg + strlen(msg),"\n");
    printf("%s",msg);
  }
  fflush(stdout);
  my_turn_complete();
#endif
  return my_local_bucketed_keys;
}


/*
 * Each PE sends the contents of its local buckets to the PE that owns that bucket.
 */
static inline KEY_TYPE ** exchange_keys(int const ** restrict const send_offsets,
                                       int const ** restrict const local_bucket_sizes,
                                       KEY_TYPE const ** restrict const my_local_bucketed_keys)
{
  timer_start(&timers[TIMER_ATA_KEYS]);

  const int my_rank = shmem_my_pe();
  unsigned int total_keys_sent[WORKERS_PER_PE];
  memset(total_keys_sent, 0x00, WORKERS_PER_PE*sizeof(int));

  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
  const int v_my_rank = GET_VIRTUAL_RANK(my_rank, wid);
  for(uint64_t i = 0; i < NUM_PES*WORKERS_PER_PE; ++i){
#ifdef PERMUTE
    const int target_pe = permute_array[i];
    assert("Not implemented for WORKERS_PER_PE" && 0);
#elif INCAST
    const int target_pe = i;
    assert("Not implemented for WORKERS_PER_PE" && 0);
#else
    const int v_target_pe = (v_my_rank + i) % (NUM_PES*WORKERS_PER_PE);
#endif
    const int r_target_pe = GET_REAL_RANK(v_target_pe);
    const int read_offset_from_self = send_offsets[wid][v_target_pe];
    const int my_send_size = local_bucket_sizes[wid][v_target_pe];
    const int min_v_rank_target_pe = GET_VIRTUAL_RANK(r_target_pe, 0);
    const int wid_r_target_pe = v_target_pe - min_v_rank_target_pe;
    const long long int write_offset_into_target = shmem_longlong_fadd(&receive_offset[wid_r_target_pe], (long long int)my_send_size, r_target_pe);
#ifdef DEBUG
    printf("V_Rank: %d Target: %d Offset into target: %lld Offset into myself: %d Send Size: %d\n",
        v_my_rank, v_target_pe, write_offset_into_target, read_offset_from_self, my_send_size);
#endif

    shmem_int_put(&(my_bucket_keys[wid_r_target_pe][write_offset_into_target]),
                  &(my_local_bucketed_keys[wid][read_offset_from_self]),
                  my_send_size,
                  r_target_pe);

    total_keys_sent[wid] += my_send_size;
  } }

#ifdef BARRIER_ATA
  shmem_barrier_all();
#endif

  timer_stop(&timers[TIMER_ATA_KEYS]);
  timer_count(&timers[TIMER_ATA_KEYS], total_keys_sent);

#ifdef DEBUG
  wait_my_turn();
  for(int wid = 0; wid<WORKERS_PER_PE; wid++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
    char msg[1024];
    sprintf(msg,"V_Rank %d: Bucket Size %lld | Total Keys Sent: %u | Keys after exchange:",
                            v_rank, receive_offset[wid], total_keys_sent[wid]);
    for(long long int i = 0; i < receive_offset[wid]; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", my_bucket_keys[wid][i]);
    }
    sprintf(msg + strlen(msg),"\n");
    printf("%s",msg);
  }
  fflush(stdout);
  my_turn_complete();
#endif
  return my_bucket_keys;
}

#if defined(_SHMEM_WORKERS)
void count_local_keys_async(void* arg, int wid) {
  int ** restrict const my_local_key_counts = *((int *** restrict const )arg);

  const int my_rank = shmem_my_pe();
  const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
  const int my_min_key = v_rank * BUCKET_WIDTH;
  // Count the occurences of each key in my bucket
  for(long long int i = 0; i < my_bucket_size[wid]; ++i) {
    const unsigned int key_index = my_bucket_keys[wid][i] - my_min_key;
    assert(my_bucket_keys[wid][i] >= my_min_key);
    assert(key_index < BUCKET_WIDTH);
    my_local_key_counts[wid][key_index]++;
  }   
}
#endif

/*
 * Counts the occurence of each key in my bucket. 
 * Key indices into the count array are the key's value minus my bucket's 
 * minimum key value to allow indexing from 0.
 * my_bucket_keys: All keys in my bucket unsorted [my_rank * BUCKET_WIDTH, (my_rank+1)*BUCKET_WIDTH)
 */
static inline int ** count_local_keys()
{
  int ** restrict const my_local_key_counts = (int**) malloc(WORKERS_PER_PE * sizeof(int*));
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    my_local_key_counts[wid] = (int*) malloc(BUCKET_WIDTH*sizeof(int));
    memset(my_local_key_counts[wid], 0x00, BUCKET_WIDTH * sizeof(int));
  }

  timer_start(&timers[TIMER_SORT]);

  const int my_rank = shmem_my_pe();
  int wid;
  // parallel block
#if defined(_SHMEM_WORKERS)
  int lowBound = 0;
  int highBound = WORKERS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(count_local_keys_async, (void*)(&my_local_key_counts), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(wid) schedule (dynamic,1) 
#endif
  for(wid=0; wid<WORKERS_PER_PE; wid++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
    const int my_min_key = v_rank * BUCKET_WIDTH;
    // Count the occurences of each key in my bucket
    for(long long int i = 0; i < my_bucket_size[wid]; ++i) {
      const unsigned int key_index = my_bucket_keys[wid][i] - my_min_key;
      assert(my_bucket_keys[wid][i] >= my_min_key);
      assert(key_index < BUCKET_WIDTH);
      my_local_key_counts[wid][key_index]++;
    }   
  }
#endif

  timer_stop(&timers[TIMER_SORT]);

#ifdef DEBUG
  wait_my_turn();
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    char msg[4096];
    const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
    sprintf(msg,"V_Rank %d: Bucket Size %lld | Local Key Counts:", v_rank, my_bucket_size[wid]);
    for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", my_local_key_counts[wid][i]);
    }
    sprintf(msg + strlen(msg),"\n");
    printf("%s",msg);
  }
  fflush(stdout);
  my_turn_complete();
#endif

  return my_local_key_counts;
}

#if defined(_SHMEM_WORKERS)
typedef struct verify_results_t {
  int const ** restrict const my_local_key_counts;
  int* error;
} verify_results_t;

void verify_results_async(void* arg, int wid) {
  verify_results_t* args = (verify_results_t*) arg;
  int const ** restrict const my_local_key_counts = args->my_local_key_counts;
  int* error = args->error;
  const int my_rank = shmem_my_pe();

  const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
  const int my_min_key = v_rank * BUCKET_WIDTH;
  const int my_max_key = (v_rank+1) * BUCKET_WIDTH - 1;
  // Verify all keys are within bucket boundaries
  for(long long int i = 0; i < my_bucket_size[wid]; ++i){
    const int key = my_bucket_keys[wid][i];
    if((key < my_min_key) || (key > my_max_key)){
      printf("Rank %d Failed Verification!\n",v_rank);
      printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
      error[wid] = 1;
    }
  }
  // Verify the sum of the key population equals the expected bucket size
  long long int bucket_size_test = 0;
  for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
    bucket_size_test +=  my_local_key_counts[wid][i];
  }
  if(bucket_size_test != my_bucket_size[wid]){
    printf("Rank %d Failed Verification!\n",v_rank);
    printf("Actual Bucket Size: %lld Should be %lld\n", bucket_size_test, my_bucket_size[wid]);
    error[wid] = 1;
  }
}
#endif
/*
 * Verifies the correctness of the sort. 
 * Ensures all keys are within a PE's bucket boundaries.
 * Ensures the final number of keys is equal to the initial.
 */
static int verify_results(int const ** restrict const my_local_key_counts)
{

  shmem_barrier_all();

  int error[WORKERS_PER_PE];
  memset(error, 0x00, sizeof(int) * WORKERS_PER_PE);

  const int my_rank = shmem_my_pe();

  int wid;
  // parallel block
#if defined(_SHMEM_WORKERS)
  verify_results_t args = { my_local_key_counts, error };
  int lowBound = 0;
  int highBound = WORKERS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(verify_results_async, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(wid) schedule (dynamic,1) 
#endif
  for(wid=0; wid<WORKERS_PER_PE; wid++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, wid);
    const int my_min_key = v_rank * BUCKET_WIDTH;
    const int my_max_key = (v_rank+1) * BUCKET_WIDTH - 1;
    // Verify all keys are within bucket boundaries
    for(long long int i = 0; i < my_bucket_size[wid]; ++i){
      const int key = my_bucket_keys[wid][i];
      if((key < my_min_key) || (key > my_max_key)){
        printf("Rank %d Failed Verification!\n",v_rank);
        printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
        error[wid] = 1;
      }
    }
    // Verify the sum of the key population equals the expected bucket size
    long long int bucket_size_test = 0;
    for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
      bucket_size_test +=  my_local_key_counts[wid][i];
    }
    if(bucket_size_test != my_bucket_size[wid]){
      printf("Rank %d Failed Verification!\n",v_rank);
      printf("Actual Bucket Size: %lld Should be %lld\n", bucket_size_test, my_bucket_size[wid]);
      error[wid] = 1;
    }
  }
#endif

  // NOT Parallel
  static long long int total_my_bucket_size = 0;
  int tError=0;
  for(int wid=0; wid<WORKERS_PER_PE; wid++) {
    total_my_bucket_size += my_bucket_size[wid];
    tError += error[wid];
  }

  // Verify the final number of keys equals the initial number of keys
  static long long int total_num_keys = 0;
  shmem_longlong_sum_to_all(&total_num_keys, &total_my_bucket_size, 1, 0, 0, NUM_PES, llWrk, pSync);
  shmem_barrier_all();

  if(total_num_keys != (long long int)(NUM_KEYS_PER_WORKERS * NUM_PES * WORKERS_PER_PE)){
    if(my_rank == ROOT_PE){
      printf("Verification Failed!\n");
      printf("Actual total number of keys: %lld Expected %" PRIu64 "\n", total_num_keys, NUM_KEYS_PER_PE * NUM_PES );
      tError = 1;
    }
  }

  return tError;
}

/*
 * Gathers all the timing information from each PE and prints
 * it to a file. All information from a PE is printed as a row in a tab seperated file
 */
static void log_times(char * log_file)
{
  FILE * fp = NULL;

  for(uint64_t i = 0; i < TIMER_NTIMERS; ++i){
    timers[i].all_times = gather_rank_times(&timers[i]);
    timers[i].all_counts = gather_rank_counts(&timers[i]);
  }

  if(shmem_my_pe() == ROOT_PE)
  {
    int print_names = 0;
    if(file_exists(log_file) != 1){
      print_names = 1;
    }

    if((fp = fopen(log_file, "a+b"))==NULL){
      perror("Error opening log file:");
      exit(1);
    }

    if(print_names == 1){
      print_run_info(fp);
      print_timer_names(fp);
    }
    print_timer_values(fp);

    report_summary_stats();

    fclose(fp);
  }

}

/*
 * Computes the average total time and average all2all time and prints it to the command line
 */
static void report_summary_stats(void)
{
  
  if(timers[TIMER_TOTAL].seconds_iter > 0) {
    const uint32_t num_records = NUM_PES * timers[TIMER_TOTAL].seconds_iter;
    double temp = 0.0;
    for(uint64_t i = 0; i < num_records; ++i){
      temp += timers[TIMER_TOTAL].all_times[i];
    }
      printf("Average total time (per PE): %f seconds\n", temp/num_records);
  }

  if(timers[TIMER_ATA_KEYS].seconds_iter >0) {
    const uint32_t num_records = NUM_PES * timers[TIMER_ATA_KEYS].seconds_iter;
    double temp = 0.0;
    for(uint64_t i = 0; i < num_records; ++i){
      temp += timers[TIMER_ATA_KEYS].all_times[i];
    }
    printf("Average all2all time (per PE): %f seconds\n", temp/num_records);
  }
}

/*
 * Prints all the labels for each timer as a row to the file specified by 'fp'
 */
static void print_timer_names(FILE * fp)
{
  for(uint64_t i = 0; i < TIMER_NTIMERS; ++i){
    if(timers[i].seconds_iter > 0){
      fprintf(fp, "%s (sec)\t", timer_names[i]);
    }
    if(timers[i].count_iter > 0){
      fprintf(fp, "%s_COUNTS\t", timer_names[i]);
    }
  }
  fprintf(fp,"\n");
}

/*
 * Prints all the relevant runtime parameters as a row to the file specified by 'fp'
 */
static void print_run_info(FILE * fp)
{
  fprintf(fp,"SHMEM\t");
  fprintf(fp,"NUM_PES %" PRIu64 "\t", NUM_PES);
  fprintf(fp,"Max_Key %" PRIu64 "\t", MAX_KEY_VAL); 
  fprintf(fp,"Num_Iters %u\t", NUM_ITERATIONS);

  switch(SCALING_OPTION){
    case STRONG: {
        fprintf(fp,"Strong Scaling: %" PRIu64 " total keys\t", NUM_KEYS_PER_PE * NUM_PES);
        break;
      }
    case WEAK: {
        fprintf(fp,"Weak Scaling: %" PRIu64 " keys per PE\t", NUM_KEYS_PER_PE);
        break;
      }
    case WEAK_ISOBUCKET: {
        fprintf(fp,"Weak Scaling Constant Bucket Width: %" PRIu64 "u keys per PE \t", NUM_KEYS_PER_PE);
        fprintf(fp,"Constant Bucket Width: %" PRIu64 "\t", BUCKET_WIDTH);
        break;
      }
    default:
      {
        fprintf(fp,"Invalid Scaling Option!\t");
        break;
      }

  }

#ifdef PERMUTE
    fprintf(fp,"Randomized All2All\t");
#elif INCAST
    fprintf(fp,"Incast All2All\t");
#else
    fprintf(fp,"Round Robin All2All\t");
#endif

    fprintf(fp,"\n");
}

/*
 * Prints all of the timining information for an individual PE as a row
 * to the file specificed by 'fp'. 
 */
static void print_timer_values(FILE * fp)
{
  unsigned int num_records = NUM_PES * NUM_ITERATIONS; 

  for(uint64_t i = 0; i < num_records; ++i) {
    for(int t = 0; t < TIMER_NTIMERS; ++t){
      if(timers[t].all_times != NULL){
        fprintf(fp,"%f\t", timers[t].all_times[i]);
      }
      if(timers[t].all_counts != NULL){
        fprintf(fp,"%u\t", timers[t].all_counts[i]);
      }
    }
    fprintf(fp,"\n");
  }
}

/* 
 * Aggregates the per PE timing information
 */ 
static double * gather_rank_times(_timer_t * const timer)
{
  if(timer->seconds_iter > 0) {

    assert(timer->seconds_iter == timer->num_iters);

    const unsigned int num_records = NUM_PES * timer->seconds_iter;
#ifdef OPENSHMEM_COMPLIANT
    double * my_times = shmem_malloc(timer->seconds_iter * sizeof(double));
#else
    double * my_times = shmalloc(timer->seconds_iter * sizeof(double));
#endif
    memcpy(my_times, timer->seconds, timer->seconds_iter * sizeof(double));

#ifdef OPENSHMEM_COMPLIANT
    double * all_times = shmem_malloc( num_records * sizeof(double));
#else
    double * all_times = shmalloc( num_records * sizeof(double));
#endif

    shmem_barrier_all();

    shmem_fcollect64(all_times, my_times, timer->seconds_iter, 0, 0, NUM_PES, pSync);
    shmem_barrier_all();

#ifdef OPENSHMEM_COMPLIANT
    shmem_free(my_times);
#else
    shfree(my_times);
#endif

    return all_times;
  }
  else{
    return NULL;
  }
}

/*
 * Aggregates the per PE timing 'count' information 
 */
static unsigned int * gather_rank_counts(_timer_t * const timer)
{
  if(timer->count_iter > 0){
    const unsigned int num_records = NUM_PES * timer->num_iters;

#ifdef OPENSHMEM_COMPLIANT
    unsigned int * my_counts = shmem_malloc(timer->num_iters * sizeof(unsigned int));
#else
    unsigned int * my_counts = shmalloc(timer->num_iters * sizeof(unsigned int));
#endif
    memcpy(my_counts, timer->count, timer->num_iters*sizeof(unsigned int));

#ifdef OPENSHMEM_COMPLIANT
    unsigned int * all_counts = shmem_malloc( num_records * sizeof(unsigned int) );
#else
    unsigned int * all_counts = shmalloc( num_records * sizeof(unsigned int) );
#endif

    shmem_barrier_all();

    shmem_collect32(all_counts, my_counts, timer->num_iters, 0, 0, NUM_PES, pSync);

    shmem_barrier_all();

#ifdef OPENSHMEM_COMPLIANT
    shmem_free(my_counts);
#else
    shfree(my_counts);
#endif

    return all_counts;
  }
  else{
    return NULL;
  }

}
/*
 * Seeds each rank based on the worker number, rank and time
 */
static inline pcg32_random_t seed_my_worker(int wid)
{
  const unsigned int my_rank = shmem_my_pe();
  const unsigned int my_virtual_rank = GET_VIRTUAL_RANK(my_rank, wid);
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, (uint64_t) my_virtual_rank, (uint64_t) my_virtual_rank );
  return rng;
}

/*
 * Seeds each rank based on the rank number and time
 */
static inline pcg32_random_t seed_my_rank(void)
{
  const unsigned int my_rank = shmem_my_pe();
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, (uint64_t) my_rank, (uint64_t) my_rank );
  return rng;
}

/*
 * Initializes the work array required for SHMEM collective functions
 */
static void init_shmem_sync_array(long * restrict const pSync)
{
  for(uint64_t i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; ++i){
    pSync[i] = _SHMEM_SYNC_VALUE;
  }
  shmem_barrier_all();
}

/*
 * Tests whether or not a file exists. 
 * Returns 1 if file exists
 * Returns 0 if file does not exist
 */
static int file_exists(char * filename)
{
  struct stat buffer;

  if(stat(filename,&buffer) == 0){
    return 1;
  }
  else {
    return 0;
  }

}

#ifdef DEBUG
static void wait_my_turn()
{
  shmem_barrier_all();
  whose_turn = 0;
  shmem_barrier_all();
  const int my_rank = shmem_my_pe();

  shmem_int_wait_until((int*)&whose_turn, SHMEM_CMP_EQ, my_rank);
  sleep(1);

}

static void my_turn_complete()
{
  const int my_rank = shmem_my_pe();
  const int next_rank = my_rank+1;

  if(my_rank < (NUM_PES-1)){ // Last rank updates no one
    shmem_int_put((int *) &whose_turn, &next_rank, 1, next_rank);
  }
  shmem_barrier_all();
}
#endif

#ifdef PERMUTE
/*
 * Creates a randomly ordered array of PEs used in the exchange_keys function
 */
static void create_permutation_array()
{

  permute_array = (int *) malloc( NUM_PES * sizeof(int) );

  for(uint64_t i = 0; i < NUM_PES; ++i){
    permute_array[i] = i;
  }

  shuffle(permute_array, NUM_PES, sizeof(int));
}

/*
 * Randomly shuffles a generic array
 */
static void shuffle(void * array, size_t n, size_t size)
{
  char tmp[size];
  char * arr = array;
  size_t stride = size * sizeof(char);
  if(n > 1){
    for(size_t i = 0; i < (n - 1); ++i){
      size_t rnd = (size_t) rand();
      size_t j = i + rnd/(RAND_MAX/(n - i) + 1);
      memcpy(tmp, arr + j*stride, size);
      memcpy(arr + j*stride, arr + i*stride, size);
      memcpy(arr + i*stride, tmp, size);
    }
  }
}
#endif

