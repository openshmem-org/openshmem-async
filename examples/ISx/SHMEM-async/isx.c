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
#include "pthread.h"
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

#define EXTRA_STATS

#ifdef EXTRA_STATS
float avg_time=0, avg_time_all2all = 0;
#endif
/*
 * This variable sets the maximum number of chunks allowed
 * to participate in computation per pe.
 */
int CHUNKS_PER_PE=1;
#if defined (_SHMEM_WORKERS) || defined (_OPENMP)
#define MAX_CHUNKS_ALLOWED 512
#define GET_VIRTUAL_RANK(rank, chunk) ((rank * CHUNKS_PER_PE) + (chunk))
#define GET_REAL_RANK(vrank) ((int)(vrank / CHUNKS_PER_PE))
#define PARALLEL_FOR_MODE SHMEM_PARALLEL_FOR_RECURSIVE_MODE
//#define PARALLEL_FOR_MODE SHMEM_PARALLEL_FOR_FLAT_MODE
#else
#define MAX_CHUNKS_ALLOWED 1
#define GET_VIRTUAL_RANK(rank, chunk) (rank)
#define GET_REAL_RANK(vrank) (vrank)
#endif
uint64_t NUM_KEYS_PER_CHUNK;
int actual_num_workers;
// This is done due to current limitation that entrypoint function
// cannot accept arguments. This will be resolved in future version of 
// AsyncSHMEM
int m_argc;
char** m_argv;

volatile int whose_turn;

long long int* receive_offset;

/*
 * In case of CHUNKS_PER_PE>1, setting KEY_BUFFER_SIZE_PER_PE=(1uLL<<28uLL) 
 * would cause compilation error (in gasnet) with the declaration of:
 * "KEY_TYPE my_bucket_keys[CHUNKS_PER_PE][KEY_BUFFER_SIZE_PER_PE];"
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

// used at sender PE to hold the number of keys its planning to 
// exchange with each of the chunk at the remote PE
long long int* total_keys_per_pe_per_chunk;
// this is the result of alltoall operation after each
// PE sends their list of total_keys_per_pe_per_chunk
long long int** total_keys_per_pe_per_chunk_alltoall;
// we subdivide both the sent and receive copies of my_bucket_keys
// variable into regions for each PE and each of their chunks.

// Here we store the starting index in the my_bucket_keys_sent array
// for a particular PE's keys
long long int* starting_index_pe_sent_bucket;
// Similar to above but this is meant for the my_bucket_keys_received array
long long int* starting_index_pe_receive_bucket;
// result of the alltoall operation of the above variable starting_index_pe_receive_bucket
long long int** starting_index_pe_receive_bucket_alltoall;
// this array contains keys received from remote PEs and their chunks
// after the key exchange call
static KEY_TYPE* my_bucket_keys_received;
#define MY_START_INDEX_RECEIVE_BUCKET(pe, myrank) (starting_index_pe_receive_bucket_alltoall[pe][myrank]) 
#define GET_INDEX_RECEIVE_BUCKET(pe, myrank, x)  (MY_START_INDEX_RECEIVE_BUCKET(pe, myrank) + (x))

// this array contains the key this PE would sent to all the remote PE and their respective chunks
static KEY_TYPE* my_bucket_keys_sent;
#define START_INDEX_PE_SENT_BUCKET(pe)  starting_index_pe_sent_bucket[pe]
#define GET_INDEX_SENT_BUCKET(pe, x)  (START_INDEX_PE_SENT_BUCKET(pe) + x)

/*
 * For easy access the received keys (after exchange operation), we store
 * the keys in the array my_bucket_keys_sent (yes, the name of the variable is not supporting this).
 * This macro simply gets the index of a key in a particular chunk.
 * This macro is used only after exchanging the keys with remote PEs.
 */
#define RECEIVE_BUCKET_INDEX_IN_CHUNK(chunk, index) my_bucket_keys_sent[total_keys_per_pe_per_chunk[chunk] + index]

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

#ifdef EXTRA_STATS
  _timer_t stage_time;
  if(shmem_my_pe() == 0) {
    printf("\n-----\nmkdir timedrun fake\n\n");
    timer_start(&stage_time);
  }
#endif
#if defined(_SHMEM_WORKERS)
  shmem_workers_init(entrypoint, NULL);
#else
  entrypoint(NULL);
#endif

#ifdef EXTRA_STATS
  if(shmem_my_pe() == 0) {
    just_timer_stop(&stage_time);
    double tTime = ( stage_time.stop.tv_sec - stage_time.start.tv_sec ) + ( stage_time.stop.tv_nsec - stage_time.start.tv_nsec )/1E9;
    avg_time *= 1000;
    avg_time_all2all *= 1000;
    printf("\n============================ MMTk Statistics Totals ============================\n");
    if(NUM_ITERATIONS == 1) { //TODO: fix time calculation below for more number of iterations
      printf("time.mu\tt.ATA_KEYS\tt.MAKE_INPUT\tt.COUNT_BUCKET_SIZES\tt.BUCKETIZE\tt.COMPUTE_OFFSETS\tt.LOCAL_SORT\tnWorkers\tnPEs\n");
      double TIMES[TIMER_NTIMERS];
      memset(TIMES, 0x00, sizeof(double) * TIMER_NTIMERS);
      for(int i=0; i<NUM_PES; i++) {
        for(int t = 0; t < TIMER_NTIMERS; ++t){
          if(t==2) continue;
          int index = t < 2 ? t : t-1;
          if(timers[t].all_times != NULL){
            TIMES[index] += timers[t].all_times[i];
          }
        }
      }
      for(int t = 0; t < TIMER_NTIMERS-1; ++t){
        printf("%.3f\t", (TIMES[t]/NUM_PES)*1000);
      }
      printf("%d\t%d\n",actual_num_workers,NUM_PES);
    }
    else {
      printf("time.mu\ttimeAll2All\tnWorkers\tnPEs\n");
      printf("%.3f\t%.3f\t%d\t%d\n",avg_time,avg_time_all2all,actual_num_workers,NUM_PES);
      printf("Total time: %.3f\n",avg_time);
    }

    printf("------------------------------ End MMTk Statistics -----------------------------\n");
    printf("===== TEST PASSED in %.3f msec =====\n",(tTime*1000));
  }
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
  CHUNKS_PER_PE = chunks_env ? atoi(chunks_env) : 1;
#pragma omp parallel
  actual_num_workers = omp_get_num_threads();
#elif defined(_SHMEM_WORKERS)
  const char* chunks_env = getenv("ISX_PE_CHUNKS");
  CHUNKS_PER_PE = chunks_env ? atoi(chunks_env) : 1;
  actual_num_workers = shmem_n_workers();
#else
  actual_num_workers = 1;
  CHUNKS_PER_PE = 1;
#endif
  assert(CHUNKS_PER_PE <= MAX_CHUNKS_ALLOWED);
  NUM_PES = (uint64_t) shmem_n_pes();
  MAX_KEY_VAL = DEFAULT_MAX_KEY;
  NUM_BUCKETS = NUM_PES*CHUNKS_PER_PE;
  BUCKET_WIDTH = (uint64_t) ceil((double)MAX_KEY_VAL/NUM_BUCKETS);
  char * log_file = argv[2];
  char scaling_msg[64];

  switch(SCALING_OPTION){
    case STRONG:
      {
        TOTAL_KEYS = (uint64_t) atoi(argv[1]);
        NUM_KEYS_PER_CHUNK = (uint64_t) ceil((double)TOTAL_KEYS/(NUM_PES * CHUNKS_PER_PE));
        NUM_KEYS_PER_PE = NUM_KEYS_PER_CHUNK * CHUNKS_PER_PE;
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
        assert(NUM_KEYS_PER_PE%CHUNKS_PER_PE == 0); // if this is not satisfied, change the input
        NUM_KEYS_PER_CHUNK = NUM_KEYS_PER_PE / CHUNKS_PER_PE;
        sprintf(scaling_msg,"WEAK");
        break;
      }

    case WEAK_ISOBUCKET:
      {
        NUM_KEYS_PER_PE = ((uint64_t) (atoi(argv[1]))) * actual_num_workers;
        assert(NUM_KEYS_PER_PE%CHUNKS_PER_PE == 0); // if this is not satisfied, change the input
        NUM_KEYS_PER_CHUNK = NUM_KEYS_PER_PE / CHUNKS_PER_PE;
        BUCKET_WIDTH = ISO_BUCKET_WIDTH; 
        MAX_KEY_VAL = (uint64_t) (NUM_PES * actual_num_workers *  BUCKET_WIDTH);
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
    printf("  Number of Keys per Chunk: %" PRIu64 "\n", NUM_KEYS_PER_CHUNK);
    printf("  Number of Chunks per PE (ISX_PE_CHUNKS): %d\n",CHUNKS_PER_PE);
#elif defined(_SHMEM_WORKERS)
    printf("  AsyncSHMEM Version, total workers: %d\n",actual_num_workers);
    printf("  Number of Keys per Chunk: %" PRIu64 "\n", NUM_KEYS_PER_CHUNK);
    printf("  Number of Chunks per PE (ISX_PE_CHUNKS): %d\n",CHUNKS_PER_PE);
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

  /*
   * These two allocations are 1gb each. I have found that moving these allocation
   * later in the code (and not here), i.e. after doing below other small allocations, would
   * cause unexplained segfaults. Hence, its IMPORTANT do not move this allocation
   * from this place.
   */
  my_bucket_keys_received = (KEY_TYPE *) shmem_malloc(sizeof(KEY_TYPE) * KEY_BUFFER_SIZE_PER_PE);
  my_bucket_keys_sent = (KEY_TYPE *) shmem_malloc(sizeof(KEY_TYPE) * KEY_BUFFER_SIZE_PER_PE);

  receive_offset = (long long int*) malloc(sizeof(long long int) * CHUNKS_PER_PE);

  starting_index_pe_sent_bucket = (long long int*) malloc(sizeof(long long int) * NUM_PES);
  starting_index_pe_receive_bucket = (long long int*) shmem_malloc(sizeof(long long int) * NUM_PES);
  starting_index_pe_receive_bucket_alltoall = (long long int**) shmem_malloc(sizeof(long long int*) * NUM_PES);
  total_keys_per_pe_per_chunk = (long long int*) shmem_malloc(sizeof(long long int) * NUM_PES * CHUNKS_PER_PE);
  total_keys_per_pe_per_chunk_alltoall = (long long int**) shmem_malloc(sizeof(long long int*) * NUM_PES * CHUNKS_PER_PE);
  for(int i=0; i<NUM_PES; i++) {
    total_keys_per_pe_per_chunk_alltoall[i] = (long long int*) shmem_malloc(sizeof(long long int) * CHUNKS_PER_PE);
    starting_index_pe_receive_bucket_alltoall[i] = (long long int*) shmem_malloc(sizeof(long long int) * NUM_PES);
  }

  for(uint64_t i = 0; i < (NUM_ITERATIONS + BURN_IN); ++i)
  {
    // Reset timers after burn in 
    if(i == BURN_IN){ init_timers(NUM_ITERATIONS); } 

    shmem_barrier_all();

    timer_start(&timers[TIMER_TOTAL]);

    // Reset offsets used in exchange_keys
    memset(receive_offset, 0x00, CHUNKS_PER_PE * sizeof(long long int));
    memset(total_keys_per_pe_per_chunk, 0x00, NUM_PES * CHUNKS_PER_PE * sizeof(long long int));

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

// Time phase = 2790.646
    int** my_local_key_counts = count_local_keys();

    shmem_barrier_all();

    timer_stop(&timers[TIMER_TOTAL]);

    // Only the last iteration is verified
    if(i == NUM_ITERATIONS) {      
      err = verify_results(my_local_key_counts);
    }

    shmem_barrier_all();
 
    // free all local malloc-ed memory
    for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
      free(my_local_key_counts[chunk]);
      free(my_local_bucketed_keys[chunk]);      
      free(local_bucket_sizes[chunk]);      
      free(local_bucket_offsets[chunk]);      
      free(my_keys[chunk]);      
    }
    free(my_local_key_counts);
    free(my_local_bucketed_keys);
    free(local_bucket_sizes);
    free(local_bucket_offsets);
    free(my_keys);

  }

  // free all global memory
  for(int k=0; k<NUM_PES; k++) {
    shmem_free(total_keys_per_pe_per_chunk_alltoall[k]);
    shmem_free(starting_index_pe_receive_bucket_alltoall[k]);
  }
  shmem_free(total_keys_per_pe_per_chunk_alltoall);
  shmem_free(starting_index_pe_receive_bucket_alltoall);
  shmem_free(total_keys_per_pe_per_chunk);
  shmem_free(starting_index_pe_receive_bucket);
  free(starting_index_pe_sent_bucket);
  shmem_free(my_bucket_keys_sent);
  shmem_free(my_bucket_keys_received);

  return err;
}

#if defined(_SHMEM_WORKERS)
void make_input_async(void *args, int chunk) {
  KEY_TYPE ** restrict my_keys = *((KEY_TYPE *** restrict) args);
  // note that we are able to move this malloc here
  // just because its a local malloc and not a shmem_malloc.
  my_keys[chunk] = (KEY_TYPE*) malloc(NUM_KEYS_PER_CHUNK * sizeof(KEY_TYPE));
  
  KEY_TYPE* restrict my_keys_1D = my_keys[chunk];
  pcg32_random_t rng = seed_my_chunk(chunk);
  for(uint64_t i = 0; i < NUM_KEYS_PER_CHUNK; ++i) {
    *my_keys_1D = pcg32_boundedrand_r(&rng, MAX_KEY_VAL);
    my_keys_1D += 1;
  }
}
#endif

/*
 * Generates uniformly random keys [0, MAX_KEY_VAL] on each rank using the time and rank
 * number as a seed
 */
static KEY_TYPE ** make_input(void)
{
  KEY_TYPE ** restrict const my_keys = (KEY_TYPE**) malloc(CHUNKS_PER_PE * sizeof(KEY_TYPE*));
 
  timer_start(&timers[TIMER_INPUT]);

  int chunk; 
#if defined(_SHMEM_WORKERS)
  int lowBound = 0;
  int highBound = CHUNKS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(make_input_async, (void*)(&my_keys), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  // parallel block
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    my_keys[chunk] = (KEY_TYPE*) malloc(NUM_KEYS_PER_CHUNK * sizeof(KEY_TYPE));
    pcg32_random_t rng = seed_my_chunk(chunk);
    KEY_TYPE* restrict my_keys_1D = my_keys[chunk];
    for(uint64_t i = 0; i < NUM_KEYS_PER_CHUNK; ++i) {
      *my_keys_1D = pcg32_boundedrand_r(&rng, MAX_KEY_VAL);
      my_keys_1D += 1;
    }
  }
#endif

  timer_stop(&timers[TIMER_INPUT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
    sprintf(msg,"V_Rank %d: Initial Keys: ", v_rank);
    for(uint64_t i = 0; i < NUM_KEYS_PER_CHUNK; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", my_keys[chunk][i]);
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

void count_local_bucket_sizes_async(void* arg, int chunk) {
  count_local_bucket_sizes_t* args = (count_local_bucket_sizes_t*) arg;
  int ** restrict local_bucket_sizes = (*(args->local_bucket_sizes));
  local_bucket_sizes[chunk] = (int*) malloc(NUM_BUCKETS * sizeof(int));
  // note that we are able to move this malloc here
  // just because its a local malloc and not a shmem_malloc.
  KEY_TYPE ** const restrict const my_keys = args->my_keys;
  KEY_TYPE* restrict my_keys_1D = my_keys[chunk];
  int * restrict local_bucket_sizes_1D = local_bucket_sizes[chunk];

  init_array(local_bucket_sizes[chunk] , NUM_BUCKETS); // doing memset 0x00
  for(uint64_t i = 0; i < NUM_KEYS_PER_CHUNK; ++i){
    const uint32_t bucket_index = my_keys_1D[i]/BUCKET_WIDTH;
    local_bucket_sizes_1D[bucket_index]++;
  }
}
#endif

/*
 * Computes the size of each bucket by iterating all keys and incrementing
 * their corresponding bucket's size
 */
static inline int ** count_local_bucket_sizes(KEY_TYPE const ** restrict const my_keys)
{
  int ** restrict const local_bucket_sizes = (int**) malloc(CHUNKS_PER_PE * sizeof(int*));

  timer_start(&timers[TIMER_BCOUNT]);

  int chunk;
  // parallel block
#if defined(_SHMEM_WORKERS)
  count_local_bucket_sizes_t args = { &local_bucket_sizes, my_keys };
  int lowBound = 0;
  int highBound = CHUNKS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(count_local_bucket_sizes_async, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    local_bucket_sizes[chunk] = (int*) malloc(NUM_BUCKETS * sizeof(int));
    init_array(local_bucket_sizes[chunk] , NUM_BUCKETS); // doing memset 0x00
    KEY_TYPE* restrict my_keys_1D = my_keys[chunk];
    int * restrict local_bucket_sizes_1D = local_bucket_sizes[chunk];
    for(uint64_t i = 0; i < NUM_KEYS_PER_CHUNK; ++i){
      const uint32_t bucket_index = my_keys_1D[i]/BUCKET_WIDTH;
      local_bucket_sizes_1D[bucket_index]++;
    }
  }
#endif

  timer_stop(&timers[TIMER_BCOUNT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
    sprintf(msg,"V_Rank %d: local bucket sizes: ", v_rank);
    for(uint64_t i = 0; i < NUM_BUCKETS; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", local_bucket_sizes[chunk][i]);
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

void compute_local_bucket_offsets_async(void *arg, int chunk) {
  compute_local_bucket_offsets_t *args = (compute_local_bucket_offsets_t*) arg;
  int **restrict local_bucket_offsets = *(args->local_bucket_offsets);
  int const ** restrict const local_bucket_sizes = args->local_bucket_sizes;
  int *** restrict send_offsets = args->send_offsets;
  // note that we are able to move this malloc here
  // just because its a local malloc and not a shmem_malloc.
  local_bucket_offsets[chunk] = (int*) malloc(NUM_BUCKETS * sizeof(int));
  (*send_offsets)[chunk] = (int*) malloc(NUM_BUCKETS * sizeof(int));
  int * restrict send_offsets_1D = (*send_offsets)[chunk];

  local_bucket_offsets[chunk][0] = 0;
  (*send_offsets)[chunk][0] = 0;
  int temp = 0;
  int * restrict local_bucket_offsets_1D = local_bucket_offsets[chunk];
  int * restrict local_bucket_sizes_1D = local_bucket_sizes[chunk];
  for(uint64_t i = 1; i < NUM_BUCKETS; i++){
    temp = local_bucket_offsets_1D[i-1] + local_bucket_sizes_1D[i-1];
    local_bucket_offsets_1D[i] = temp;
    send_offsets_1D[i] = temp;
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
  int ** restrict const local_bucket_offsets = (int**) malloc(CHUNKS_PER_PE * sizeof(int*));
  (*send_offsets) = (int**) malloc(CHUNKS_PER_PE * sizeof(int*));

  timer_start(&timers[TIMER_BOFFSET]);

  int chunk;
  // parallel block
#if defined(_SHMEM_WORKERS)
  compute_local_bucket_offsets_t args = { &local_bucket_offsets, local_bucket_sizes, send_offsets };
  int lowBound = 0;
  int highBound = CHUNKS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(compute_local_bucket_offsets_async, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    local_bucket_offsets[chunk] = (int*) malloc(NUM_BUCKETS * sizeof(int));
    (*send_offsets)[chunk] = (int*) malloc(NUM_BUCKETS * sizeof(int));
    local_bucket_offsets[chunk][0] = 0;
    (*send_offsets)[chunk][0] = 0;
    int temp = 0;
    int * restrict local_bucket_offsets_1D = local_bucket_offsets[chunk];
    int * restrict local_bucket_sizes_1D = local_bucket_sizes[chunk];
    int * restrict send_offsets_1D = (*send_offsets)[chunk];
    for(uint64_t i = 1; i < NUM_BUCKETS; i++){
      temp = local_bucket_offsets_1D[i-1] + local_bucket_sizes_1D[i-1];
      local_bucket_offsets_1D[i] = temp;
      send_offsets_1D[i] = temp;
    } 
  }
#endif

  timer_stop(&timers[TIMER_BOFFSET]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
    sprintf(msg,"V_Rank %d: local bucket offsets: ", v_rank);
    for(uint64_t i = 0; i < NUM_BUCKETS; ++i){
      if(i < PRINT_MAX)
        sprintf(msg + strlen(msg),"%d ", local_bucket_offsets[chunk][i]);
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

void bucketize_local_keys_async(void* arg, int chunk) {
  bucketize_local_keys_t* args = (bucketize_local_keys_t*) arg;
  KEY_TYPE ** restrict my_local_bucketed_keys = *(args->my_local_bucketed_keys);
  // note that we are able to move this malloc here
  // just because its a local malloc and not a shmem_malloc.
  my_local_bucketed_keys[chunk] = (KEY_TYPE*) malloc(NUM_KEYS_PER_CHUNK * sizeof(KEY_TYPE));
  KEY_TYPE const ** restrict const my_keys = args->my_keys;
  int ** restrict const local_bucket_offsets = args->local_bucket_offsets;

  KEY_TYPE * restrict my_keys_1D = my_keys[chunk];
  int * restrict local_bucket_offsets_1D = local_bucket_offsets[chunk];
  KEY_TYPE * restrict my_local_bucketed_keys_1D = my_local_bucketed_keys[chunk];
  for(uint64_t i = 0; i < NUM_KEYS_PER_CHUNK; ++i){
    const KEY_TYPE key = my_keys_1D[i];
    const uint32_t bucket_index = key / BUCKET_WIDTH; 
    uint32_t index;
    assert(local_bucket_offsets_1D[bucket_index] >= 0);
    index = local_bucket_offsets_1D[bucket_index]++;
    assert(index < NUM_KEYS_PER_CHUNK);
    my_local_bucketed_keys_1D[index] = key;
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
  KEY_TYPE ** restrict const my_local_bucketed_keys = (KEY_TYPE**) malloc(CHUNKS_PER_PE * sizeof(KEY_TYPE*));

  timer_start(&timers[TIMER_BUCKETIZE]);

  int chunk;
  // parallel block
#if defined(_SHMEM_WORKERS)
  bucketize_local_keys_t args = { &my_local_bucketed_keys, my_keys, local_bucket_offsets };
  int lowBound = 0;
  int highBound = CHUNKS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(bucketize_local_keys_async, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    my_local_bucketed_keys[chunk] = (KEY_TYPE*) malloc(NUM_KEYS_PER_CHUNK * sizeof(KEY_TYPE));
    KEY_TYPE * restrict my_keys_1D = my_keys[chunk];
    int * restrict local_bucket_offsets_1D = local_bucket_offsets[chunk];
    KEY_TYPE * restrict my_local_bucketed_keys_1D = my_local_bucketed_keys[chunk];
    for(uint64_t i = 0; i < NUM_KEYS_PER_CHUNK; ++i){
      const KEY_TYPE key = my_keys_1D[i];
      const uint32_t bucket_index = key / BUCKET_WIDTH;
      uint32_t index;
      assert(local_bucket_offsets_1D[bucket_index] >= 0);
      index = local_bucket_offsets_1D[bucket_index]++;
      assert(index < NUM_KEYS_PER_CHUNK);
      my_local_bucketed_keys_1D[index] = key;
    }
  }
#endif

  timer_stop(&timers[TIMER_BUCKETIZE]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
    sprintf(msg,"V_Rank %d: local bucketed keys: ", v_rank);
    for(uint64_t i = 0; i < NUM_KEYS_PER_CHUNK; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", my_local_bucketed_keys[chunk][i]);
    }
    sprintf(msg + strlen(msg),"\n");
    printf("%s",msg);
  }
  fflush(stdout);
  my_turn_complete();
#endif
  return my_local_bucketed_keys;
}

long long int** keys_sent_currently_pe_chunk = NULL;
long long int** fixed_starting_index_pe_chunk = NULL;
unsigned int * total_keys_sent_per_chunk = NULL;
#if defined(_SHMEM_WORKERS)
pthread_mutex_t *mymutex = NULL;

typedef struct copy_into_sent_bucket_t {
  int const ** restrict const local_bucket_sizes;
  int const ** restrict const send_offsets;
  KEY_TYPE const ** restrict const my_local_bucketed_keys;
} copy_into_sent_bucket_t;

void copy_into_sent_bucket(void * args, int chunk) {
  copy_into_sent_bucket_t * arg = (copy_into_sent_bucket_t*) args;
  int const ** restrict const  send_offsets = arg->send_offsets;
  int const ** restrict const local_bucket_sizes = arg->local_bucket_sizes;
  KEY_TYPE const ** restrict const my_local_bucketed_keys = arg->my_local_bucketed_keys;

  const int my_rank = shmem_my_pe();
  const int v_my_rank = GET_VIRTUAL_RANK(my_rank, chunk);
  for(uint64_t i = 0; i < NUM_PES*CHUNKS_PER_PE; ++i){
    const int v_target_pe = (v_my_rank + i) % (NUM_PES*CHUNKS_PER_PE);
    const int r_target_pe = GET_REAL_RANK(v_target_pe);
    const int read_offset_from_self = send_offsets[chunk][v_target_pe];
    const int my_send_size = local_bucket_sizes[chunk][v_target_pe];
    const int min_v_rank_target_pe = GET_VIRTUAL_RANK(r_target_pe, 0);
    const int chunk_r_target_pe = v_target_pe - min_v_rank_target_pe;
    long long int keys_sent_currently_pe_chunk_current;

    // critical section -- 2 lines
    const int index = r_target_pe*CHUNKS_PER_PE + chunk_r_target_pe;
    pthread_mutex_lock(&(mymutex[index]));
    keys_sent_currently_pe_chunk_current = keys_sent_currently_pe_chunk[r_target_pe][chunk_r_target_pe];
    keys_sent_currently_pe_chunk[r_target_pe][chunk_r_target_pe] += my_send_size;
    pthread_mutex_unlock(&(mymutex[index]));

    const long long int write_offset_into_target = fixed_starting_index_pe_chunk[r_target_pe][chunk_r_target_pe] +
                                                     keys_sent_currently_pe_chunk_current;

    const long long int max_key_pe_chunk = total_keys_per_pe_per_chunk[r_target_pe*CHUNKS_PER_PE + chunk_r_target_pe];
    const long long int remaining_key_slots_left = max_key_pe_chunk - keys_sent_currently_pe_chunk_current;
    assert(my_send_size <= remaining_key_slots_left);

    memcpy(&(my_bucket_keys_sent[GET_INDEX_SENT_BUCKET(r_target_pe, write_offset_into_target)]),
           &(my_local_bucketed_keys[chunk][read_offset_from_self]),
           my_send_size * sizeof(KEY_TYPE));
    total_keys_sent_per_chunk[chunk] += my_send_size;
  }
}

void calculate_num_keys_tosend(void* args, int chunk) {
  copy_into_sent_bucket_t * arg = (copy_into_sent_bucket_t*) args;
  int const ** restrict const  send_offsets = arg->send_offsets;
  int const ** restrict const local_bucket_sizes = arg->local_bucket_sizes;
  const int my_rank = shmem_my_pe();
  const int v_my_rank = GET_VIRTUAL_RANK(my_rank, chunk);
  for(uint64_t i = 0; i < NUM_PES*CHUNKS_PER_PE; ++i){
#ifdef PERMUTE
    const int target_pe = permute_array[i];
    assert("Not implemented for CHUNKS_PER_PE" && 0);
#elif INCAST
    const int target_pe = i;
    assert("Not implemented for CHUNKS_PER_PE" && 0);
#else
    const int v_target_pe = (v_my_rank + i) % (NUM_PES*CHUNKS_PER_PE);
#endif
    const int r_target_pe = GET_REAL_RANK(v_target_pe);
    const int read_offset_from_self = send_offsets[chunk][v_target_pe];
    const int my_send_size = local_bucket_sizes[chunk][v_target_pe];
    const int min_v_rank_target_pe = GET_VIRTUAL_RANK(r_target_pe, 0);
    const int chunk_r_target_pe = v_target_pe - min_v_rank_target_pe;
    const int index = r_target_pe*CHUNKS_PER_PE + chunk_r_target_pe;
    pthread_mutex_lock(&(mymutex[index]));
    // critical section -- 1 line
    total_keys_per_pe_per_chunk[index] += my_send_size;
    pthread_mutex_unlock(&(mymutex[index]));
  }
}

void organize_keys_from_receive_bucket(void* args, int chunk) {
  const int my_rank = shmem_my_pe();
  long long int** read_offset = (long long int**) args;
  for(uint64_t i = 0; i < NUM_PES; ++i){
    long long int total_keys = total_keys_per_pe_per_chunk_alltoall[i][chunk];
    if(total_keys == 0) continue;

    memcpy(&(RECEIVE_BUCKET_INDEX_IN_CHUNK(chunk, receive_offset[chunk])),
           &(my_bucket_keys_received[GET_INDEX_RECEIVE_BUCKET(my_rank, i,read_offset[chunk][i])]),
           total_keys * sizeof(KEY_TYPE));
    receive_offset[chunk] += total_keys;
  }
}
#endif
 
/*
 * Each PE sends the contents of its local buckets to the PE that owns that bucket.
 */
static inline KEY_TYPE ** exchange_keys(int const ** restrict const send_offsets,
                                       int const ** restrict const local_bucket_sizes,
                                       KEY_TYPE const ** restrict const my_local_bucketed_keys)
{
  timer_start(&timers[TIMER_ATA_KEYS]);
  
  const int my_rank = shmem_my_pe();

  /*
   * These variables are used only in this exchange function and allocated only once.
   * This is during the first iteration only.
   */
  if(keys_sent_currently_pe_chunk == NULL) {
    keys_sent_currently_pe_chunk = (long long int**) malloc(sizeof(long long int*) * NUM_PES);
    fixed_starting_index_pe_chunk = (long long int**) malloc(sizeof(long long int*) * NUM_PES);
    total_keys_sent_per_chunk = (unsigned int*) malloc(sizeof(unsigned int) * CHUNKS_PER_PE);
    for(int i=0; i<NUM_PES; i++) {
      keys_sent_currently_pe_chunk[i] = (long long int*) malloc(sizeof(long long int) * CHUNKS_PER_PE);
      fixed_starting_index_pe_chunk[i] = (long long int*) malloc(sizeof(long long int) * CHUNKS_PER_PE);
    }
#if defined(_SHMEM_WORKERS)
    mymutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t) * NUM_PES * CHUNKS_PER_PE);
    for(int i=0; i<NUM_PES*CHUNKS_PER_PE; i++) {
      pthread_mutex_init(&(mymutex[i]), NULL);
    }
#endif
  }

  /*
   * We first calculate the number of keys we would be sending to
   * each of the chunks in all the remote PEs. 
   */
#if defined(_SHMEM_WORKERS)
  copy_into_sent_bucket_t args = { local_bucket_sizes, send_offsets, my_local_bucketed_keys };
  int lowBound = 0;
  int highBound = CHUNKS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(calculate_num_keys_tosend, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
  int chunk;
  omp_lock_t mymutex[NUM_PES * CHUNKS_PER_PE];
  for(int i=0; i<NUM_PES * CHUNKS_PER_PE; i++) omp_init_lock(&(mymutex[i]));
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
  const int v_my_rank = GET_VIRTUAL_RANK(my_rank, chunk);
  for(uint64_t i = 0; i < NUM_PES*CHUNKS_PER_PE; ++i){
    #ifdef PERMUTE
    const int target_pe = permute_array[i];
    assert("Not implemented for CHUNKS_PER_PE" && 0);
#elif INCAST
    const int target_pe = i;
    assert("Not implemented for CHUNKS_PER_PE" && 0);
#else
    const int v_target_pe = (v_my_rank + i) % (NUM_PES*CHUNKS_PER_PE);
#endif
    const int r_target_pe = GET_REAL_RANK(v_target_pe);
    const int read_offset_from_self = send_offsets[chunk][v_target_pe];
    const int my_send_size = local_bucket_sizes[chunk][v_target_pe];
    const int min_v_rank_target_pe = GET_VIRTUAL_RANK(r_target_pe, 0);
    const int chunk_r_target_pe = v_target_pe - min_v_rank_target_pe;
    const int index = r_target_pe*CHUNKS_PER_PE + chunk_r_target_pe;
    omp_set_lock(&(mymutex[index]));
    // critical section -- 1 line
    total_keys_per_pe_per_chunk[index] += my_send_size;
    omp_unset_lock(&(mymutex[index]));
  }
  } 
#endif

  /*
   * We should inform the remote PEs about the number 
   * of keys we would be sending them for each of their chunks
   */
  for(int i=0; i<NUM_PES; i++) {
    if(my_rank == i) {  
      memcpy(total_keys_per_pe_per_chunk_alltoall[my_rank], 
             &(total_keys_per_pe_per_chunk[my_rank * CHUNKS_PER_PE]), 
             CHUNKS_PER_PE * sizeof(long long int));
    }
    else {
      shmem_longlong_put(total_keys_per_pe_per_chunk_alltoall[my_rank], 
             &(total_keys_per_pe_per_chunk[i * CHUNKS_PER_PE]), 
             (long long int) CHUNKS_PER_PE , i);
    }
  }

#ifdef BARRIER_ATA
  shmem_barrier_all();
#endif

  /*
   * We use the same array my_bucket_keys_sent for storing the 
   * keys we would be sending to remote PE. Hence find the offset
   * in my_bucket_keys_sent for each PE
   */
  long long int index_per_pe_sent = 0;
  /*
   * We use the same array my_bucket_keys_received for storing the
   * keys we are going to get from all the remote PEs. Hence,
   * here we first calculate the starting offset for each PE
   * in the receive buffer (my_bucket_keys_received)
   */
  long long int index_per_pe_received = 0;
  for(int i=0; i<NUM_PES; i++) {
    START_INDEX_PE_SENT_BUCKET(i) = index_per_pe_sent;
    starting_index_pe_receive_bucket[i] = index_per_pe_received;
    for(int j=0; j<CHUNKS_PER_PE; j++) {
      index_per_pe_sent += total_keys_per_pe_per_chunk[i*CHUNKS_PER_PE + j];
      index_per_pe_received += total_keys_per_pe_per_chunk_alltoall[i][j];
    }
  }

  /*
   * The offset calculation we did above is currently only available
   * to this PE. All the other remote PEs should also be aware of this
   * as then only they can decide in which slot in my my_bucket_keys_received
   * they do a put operation and at what index
   */
  for(int i=0; i<NUM_PES; i++) {
    if(my_rank == i) {
      memcpy(starting_index_pe_receive_bucket_alltoall[my_rank], starting_index_pe_receive_bucket, 
             sizeof(long long int) * NUM_PES);
    }
    else {
      shmem_longlong_put(starting_index_pe_receive_bucket_alltoall[my_rank], starting_index_pe_receive_bucket,
             (long long int)NUM_PES, i);
    }
  }

#ifdef BARRIER_ATA
  shmem_barrier_all();
#endif

  /*
   * Calculate the starting index for each chunk (for each PE) in the 
   * array my_bucket_keys_sent. We also keep track of the total number
   * of keys we want to send to each chunk on remote PE
   */
  static long long int total_expected = 0;
  static long long int total_num_keys = 0;
  for(int i=0; i<NUM_PES; i++) {
    memset(keys_sent_currently_pe_chunk[i], 0x00, sizeof(long long int) * CHUNKS_PER_PE);
    long long int keys_in = 0, keys_out = 0;
    for(int j=0; j<CHUNKS_PER_PE; j++) {
      keys_in += total_keys_per_pe_per_chunk_alltoall[i][j];
      fixed_starting_index_pe_chunk[i][j] = keys_out;
      keys_out += total_keys_per_pe_per_chunk[(i * CHUNKS_PER_PE) + j];
    }
    total_expected += keys_in;
  }

#ifdef DEBUG
  // Verify the final number of keys equals the initial number of keys
  // This is just an error check and may be removed in the production version of this ISx
  shmem_longlong_sum_to_all(&total_num_keys, &total_expected, 1, 0, 0, NUM_PES, llWrk, pSync);
  shmem_barrier_all();
  assert(total_num_keys == (long long int)(NUM_KEYS_PER_CHUNK * NUM_PES * CHUNKS_PER_PE));
#endif
  total_expected = 0;
  total_num_keys = 0;

  unsigned int total_keys_sent = 0;
  memset(total_keys_sent_per_chunk, 0x00, CHUNKS_PER_PE*sizeof(unsigned int));

  /*
   * In this version of ISx each chunk is thought to be owned
   * by a virtual PE. There are total of CHUNKS_PER_PE numbers of
   * virtual PEs on each actual (or real) PEs.
   *
   * Each of these virtual PE would perform exchange with remote virtual PEs
   * If we don't optimize this exchange then there would be total of 
   * CHUNKS_PER_PE * NUM_PES * NUM_PES. This would be a severe bottleneck
   * in this version of ISx. To avoid this issue we are doing optimized 
   * communications. 
   * We first copy of the keys the virtual PEs on this rank wants to 
   * send to all the virtual PEs on any remote rank. We store that
   * keys inside array my_bucket_keys_sent.
   * Now, when the keys are copied, all the keys can be send using one put operation.
   */
#if defined(_SHMEM_WORKERS)
  //copy_into_sent_bucket_t args = { local_bucket_sizes, send_offsets, my_local_bucketed_keys };
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(copy_into_sent_bucket , (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
  const int v_my_rank = GET_VIRTUAL_RANK(my_rank, chunk);
  for(uint64_t i = 0; i < NUM_PES*CHUNKS_PER_PE; ++i){
    const int v_target_pe = (v_my_rank + i) % (NUM_PES*CHUNKS_PER_PE);
    const int r_target_pe = GET_REAL_RANK(v_target_pe);
    const int read_offset_from_self = send_offsets[chunk][v_target_pe];
    const int my_send_size = local_bucket_sizes[chunk][v_target_pe];
    const int min_v_rank_target_pe = GET_VIRTUAL_RANK(r_target_pe, 0);
    const int chunk_r_target_pe = v_target_pe - min_v_rank_target_pe;
    long long int keys_sent_currently_pe_chunk_current;

    // critical section -- 2 lines
    const int index = r_target_pe*CHUNKS_PER_PE + chunk_r_target_pe;
    omp_set_lock(&(mymutex[index]));
    keys_sent_currently_pe_chunk_current = keys_sent_currently_pe_chunk[r_target_pe][chunk_r_target_pe];
    keys_sent_currently_pe_chunk[r_target_pe][chunk_r_target_pe] += my_send_size;
    omp_unset_lock(&(mymutex[index]));

    const long long int write_offset_into_target = fixed_starting_index_pe_chunk[r_target_pe][chunk_r_target_pe] +
                                                     keys_sent_currently_pe_chunk_current;

    const long long int max_key_pe_chunk = total_keys_per_pe_per_chunk[r_target_pe*CHUNKS_PER_PE + chunk_r_target_pe];
    const long long int remaining_key_slots_left = max_key_pe_chunk - keys_sent_currently_pe_chunk_current;
    assert(my_send_size <= remaining_key_slots_left);

    memcpy(&(my_bucket_keys_sent[GET_INDEX_SENT_BUCKET(r_target_pe, write_offset_into_target)]),
           &(my_local_bucketed_keys[chunk][read_offset_from_self]),
           my_send_size * sizeof(KEY_TYPE));
    total_keys_sent_per_chunk[chunk] += my_send_size;
  }
  }
#endif
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) total_keys_sent += total_keys_sent_per_chunk[chunk];

#ifdef BARRIER_ATA
  shmem_barrier_all();
#endif
 
  /*
   * Send the keys in the my_bucket_keys_sent array to the remote virtual PEs. We store all these
   * keys at remote PEs in the array my_bucket_keys_received that is also partitioned for 
   * each virtual PE and for each rank
   */ 
  for(uint64_t i = 0; i < NUM_PES; ++i){
    long long int total_keys_to_send = 0;
    for(int j=0; j<CHUNKS_PER_PE; j++) {
      if(total_keys_per_pe_per_chunk[(i * CHUNKS_PER_PE) + j] == 0) continue;
      total_keys_to_send += total_keys_per_pe_per_chunk[(i * CHUNKS_PER_PE) + j];
      assert(keys_sent_currently_pe_chunk[i][j] == total_keys_per_pe_per_chunk[(i * CHUNKS_PER_PE) + j]);
    }    
    if(total_keys_to_send == 0) continue;
    if(i == my_rank) {
      memcpy(&(my_bucket_keys_received[GET_INDEX_RECEIVE_BUCKET(my_rank,my_rank,0)]), &(my_bucket_keys_sent[GET_INDEX_SENT_BUCKET(my_rank,0)]), (int) total_keys_to_send * sizeof(KEY_TYPE));
    }
    else {
      shmem_int_put(&(my_bucket_keys_received[GET_INDEX_RECEIVE_BUCKET(i,my_rank,0)]), &(my_bucket_keys_sent[GET_INDEX_SENT_BUCKET(i,0)]), (int) total_keys_to_send, i);
    }
  }

#ifdef BARRIER_ATA
  shmem_barrier_all();
#endif

  /*
   * We have performed the key exchange with all the remote PEs and the result
   * is currently stored in the array my_bucket_keys_received. Later code of
   * this ISx would be using the array my_bucket_keys to use the keys resulting 
   * from the global alltoall exchange. Hence, to support those code part, we 
   * copy the keys we received in my_bucket_keys_received into the array
   * my_bucket_keys.
   * In future we can get rid of this extra my_bucket_keys_received and instead
   * use only one receive buffer my_bucket_keys
   */
  long long int key_index_current[NUM_PES];
  memset(key_index_current, 0x00, sizeof(long long int) * NUM_PES);
  /*
   * We would now use the starting CHUNKS_PER_PE indexes in the array total_keys_per_pe_per_chunk
   * to store the starting index for the first key in the chunk in the received bucket keys.
   * Hence, the index total_keys_per_pe_per_chunk[I] (where, I <CHUNKS_PER_PE) 
   * represents the index to the first key in the chunk number I.
   * This index is in the array my_bucket_keys_sent. We will now switch to this array
   * to store the received keys 
   */
    
  long long keys_index = 0;
  long long int** read_offset = (long long int**) malloc(sizeof(long long int*) * CHUNKS_PER_PE);
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    total_keys_per_pe_per_chunk[chunk] = keys_index;
    read_offset[chunk] = (long long int*) malloc(sizeof(long long int) * NUM_PES);
    for(uint64_t i = 0; i < NUM_PES; ++i){
      const long long int total_keys = total_keys_per_pe_per_chunk_alltoall[i][chunk];
      if(total_keys == 0) continue;
      keys_index += total_keys;
      read_offset[chunk][i] = key_index_current[i];
      key_index_current[i] += total_keys;
    }
  }
#if defined(_SHMEM_WORKERS)
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(organize_keys_from_receive_bucket, (void*)read_offset, NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
int chunk;
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    for(uint64_t i = 0; i < NUM_PES; ++i){
      long long int total_keys = total_keys_per_pe_per_chunk_alltoall[i][chunk];
      if(total_keys == 0) continue;

      memcpy(&(RECEIVE_BUCKET_INDEX_IN_CHUNK(chunk, receive_offset[chunk])),
             &(my_bucket_keys_received[GET_INDEX_RECEIVE_BUCKET(my_rank, i,read_offset[chunk][i])]),
             total_keys * sizeof(KEY_TYPE));
      receive_offset[chunk] += total_keys;
    }
  }
#endif
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) { free(read_offset[chunk]); }
  free(read_offset);

  timer_stop(&timers[TIMER_ATA_KEYS]);
  timer_count(&timers[TIMER_ATA_KEYS], total_keys_sent);

#ifdef DEBUG
  wait_my_turn();
  for(int chunk = 0; chunk<CHUNKS_PER_PE; chunk++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
    char msg[1024];
    sprintf(msg,"V_Rank %d: Bucket Size %lld | Total Keys Sent: %u | Keys after exchange:",
                            v_rank, receive_offset[chunk], total_keys_sent_per_chunk[chunk]);
    for(long long int i = 0; i < receive_offset[chunk]; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", RECEIVE_BUCKET_INDEX_IN_CHUNK(chunk, i));
    }
    sprintf(msg + strlen(msg),"\n");
    printf("%s",msg);
  }
  fflush(stdout);
  my_turn_complete();
#endif
  return &my_bucket_keys_sent; // TODO remove this return as not required
}

KEY_TYPE* temp_buffer = NULL;

#if defined(_SHMEM_WORKERS)
void count_local_keys_async(void* arg, int chunk) {
  const long long int size = receive_offset[chunk];
  int ** restrict const my_local_key_counts = *((int *** restrict const )arg);
  // note that we are able to move this malloc here
  // just because its a local malloc and not a shmem_malloc.
  my_local_key_counts[chunk] = (int*) malloc(BUCKET_WIDTH*sizeof(int));
  memset(my_local_key_counts[chunk], 0x00, BUCKET_WIDTH * sizeof(int));

  if(size == 0) return;
  KEY_TYPE* restrict my_bucket_keys_1D = &(RECEIVE_BUCKET_INDEX_IN_CHUNK(chunk, 0));

  const int my_rank = shmem_my_pe();
  const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
  const int my_min_key = v_rank * BUCKET_WIDTH;
  int * restrict my_local_key_counts_1D = my_local_key_counts[chunk];
  // Count the occurences of each key in my bucket
  for(long long int i = 0; i < size; ++i) {
    const unsigned int key_index = *(my_bucket_keys_1D) - my_min_key;
    assert(*(my_bucket_keys_1D) >= my_min_key);
    assert(key_index < BUCKET_WIDTH);
    my_local_key_counts_1D[key_index]++;
    my_bucket_keys_1D += 1;
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
  int ** restrict const my_local_key_counts = (int**) malloc(CHUNKS_PER_PE * sizeof(int*));

  timer_start(&timers[TIMER_SORT]);

  const int my_rank = shmem_my_pe();
  int chunk;
  // parallel block
#if defined(_SHMEM_WORKERS)
  int lowBound = 0;
  int highBound = CHUNKS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(count_local_keys_async, (void*)(&my_local_key_counts), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    my_local_key_counts[chunk] = (int*) malloc(BUCKET_WIDTH*sizeof(int));
    memset(my_local_key_counts[chunk], 0x00, BUCKET_WIDTH * sizeof(int));

    const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
    const int my_min_key = v_rank * BUCKET_WIDTH;
    KEY_TYPE* restrict my_bucket_keys_1D = &(RECEIVE_BUCKET_INDEX_IN_CHUNK(chunk, 0));
    int * restrict my_local_key_counts_1D = my_local_key_counts[chunk];
    const long long int size = receive_offset[chunk];
    // Count the occurences of each key in my bucket
    for(long long int i = 0; i < size; ++i) {
      const unsigned int key_index = *(my_bucket_keys_1D) - my_min_key;
      assert(*(my_bucket_keys_1D) >= my_min_key);
      assert(key_index < BUCKET_WIDTH);
      my_local_key_counts_1D[key_index]++;
      my_bucket_keys_1D += 1;
    }   
  }
#endif

  timer_stop(&timers[TIMER_SORT]);

#ifdef DEBUG
  wait_my_turn();
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    char msg[4096];
    const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
    sprintf(msg,"V_Rank %d: Bucket Size %lld | Local Key Counts:", v_rank, receive_offset[chunk]);
    for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
      if(i < PRINT_MAX)
      sprintf(msg + strlen(msg),"%d ", my_local_key_counts[chunk][i]);
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

void verify_results_async(void* arg, int chunk) {
  verify_results_t* args = (verify_results_t*) arg;
  int const ** restrict const my_local_key_counts = args->my_local_key_counts;
  int* error = args->error;
  const int my_rank = shmem_my_pe();

  const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
  const int my_min_key = v_rank * BUCKET_WIDTH;
  const int my_max_key = (v_rank+1) * BUCKET_WIDTH - 1;
  KEY_TYPE* restrict my_bucket_keys_1D = &(RECEIVE_BUCKET_INDEX_IN_CHUNK(chunk, 0));
  // Verify all keys are within bucket boundaries
  for(long long int i = 0; i < receive_offset[chunk]; ++i){
    const int key = my_bucket_keys_1D[i];
    if((key < my_min_key) || (key > my_max_key)){
      printf("Rank %d Failed Verification!\n",v_rank);
      printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
      error[chunk] = 1;
    }
  }
  // Verify the sum of the key population equals the expected bucket size
  long long int bucket_size_test = 0;
  int * restrict my_local_key_counts_1D = my_local_key_counts[chunk];
  for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
    bucket_size_test +=  my_local_key_counts_1D[i];
  }
  if(bucket_size_test != receive_offset[chunk]){
    printf("Rank %d Failed Verification!\n",v_rank);
    printf("Actual Bucket Size: %lld Should be %lld\n", bucket_size_test, receive_offset[chunk]);
    error[chunk] = 1;
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

  int error[CHUNKS_PER_PE];
  memset(error, 0x00, sizeof(int) * CHUNKS_PER_PE);

  const int my_rank = shmem_my_pe();

  int chunk;
  // parallel block
#if defined(_SHMEM_WORKERS)
  verify_results_t args = { my_local_key_counts, error };
  int lowBound = 0;
  int highBound = CHUNKS_PER_PE;
  int stride = 1;
  int tile_size = 1;
  int loop_dimension = 1;
  shmem_task_scope_begin();
  shmem_parallel_for_nbi(verify_results_async, (void*)(&args), NULL, lowBound, highBound, stride, tile_size, loop_dimension, PARALLEL_FOR_MODE);
  shmem_task_scope_end();
#else
#if defined(_OPENMP)
#pragma omp parallel for private(chunk) schedule (dynamic,1) 
#endif
  for(chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    const int v_rank = GET_VIRTUAL_RANK(my_rank, chunk);
    const int my_min_key = v_rank * BUCKET_WIDTH;
    const int my_max_key = (v_rank+1) * BUCKET_WIDTH - 1;
    KEY_TYPE* restrict my_bucket_keys_1D = &(RECEIVE_BUCKET_INDEX_IN_CHUNK(chunk, 0));
    // Verify all keys are within bucket boundaries
    for(long long int i = 0; i < receive_offset[chunk]; ++i){
      const int key = my_bucket_keys_1D[i];
      if((key < my_min_key) || (key > my_max_key)){
        printf("Rank %d Failed Verification!\n",v_rank);
        printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
        error[chunk] = 1;
      }
    }
    // Verify the sum of the key population equals the expected bucket size
    long long int bucket_size_test = 0;
    int * restrict my_local_key_counts_1D = my_local_key_counts[chunk];
    for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
      bucket_size_test +=  my_local_key_counts_1D[i];
    }
    if(bucket_size_test != receive_offset[chunk]){
      printf("Rank %d Failed Verification!\n",v_rank);
      printf("Actual Bucket Size: %lld Should be %lld\n", bucket_size_test, receive_offset[chunk]);
      error[chunk] = 1;
    }
  }
#endif

  // NOT Parallel
  static long long int total_my_bucket_size = 0;
  int tError=0;
  for(int chunk=0; chunk<CHUNKS_PER_PE; chunk++) {
    total_my_bucket_size += receive_offset[chunk];
    tError += error[chunk];
  }

  // Verify the final number of keys equals the initial number of keys
  static long long int total_num_keys = 0;
  shmem_longlong_sum_to_all(&total_num_keys, &total_my_bucket_size, 1, 0, 0, NUM_PES, llWrk, pSync);
  shmem_barrier_all();

  if(total_num_keys != (long long int)(NUM_KEYS_PER_CHUNK * NUM_PES * CHUNKS_PER_PE)){
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
#ifdef EXTRA_STATS
    avg_time = temp/num_records;
#endif
      printf("Average total time (per PE): %f seconds\n", temp/num_records);
  }

  if(timers[TIMER_ATA_KEYS].seconds_iter >0) {
    const uint32_t num_records = NUM_PES * timers[TIMER_ATA_KEYS].seconds_iter;
    double temp = 0.0;
    for(uint64_t i = 0; i < num_records; ++i){
      temp += timers[TIMER_ATA_KEYS].all_times[i];
    }
#ifdef EXTRA_STATS
    avg_time_all2all = temp/num_records;
#endif
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

double my_times[NUM_ITERATIONS];
unsigned int my_counts[NUM_ITERATIONS];
/* 
 * Aggregates the per PE timing information
 */ 
static double * gather_rank_times(_timer_t * const timer)
{
  if(timer->seconds_iter > 0) {

    assert(timer->seconds_iter == timer->num_iters && timer->seconds_iter == NUM_ITERATIONS);
    const unsigned int num_records = NUM_PES * timer->seconds_iter;
    
    memcpy(my_times, timer->seconds, timer->seconds_iter * sizeof(double));
#ifdef OPENSHMEM_COMPLIANT
    double * all_times = shmem_malloc( num_records * sizeof(double));
#else
    double * all_times = shmalloc( num_records * sizeof(double));
#endif

    shmem_barrier_all();
    shmem_fcollect64(all_times, my_times, timer->seconds_iter, 0, 0, NUM_PES, pSync);
    shmem_barrier_all();

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

    memcpy(my_counts, timer->count, timer->num_iters*sizeof(unsigned int));

#ifdef OPENSHMEM_COMPLIANT
    unsigned int * all_counts = shmem_malloc( num_records * sizeof(unsigned int) );
#else
    unsigned int * all_counts = shmalloc( num_records * sizeof(unsigned int) );
#endif
    shmem_barrier_all();

    shmem_collect32(all_counts, my_counts, timer->num_iters, 0, 0, NUM_PES, pSync);

    shmem_barrier_all();

    return all_counts;
  }
  else{
    return NULL;
  }

}
/*
 * Seeds each rank based on the worker number, rank and time
 */
static inline pcg32_random_t seed_my_chunk(int chunk)
{
  const unsigned int my_rank = shmem_my_pe();
  const unsigned int my_virtual_rank = GET_VIRTUAL_RANK(my_rank, chunk);
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

