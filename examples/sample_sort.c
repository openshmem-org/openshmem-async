
/*********************************************************************

                    samplesort.c: source: http://www.cse.iitd.ernet.in/~dheerajb/MPI/codes/day-3/c/samplesort.c

     Objective      : To sort unsorted integers by sample sort algorithm 
                      Write a MPI program to sort n integers, using sample
                      sort algorithm on a p processor of PARAM 10000. 
                      Assume n is multiple of p. Sorting is defined as the
                      task of arranging an unordered collection of elements
                      into monotonically increasing (or decreasing) order. 

                      postcds: array[] is sorted in ascending order ANSI C 
                      provides a quicksort function called sorting(). Its 
                      function prototype is in the standard header file
                      <stdlib.h>

     Description    : 1. Partitioning of the input data and local sort :

                      The first step of sample sort is to partition the data.
                      Initially, each one of the p processors stores n/p
                      elements of the sequence of the elements to be sorted.
                      Let Ai be the sequence stored at processor Pi. In the
                      first phase each processor sorts the local n/p elements
                      using a serial sorting algorithm. (You can use C 
                      library sorting() for performing this local sort).

                      2. Choosing the Splitters : 

                      The second phase of the algorithm determines the p-1
                      splitter elements S. This is done as follows. Each 
                      processor Pi selects p-1 equally spaced elements from
                      the locally sorted sequence Ai. These p-1 elements
                      from these p(p-1) elements are selected to be the
                      splitters.

                      3. Completing the sort :

                      In the third phase, each processor Pi uses the splitters 
                      to partition the local sequence Ai into p subsequences
                      Ai,j such that for 0 <=j <p-1 all the elements in Ai,j
                      are smaller than Sj , and for j=p-1 (i.e., the last 
                      element) Ai, j contains the rest elements. Then each 
                      processor i sends the sub-sequence Ai,j to processor Pj.
                      Finally, each processor merge-sorts the received
                      sub-sequences, completing the sorting algorithm.

     Input          : Process with rank 0 generates unsorted integers 
                      using C library call rand().

     Output         : Process with rank 0 stores the sorted elements in 
                      the file sorted_data_out.

*********************************************************************/


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>

//#define TOTAL_ELEMENT_PER_PE (4*1024*1024)
#define TOTAL_ELEMENT_PER_PE (8*1024*1024)
#define ELEMENT_T int
#define COUNT_T uint64_t
#define VERIFY
#define HC_GRANULARITY  2048
//#define HC_GRANULARITY  3072

#ifdef _OSHMEM_
#include <shmem.h>
long pSync[_SHMEM_BCAST_SYNC_SIZE];
#define RESET_BCAST_PSYNC       { int _i; for(_i=0; _i<_SHMEM_BCAST_SYNC_SIZE; _i++) { pSync[_i] = _SHMEM_SYNC_VALUE; } shmem_barrier_all(); }
#endif

#ifdef _MPI_
#include <mpi.h>
#define ELEMENT_T_MPI MPI_INT
#define shmem_malloc malloc
#define shmem_free free
#endif

long seconds() {
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

static int compare(const void *i, const void *j)
{
  if ((*(ELEMENT_T*)i) > (*(ELEMENT_T *)j))
    return (1);
  if ((*(ELEMENT_T *)i) < (*(ELEMENT_T *)j))
    return (-1);
  return (0);
}

int partition(ELEMENT_T* data, int left, int right) {
  int i = left;
  int j = right;
  ELEMENT_T tmp;
  ELEMENT_T pivot = data[(left + right) / 2];
  while (i <= j) {
    while (data[i] < pivot) i++;
    while (data[j] > pivot) j--;
    if (i <= j) {
      tmp = data[i];
      data[i] = data[j];
      data[j] = tmp;
      i++;
      j--;
    }
  }
  return i;
}

#ifdef _ASYNC_OSHMEM_
typedef struct sort_data_t {
  ELEMENT_T *buffer;
  int left;
  int right;
} sort_data_t;

void par_sort(void* arg) {
  sort_data_t *in = (sort_data_t*) arg;
  ELEMENT_T* data = in->buffer;
  int left = in->left; 
  int right = in->right;

  if (right - left + 1 > HC_GRANULARITY) {
    int index = partition(data, left, right);
    shmem_task_scope_begin();
    if (left < index - 1) {
      sort_data_t* buf = (sort_data_t*) malloc(sizeof(sort_data_t)); 
      buf->buffer = data;
      buf->left = left;
      buf->right = index - 1; 
      shmem_task_nbi(par_sort, buf, NULL);
    }
    if (index < right) {
      sort_data_t* buf = (sort_data_t*) malloc(sizeof(sort_data_t)); 
      buf->buffer = data;
      buf->left = index;
      buf->right = right; 
      shmem_task_nbi(par_sort, buf, NULL);
    }
    shmem_task_scope_end();
  }
  else {
    //  quicksort in C library
    qsort(data+left, right - left + 1, sizeof(ELEMENT_T), compare);
  }
  free(arg);
}

void sorting(ELEMENT_T* buffer, int size) {
  sort_data_t* buf = (sort_data_t*) malloc(sizeof(sort_data_t)); 
  buf->buffer = buffer;
  buf->left = 0;
  buf->right = size - 1;
  long start = seconds(); 
  shmem_task_scope_begin();
  shmem_task_nbi(par_sort, buf, NULL);
  shmem_task_scope_end();
  double end = (((double)(seconds()-start))/1000000) * 1000; // msec
  printf("Sorting (%d) = %.3f\n",size, end);
}
#else  // OpenMP
void par_sort(ELEMENT_T* data, int left, int right) {

  if (right - left + 1 > HC_GRANULARITY) {
    int index = partition(data, left, right);
    if (left < index - 1) {
      #pragma omp task 
      {
        par_sort(data, left, index - 1);
      }
    }
    if (index < right) {
      #pragma omp task 
      {
        par_sort(data, index, right);
      }
    }
    #pragma omp taskwait
  }
  else {
    //  quicksort in C library
    qsort(data+left, right - left + 1, sizeof(ELEMENT_T), compare);
  }
}

void sorting(ELEMENT_T* buffer, int size) {
  long start = seconds(); 
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      par_sort(buffer, 0, size-1);
    }
  }
  double end = (((double)(seconds()-start))/1000000) * 1000; // msec
  printf("Sorting (%d) = %.3f\n",size, end);
}
#endif

#ifdef _OSHMEM_
void shmem_scatter32(ELEMENT_T* dest, ELEMENT_T* src, int root, int count) {
  int i;
  int me = shmem_my_pe();
  int procs = shmem_n_pes();
  shmem_barrier_all();
  if(me == root) {
    for(i=0; i<procs; i++) {
      ELEMENT_T* start = &src[i * count];
      shmem_put32(dest, start, count, i);
    }
  }
  shmem_barrier_all();
}

void shmem_gather32(ELEMENT_T* dest, ELEMENT_T* src, int root, int count) {
  int me = shmem_my_pe();
  int procs = shmem_n_pes();
  shmem_barrier_all();
  ELEMENT_T* target_index = &dest[me * count];
  shmem_put32(target_index, src, count, root);
  shmem_barrier_all();
}

void shmem_alltoall32(ELEMENT_T* dest, ELEMENT_T* src, int count) {
  int i;
  int me = shmem_my_pe();
  int procs = shmem_n_pes();
  shmem_barrier_all();
  for(i=0; i<procs; i++) {
    shmem_put32(&dest[me*count], &src[i*count],  count, i);   
  }
  shmem_barrier_all();
}
#endif

#ifndef HCLIB_COMM_WORKER_FIXED
void entrypoint(void *arg) {
#else
int main (int argc, char *argv[]) {
  /**** Initialising ****/
#if defined(_OSHMEM_) && defined(_MPI_)
  printf("ERROR: You cannot use both OpenSHMEM as well as MPI\n");
  exit(1); 
#endif

#if defined(_OSHMEM_)
  shmem_init (); 
#elif defined(_MPI_)
  MPI_Init(&argc, &argv);
#else
  printf("ERROR: Use either OpenSHMEM or MPI\n");
  exit(1);
#endif
#endif
  /* Variable Declarations */

  int  Numprocs,MyRank, Root = 0;
  COUNT_T i,j,k, NoofElements, NoofElements_Bloc,
				  NoElementsToSort;
  COUNT_T       count, temp;
  ELEMENT_T 	     *Input, *InputData;
  ELEMENT_T 	     *Splitter, *AllSplitter;
  ELEMENT_T 	     *Buckets, *BucketBuffer, *LocalBucket;
  ELEMENT_T 	     *OutputBuffer, *Output, *target_index;

  long start_time = seconds();
  long local_timer_start;
  double local_timer_end, end_time, init_time;
  double communication_timer=0;
  
#if defined(_MPI_)
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
#else
  Numprocs = shmem_n_pes ();
  MyRank = shmem_my_pe ();
#endif
  assert((TOTAL_ELEMENT_PER_PE * Numprocs) < INT_MAX && "Change count type from int to uint64_t");
  NoofElements = TOTAL_ELEMENT_PER_PE * Numprocs;

  /**** Reading Input ****/
  Input = (ELEMENT_T *) shmem_malloc (NoofElements*sizeof(*Input));
  if(Input == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  if (MyRank == Root){
    printf("\n-----\nmkdir timedrun fake\n\n");

    /* Initialise random number generator  */ 
    printf ("Generating input Array for Sorting %d numbers\n",NoofElements);
    srand48((ELEMENT_T)NoofElements);
    for(i=0; i< NoofElements; i++) {
      Input[i] = rand();
    }
  }

  /**** Sending Data ****/

  NoofElements_Bloc = NoofElements / Numprocs;
  InputData = (ELEMENT_T *) shmem_malloc (NoofElements_Bloc * sizeof (*InputData));
  if(InputData == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  local_timer_start = seconds();
#if defined(_MPI_)
  MPI_Scatter(Input, NoofElements_Bloc, ELEMENT_T_MPI, InputData, 
				  NoofElements_Bloc, ELEMENT_T_MPI, Root, MPI_COMM_WORLD);
#else
  shmem_scatter32(InputData, Input, 0, NoofElements_Bloc);
#endif
  local_timer_end = (((double)(seconds()-local_timer_start))/1000000) * 1000;
  communication_timer += local_timer_end;
  init_time = local_timer_end;
  printf("Scatter = %.3f\n",local_timer_end);  

  /**** Sorting Locally ****/
  sorting(InputData, NoofElements_Bloc);

  /**** Choosing Local Splitters ****/
  Splitter = (ELEMENT_T *) shmem_malloc (sizeof (ELEMENT_T) * (Numprocs-1));
  if(Splitter == NULL) {
    printf("Error : Can not allocate memory \n");
  }
  for (i=0; i< (Numprocs-1); i++){
        Splitter[i] = InputData[NoofElements/(Numprocs*Numprocs) * (i+1)];
  } 

  /**** Gathering Local Splitters at Root ****/
  AllSplitter = (ELEMENT_T *) shmem_malloc (sizeof (ELEMENT_T) * Numprocs * (Numprocs-1));
  if(AllSplitter == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  local_timer_start = seconds();
#if defined(_MPI_)
  MPI_Gather (Splitter, Numprocs-1, ELEMENT_T_MPI, AllSplitter, Numprocs-1, 
  				  ELEMENT_T_MPI, Root, MPI_COMM_WORLD);
#else
  shmem_gather32(AllSplitter, Splitter, 0, Numprocs-1);
#endif
  local_timer_end = (((double)(seconds()-local_timer_start))/1000000) * 1000;
  communication_timer += local_timer_end;
  printf("Gather = %.3f\n",local_timer_end);  

  /**** Choosing Global Splitters ****/
  if (MyRank == Root){
    sorting (AllSplitter, Numprocs*(Numprocs-1));

    for (i=0; i<Numprocs-1; i++)
      Splitter[i] = AllSplitter[(Numprocs-1)*(i+1)];
  }
 
  local_timer_start = seconds(); 
  /**** Broadcasting Global Splitters ****/
#if defined(_MPI_)
  MPI_Bcast (Splitter, Numprocs-1, ELEMENT_T_MPI, 0, MPI_COMM_WORLD);
#else
  RESET_BCAST_PSYNC;
  shmem_broadcast32(Splitter, Splitter, Numprocs-1, 0, 0, 0, Numprocs, pSync);
  shmem_barrier_all();
#endif
  local_timer_end = (((double)(seconds()-local_timer_start))/1000000) * 1000;
  communication_timer += local_timer_end;
  printf("Bcast = %.3f\n",local_timer_end);  

  /**** Creating Numprocs Buckets locally ****/
  Buckets = (ELEMENT_T *) shmem_malloc (sizeof (ELEMENT_T) * (NoofElements + Numprocs));  
  if(Buckets == NULL) {
    printf("Error : Can not allocate memory \n");
  }
  
  j = 0;
  k = 1;

  for (i=0; i<NoofElements_Bloc; i++){
    if(j < (Numprocs-1)){
       if (InputData[i] < Splitter[j]) 
			 Buckets[((NoofElements_Bloc + 1) * j) + k++] = InputData[i]; 
       else{
	       Buckets[(NoofElements_Bloc + 1) * j] = k-1;
		    k=1;
			 j++;
		    i--;
       }
    }
    else 
       Buckets[((NoofElements_Bloc + 1) * j) + k++] = InputData[i];
  }
  Buckets[(NoofElements_Bloc + 1) * j] = k - 1;
  shmem_free(Splitter);
  shmem_free(AllSplitter);
      
  /**** Sending buckets to respective processors ****/

  BucketBuffer = (ELEMENT_T *) shmem_malloc (sizeof (ELEMENT_T) * (NoofElements + Numprocs));
  if(BucketBuffer == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  local_timer_start = seconds();
#if defined(_MPI_)
  MPI_Alltoall (Buckets, NoofElements_Bloc + 1, ELEMENT_T_MPI, BucketBuffer, 
  					 NoofElements_Bloc + 1, ELEMENT_T_MPI, MPI_COMM_WORLD);
#else
  shmem_alltoall32(BucketBuffer, Buckets, NoofElements_Bloc + 1);
#endif
  local_timer_end = (((double)(seconds()-local_timer_start))/1000000) * 1000;
  communication_timer += local_timer_end;
  printf("AlltoAll = %.3f\n",local_timer_end);  

  /**** Rearranging BucketBuffer ****/
  LocalBucket = (ELEMENT_T *) shmem_malloc (sizeof (ELEMENT_T) * 2 * NoofElements / Numprocs);
  if(LocalBucket == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  count = 1;

  for (j=0; j<Numprocs; j++) {
  k = 1;
    for (i=0; i<BucketBuffer[(NoofElements/Numprocs + 1) * j]; i++) 
      LocalBucket[count++] = BucketBuffer[(NoofElements/Numprocs + 1) * j + k++];
  }
  LocalBucket[0] = count-1;
    
  /**** Sorting Local Buckets using Bubble Sort ****/
  /*sorting (InputData, NoofElements_Bloc, sizeof(int), intcompare); */

  NoElementsToSort = LocalBucket[0];
  sorting (&LocalBucket[1], NoElementsToSort); 

  /**** Gathering sorted sub blocks at root ****/
  OutputBuffer = (ELEMENT_T *) shmem_malloc (sizeof(ELEMENT_T) * 2 * NoofElements);
  if(OutputBuffer == NULL) {
    printf("Error : Can not allocate memory \n");
  }

  local_timer_start = seconds();
#if defined(_MPI_)
  MPI_Gather (LocalBucket, 2*NoofElements_Bloc, ELEMENT_T_MPI, OutputBuffer, 
  				  2*NoofElements_Bloc, ELEMENT_T_MPI, Root, MPI_COMM_WORLD);
#else
  shmem_gather32(OutputBuffer, LocalBucket, 0, (2*NoofElements_Bloc));
#endif
  local_timer_end = (((double)(seconds()-local_timer_start))/1000000) * 1000;
  communication_timer += local_timer_end;
  printf("Gather = %.3f\n",local_timer_end);  

  end_time = (((double)(seconds()-start_time))/1000000) * 1000; // msec

  /**** Rearranging output buffer ****/
  if (MyRank == Root){
    Output = (ELEMENT_T *) malloc (sizeof (ELEMENT_T) * NoofElements);
    count = 0;
    for(j=0; j<Numprocs; j++){
      k = 1;
      for(i=0; i<OutputBuffer[(2 * NoofElements/Numprocs) * j]; i++) 
        Output[count++] = OutputBuffer[(2*NoofElements/Numprocs) * j + k++];
      }
       printf ( "Number of Elements to be sorted : %d \n", NoofElements);
       ELEMENT_T prev = 0;
       int fail = 0;
       for (i=0; i<NoofElements; i++){
         if(Output[i] < prev) { printf("Failed at index %d\n",i); fail = 1; }
         prev = Output[i];
       }
       if(fail) printf("Sorting FAILED\n");  
       else  printf("Sorting PASSED\n");
       printf("Time for initialization (tInit) = %.3f\n",init_time);
       printf("Time for communicaions (tComm)= %.3f\n",communication_timer); // communication_timer includes init_time
       printf("Time for computations (tComp) = %.3f\n",(end_time - communication_timer));
       printf("Total Time (excluding initalization = tTotal) = %.3f\n",(end_time - init_time));
       free(Output);
       printf("============================ Tabulate Statistics ============================\ntInit\ttComm\ttComp\ttTotal\n%.3f\t%.3f\t%.3f\t%.3f\n",init_time, communication_timer, (end_time - communication_timer), (end_time - init_time));
       printf("=============================================================================\n===== TEST PASSED in %.3f msec =====\n",end_time);
  }/* MyRank==0*/

  shmem_free(Input);
  shmem_free(OutputBuffer);
  shmem_free(InputData);
  shmem_free(Buckets);
  shmem_free(BucketBuffer);
  shmem_free(LocalBucket);

#ifndef HCLIB_COMM_WORKER_FIXED
}

int main (int argc, char ** argv) {
#if defined(_OSHMEM_) && defined(_MPI_)
  printf("ERROR: You cannot use both OpenSHMEM as well as MPI\n");
  exit(1); 
#endif
#if defined(_OSHMEM_)
  shmem_init ();
#ifdef _ASYNC_OSHMEM_
  shmem_workers_init(entrypoint, NULL);
#else
  entrypoint(NULL);
#endif //_ASYNC_OSHMEM_
  shmem_finalize ();
#elif defined(_MPI_)
  MPI_Init(&argc, &argv);
  entrypoint(NULL);
  MPI_Finalize();
#else 
  printf("ERROR: Use either OpenSHMEM or MPI\n");
  exit(1);
#endif
  return 0;
}
#else // HCLIB_COMM_WORKER_FIXED
   /**** Finalize ****/
#if defined(_OSHMEM_)
  shmem_finalize();
#elif defined(_MPI_)
  MPI_Finalize();
#endif
}
#endif
