/*
 *
 * Copyright (c) 2011 - 2015
 *   University of Houston System and UT-Battelle, LLC.
 * Copyright (c) 2009 - 2015
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

/**
 * Sample test case taken from hclib distribution
 */

#include <stdio.h>
#include <sys/utsname.h>

#include <shmem.h>
#include <assert.h>
#include <malloc.h>

#define H1 1024
#define T1 33

//user written code
void forasync_fct1(void * argv,int idx) {
    int *ran=(int *)argv;
    assert(ran[idx] == -1);
    ran[idx] = idx;
}

void init_ran(int *ran, int size) {
    while (size > 0) {
        ran[size-1] = -1;
        size--;
    }
}

#ifndef HCLIB_COMM_WORKER_FIXED
void entrypoint(void *arg) {
    int me, npes;
    struct utsname u;

    uname (&u);
   
    me = shmem_my_pe ();
    npes = shmem_n_pes ();
    printf ("%s: Hello from PE %4d of %4d\n", u.nodename, me, npes);
    printf("hclib workers = %d\n", shmem_n_workers());

    int *ran=(int *)malloc(H1*sizeof(int));
    init_ran(ran, H1);
    int lowBound = 0;
    int highBound = H1;
    int stride = 1;
    int tile_size = T1;
    int loop_dimension = 1;
    shmem_parallel_for_nbi(forasync_fct1, (void*)(ran), NULL, lowBound, highBound, stride, tile_size, loop_dimension, SHMEM_PARALLEL_FOR_RECURSIVE_MODE);

    shmem_barrier_all();
    int i = 0;
    while(i < H1) {
        assert(ran[i] == i);
        i++;
    }
    free(ran);
    if(me == 0) printf("Passed\n");
}

int main (int argc, char ** argv) {
    shmem_init ();
    shmem_workers_init(entrypoint, NULL);
    shmem_finalize ();

    return 0;
}
#else
int main (int argc, char ** argv) {
    shmem_init ();
    shmem_workers_init();

    int me, npes;
    struct utsname u;

    uname (&u);
   
    me = shmem_my_pe ();
    npes = shmem_n_pes ();
    printf ("%s: Hello from PE %4d of %4d\n", u.nodename, me, npes);
    printf("hclib workers = %d\n", shmem_n_workers());

    int *ran=(int *)malloc(H1*sizeof(int));
    init_ran(ran, H1);
    int lowBound = 0;
    int highBound = H1;
    int stride = 1;
    int tile_size = T1;
    int loop_dimension = 1;
    shmem_parallel_for_nbi(forasync_fct1, (void*)(ran), NULL, lowBound, highBound, stride, tile_size, loop_dimension, SHMEM_PARALLEL_FOR_RECURSIVE_MODE);

    shmem_barrier_all();
    int i = 0;
    while(i < H1) {
        assert(ran[i] == i);
        i++;
    }
    free(ran);
    if(me == 0) printf("Passed\n");

    shmem_workers_finalize();
    shmem_finalize ();

    return 0;
}
#endif
