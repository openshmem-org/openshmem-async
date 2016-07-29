#!/bin/bash
#PBS -A csc205
#PBS -N pbbs
#PBS -j oe
#PBS -q debug
#PBS -l walltime=00:13:00,nodes=32
#PBS -V

export SHM_TASK=64
workers=8
export ISX_PE_CHUNKS=32

ITER=6
SHMEM_ASYNC_EXE="./ashmem/isx.weak"
SHMEM_OMP_EXE="./omp/isx.weak"
SHMEM_EXE="../SHMEM/bin/isx.weak"
export PREFIX="verify.ISxWeak.3096.1024"
BENCH_OPTION="33554432 /tmp/del.$$"

#############

cd $PBS_O_WORKDIR
LOGDIR="/ccs/proj/csc205/logs/isx"
export  LD_PRELOAD=/lustre/atlas/sw/tbb/43/sles11.3_gnu4.8.2/source/build/linux_intel64_gcc_cc4.8.2_libc2.11.3_kernel3.0.101_release/libtbbmalloc_proxy.so.2

TOTAL_CORES=`expr $workers \* $SHM_TASK`
runID=0
while [ $runID -lt ${ITER} ]; do
    LOGNAME=`mktemp -p ${LOGDIR}`
    export SMA_SYMMETRIC_SIZE=42949672960
    export GASNET_PHYSMEM_MAX=40G
    # First hclib
    rm ${LOGNAME}
    export HCLIB_WORKERS=${workers}
    LOGFILE="${LOGDIR}/${PREFIX}.SHMEM.hclib-${workers}.cores-${TOTAL_CORES}.log.chunk-${ISX_PE_CHUNKS}"
    aprun -n ${SHM_TASK} -cc 0,1,2,3,4,5,6,7:8,9,10,11,12,13,14,15 -ss -S 1 -d ${workers} ${SHMEM_ASYNC_EXE} ${BENCH_OPTION} >& "${LOGNAME}"
    cat "${LOGNAME}" | sed '/resources: utime/d' > "${LOGNAME}.t1"
    cat "${LOGNAME}.t1" | sed '/WARNING: Running without a provided HCLIB_HPT_FILE/d' >> "${LOGFILE}"
    rm "${LOGNAME}.t1"

    # now OpenMP
    rm ${LOGNAME}
    export OMP_NUM_THREADS=${workers}
    LOGFILE="${LOGDIR}/${PREFIX}.SHMEM.omp-${workers}.cores-${TOTAL_CORES}.log.chunk-${ISX_PE_CHUNKS}"
    aprun -n ${SHM_TASK} -cc 0,1,2,3,4,5,6,7:8,9,10,11,12,13,14,15 -ss -S 1 -d ${workers} ${SHMEM_OMP_EXE} ${BENCH_OPTION} >& "${LOGNAME}"
    cat "${LOGNAME}" | sed '/resources: utime/d' >> "${LOGFILE}"
    unset SMA_SYMMETRIC_SIZE
    unset GASNET_PHYSMEM_MAX

    #now SHMEM
    rm ${LOGNAME}
    LOGFILE="${LOGDIR}/${PREFIX}.SHMEM.cores-${TOTAL_CORES}.log"
    export SMA_SYMMETRIC_SIZE=4294967296
    export GASNET_PHYSMEM_MAX=4G
    aprun -n ${TOTAL_CORES} -ss -d 1 -N 16 -S 8 ${SHMEM_EXE} ${BENCH_OPTION} >& "${LOGNAME}"
    unset SMA_SYMMETRIC_SIZE
    unset GASNET_PHYSMEM_MAX
    cat "${LOGNAME}" | sed '/resources: utime/d' >> "${LOGFILE}"

    runID=`expr ${runID} + 1`
done
