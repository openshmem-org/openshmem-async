
----------------------------------------
ISx is Scalable Integer Sort Application 
----------------------------------------

* ISx is a new scalable integer sort application designed for co-design 
  in the exascale era, scalable to large numbers of nodes.

* ISx belongs to the class of bucket sort algorithms which perform an 
  all-to-all communication pattern.

* ISx is inspired by the NAS Parallel Benchmark Integer Sort and its OpenSHMEM
  implementation of University of Houston. ISx addresses identified shortcomings 
  of the NPB IS.

* ISx is a highly modular application implemented in the OpenSHMEM parallel 
  programming model and supports both strong and weak scaling studies.

* ISx uses an uniform random key distribution and guarantees load balance.  

* ISx includes a verification stage.

* ISx is not a benchmark. It does not define fixed problems that can be used 
  to rank systems. Furthermore ISx has not been optimzed for the features 
  of any particular system.

* ISx has been presented at the PGAS 2015 conference 


References:
ISx, a Scalable Integer Sort for Co-design in the Exascale Era. 
Ulf Hanebutte and Jacob Hemstad. Proc. Ninth Conf. on Partitioned Global Address Space 
Programming Models (PGAS). Washington, DC. Sep 2015. http://hpcl.seas.gwu.edu/pgas15/
http://ieeexplore.ieee.org/xpl/mostRecentIssue.jsp?punumber=7306013

Information about the NAS Parallel Benchmarks may be found here:
https://www.nas.nasa.gov/publications/npb.html

The OpenSHMEM NAS Parallel Benchmarks 1.0a by the HPCTools Group University of Houston
can be downloaded at http://www.openshmem.org/site/Downloads/Examples


STRONG SCALING (isx.strong): Total number of keys are fixed and the number of keys per PE
are reduced with increasing number of PEs
 Invariants: Total number of keys, max key value
 Variable:   Number of keys per PE, Bucket width

WEAK SCALING (isx.weak): The number of keys per PE is fixed and the total number of keys
grow with increasing number of PEs
 Invariants: Number of keys per PE, max key value
 Variable:   Total Number of Keys, Bucket width 

WEAK_ISOBUCKET (isx.weak_iso): Same as WEAK except the maximum key value grows with the 
number of PEs to keep bucket width constant This option is provided in effort to 
keep the amount of time spent in the local sort per PE constant. Without this option,
the local sort time reduces with growing numbers of PEs due to a shrinking histogram 
improving cache performance.
 Invariants: Number of keys per PE, bucket width
 Variable:   Total number of keys, max key value


-------------------------------------------
Compiling and Executing the ISx Application
-------------------------------------------

Compilation options:

make
- Compiles with basic flags

make debug
- Compiles with all debug flags, including -DDEBUG which enables verbose debugging print statements.

make optimized
- Compiles with optimization flags, including -DNDEBUG, which disables assert statements.


The params.h file has various definitions that may be modified to change application options.

Usage: ./bin/isx.strong <total_num_keys>  <log_file>
       ./bin/isx.weak <keys_per_pe> <log_file>
       ./bin/isx.weak_iso <keys_per_pe> <log_file>

The log file stores the verbose timing information for the run. Each row of the file corresponds 
to a single PE's timing results for each component of the application, for every iteration of computation. 
Reuse of the same output file will concatenate results. 

example command lines (assuming aprun) for 2^27 keys (=134217728)

Strong:
 aprun -n 24 -N 4 ./bin/isx.strong 134217728 output_strong
 
Weak:
 aprun -n 24 -N 4 ./bin/isx.weak 134217728 output_weak
 
Weak_iso:
 Note that the iso-bucket width is specified in params.h
 aprun -n 24 -N 4 ./bin/isx.weak_iso 134217728 output_weak_iso

Note: timing measurements (see timer.c) are obtained by calls to clock_gettime
with the clk_id argument set to CLOCK_MONOTONIC. However, not all systems support this clk_id.
For such situations, clk_id should be changed to CLOCK_REALTIME, which is supported by all systems.
