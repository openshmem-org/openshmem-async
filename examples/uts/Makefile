###  UTS - Unbalanced Tree Search  ###

# CONFFILE should be a symlink to your configuration
CONFFILE    = config.in
-include      $(CONFFILE)

DIST_EXCLUDE= DIST_EXCLUDE

COMMON_SRCS = uts.c

TARGETS     = uts-omp-task-shmem 

# ------------------------------------- #
# Set Random Number Generator sources:
# ------------------------------------- #

# Set the default RNG
ifndef RNG
RNG=BRG
endif

ifeq ($(RNG), Devine) 
RNG_SRC = rng/devine_sha1.c
RNG_INCL= rng/devine_sha1.h
RNG_DEF = -DDEVINE_RNG
endif
ifeq ($(RNG), BRG)
RNG_SRC = rng/brg_sha1.c
RNG_INCL= rng/brg_sha1.h
RNG_DEF = -DBRG_RNG
endif
ifeq ($(RNG), ALFG)
RNG_SRC = rng/alfg.c
RNG_INCL= rng/alfg.h
RNG_DEF = -DUTS_ALFG
endif

# ------------------------------------- #
# Targets:
# ------------------------------------- #

.PHONY: clean

all: $(TARGETS)

uts-omp-task-shmem: uts_omp_task_shmem.c $(RNG_SRC) $(COMMON_SRCS)
	$(SHMCC) $(RNG_DEF) $(FLAGS) -o $@ $+ $(OMPCC_OPTS) $(OMPLD_OPTS) $(SHMCC_OPTS) $(SHMLD_OPTS)

uts-async-task-shmem: uts_async_task_shmem.c $(RNG_SRC) $(COMMON_SRCS)
	$(SHMCC) $(RNG_DEF) $(FLAGS) -o $@ $+ $(SHMCC_OPTS) $(SHMLD_OPTS) -lm


clean:
	rm -f *.o $(TARGETS)
