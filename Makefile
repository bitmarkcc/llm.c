CC ?= clang
CXX ?= clang++
CFLAGS = -g3 -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native

# Find nvcc
SHELL_UNAME = $(shell uname)
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@
CUDA_OUTPUT_FILE = -o $@

# NVCC flags
# -t=0 is short for --threads, 0 = number of CPUs on the machine
NVCC_FLAGS = -O3 -t=0 --use_fast_math -std=c++17
NVCC_LDFLAGS = -lcublas -lcublasLt
NVCC_INCLUDES =
NVCC_LDLIBS =
NCLL_INCUDES =
NVCC_CUDNN =
# By default we don't build with cudnn because it blows up compile time from a few seconds to ~minute
USE_CUDNN ?= 0

# We will place .o files in the `build` directory (create it if it doesn't exist)
BUILD_DIR = build
ifeq ($(OS), Windows_NT)
  $(shell if not exist $(BUILD_DIR) mkdir $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := del $(BUILD_DIR)\*.obj
else
  $(shell mkdir -p $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o
endif

# Function to check if a file exists in the PATH
ifneq ($(OS), Windows_NT)
define file_exists_in_path
  $(which $(1) 2>/dev/null)
endef
else
define file_exists_in_path
  $(shell where $(1) 2>nul)
endef
endif

ifneq ($(CI),true) # if not in CI, then use the GPU query
  ifndef GPU_COMPUTE_CAPABILITY # set to defaults if: make GPU_COMPUTE_CAPABILITY=
    ifneq ($(call file_exists_in_path, __nvcc_device_query),)
      GPU_COMPUTE_CAPABILITY = $(shell __nvcc_device_query)
      GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
    endif
  endif
endif

# set to defaults if - make GPU_COMPUTE_CAPABILITY= otherwise use the compute capability detected above
ifneq ($(GPU_COMPUTE_CAPABILITY),)
  NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

# autodect a lot of various supports on current platform
$(info ---------------------------------------------)

ifneq ($(OS), Windows_NT)
  NVCC := $(shell which nvcc 2>/dev/null)
  NVCC_LDFLAGS += -lnvidia-ml

  # Function to test if the compiler accepts a given flag.
  define check_and_add_flag
    $(eval FLAG_SUPPORTED := $(shell printf "int main() { return 0; }\n" | $(CC) $(1) -x c - -o /dev/null 2>/dev/null && echo 'yes'))
    ifeq ($(FLAG_SUPPORTED),yes)
        CFLAGS += $(1)
    endif
  endef

  # Check each flag and add it if supported
  $(foreach flag,$(CFLAGS_COND),$(eval $(call check_and_add_flag,$(flag))))
else
  CFLAGS :=
  REMOVE_FILES = del *.exe,*.obj,*.lib,*.exp,*.pdb && del
  SHELL_UNAME := Windows
  ifneq ($(shell where nvcc 2> nul),"")
    NVCC := nvcc
  else
    NVCC :=
  endif
  CC := cl
  CFLAGS = /Idev /Zi /nologo /W4 /WX- /diagnostics:column /sdl /O2 /Oi /Ot /GL /D _DEBUG /D _CONSOLE /D _UNICODE /D UNICODE /Gm- /EHsc /MD /GS /Gy /fp:fast /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- \
   /external:W3 /Gd /TP /wd4996 /Fd$@.pdb /FC /openmp:llvm
  LDFLAGS :=
  LDLIBS :=
  INCLUDES :=
  NVCC_FLAGS += -I"dev"
  ifeq ($(WIN_CI_BUILD),1)
    $(info Windows CI build)
    OUTPUT_FILE = /link /OUT:$@
    CUDA_OUTPUT_FILE = -o $@
  else
    $(info Windows local build)
    OUTPUT_FILE = /link /OUT:$@ && copy /Y $@ $@.exe
    CUDA_OUTPUT_FILE = -o $@ && copy /Y $@.exe $@
  endif
endif

# Check and include cudnn if available
# You can override the path to cudnn frontend by setting CUDNN_FRONTEND_PATH on the make command line
# By default, we look for it in HOME/cudnn-frontend/include and ./cudnn-frontend/include
# Refer to the README for cuDNN install instructions
ifeq ($(USE_CUDNN), 1)
  ifeq ($(SHELL_UNAME), Linux)
    ifeq ($(shell [ -d $(HOME)/cudnn-frontend/include ] && echo "exists"), exists)
      $(info ✓ cuDNN found, will run with flash-attention)
      CUDNN_FRONTEND_PATH ?= $(HOME)/cudnn-frontend/include
    else ifeq ($(shell [ -d cudnn-frontend/include ] && echo "exists"), exists)
      $(info ✓ cuDNN found, will run with flash-attention)
      CUDNN_FRONTEND_PATH ?= cudnn-frontend/include
    else
      $(error ✗ cuDNN not found. See the README for install instructions and the Makefile for hard-coded paths)
    endif
    NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH)
    NVCC_LDFLAGS += -lcudnn
    NVCC_FLAGS += -DENABLE_CUDNN
    NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.o
  else
    ifneq ($(OS), Windows_NT)
      $(info → cuDNN is not supported on MAC OS right now)
    else
      $(info ✓ Windows cuDNN found, will run with flash-attention)
      ifeq ($(shell if exist "$(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include" (echo exists)),exists)
        CUDNN_FRONTEND_PATH ?= $(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include #override on command line if different location
      else ifeq ($(shell if exist "cudnn-frontend\include" (echo exists)),exists)
        CUDNN_FRONTEND_PATH ?= cudnn-frontend\include #override on command line if different location
      else
        $(error ✗ cuDNN not found. See the README for install instructions and the Makefile for hard-coded paths)
      endif
      CUDNN_INCLUDE_PATH ?= -I"C:\Program Files\NVIDIA\CUDNN\v9.1\include\12.4"
      CUDNN_FRONTEND_PATH += $(CUDNN_INCLUDE_PATH)
      NVCC_FLAGS += --std c++20 -Xcompiler "/std:c++20" -Xcompiler "/EHsc /W0 /nologo /Ox /FS" -maxrregcount=0 --machine 64
      NVCC_CUDNN = $(BUILD_DIR)\cudnn_att.obj
      NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH)
      NVCC_LDFLAGS += -L"C:\Program Files\NVIDIA\CUDNN\v9.1\lib\12.4\x64" -lcudnn
      NVCC_FLAGS += -DENABLE_CUDNN
    endif
  endif
else
  $(info → cuDNN is manually disabled by default, run make with `USE_CUDNN=1` to try to enable)
endif

# Check if OpenMP is available
# This is done by attempting to compile an empty file with OpenMP flags
# OpenMP makes the code a lot faster so I advise installing it
# e.g. on MacOS: brew install libomp
# e.g. on Ubuntu: sudo apt-get install libomp-dev
# later, run the program by prepending the number of threads, e.g.: OMP_NUM_THREADS=8 ./gpt2
# First, check if NO_OMP is set to 1, if not, proceed with the OpenMP checks
ifeq ($(NO_OMP), 1)
  $(info OpenMP is manually disabled)
else
  ifneq ($(OS), Windows_NT)
  # Detect if running on macOS or Linux
    ifeq ($(SHELL_UNAME), Darwin)
      # Check for Homebrew's libomp installation in different common directories
      ifeq ($(shell [ -d /opt/homebrew/opt/libomp/lib ] && echo "exists"), exists)
        # macOS with Homebrew on ARM (Apple Silicon)
        CFLAGS += -Xclang -fopenmp -DOMP
        LDFLAGS += -L/opt/homebrew/opt/libomp/lib
        LDLIBS += -lomp
        INCLUDES += -I/opt/homebrew/opt/libomp/include
        $(info ✓ OpenMP found)
      else ifeq ($(shell [ -d /usr/local/opt/libomp/lib ] && echo "exists"), exists)
        # macOS with Homebrew on Intel
        CFLAGS += -Xclang -fopenmp -DOMP
        LDFLAGS += -L/usr/local/opt/libomp/lib
        LDLIBS += -lomp
        INCLUDES += -I/usr/local/opt/libomp/include
        $(info ✓ OpenMP found)
      else
        $(info ✗ OpenMP not found)
      endif
    else
      # Check for OpenMP support in GCC or Clang on Linux
      ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
        CFLAGS += -fopenmp -DOMP
        LDLIBS += -lgomp
        $(info ✓ OpenMP found)
      else
        $(info ✗ OpenMP not found)
      endif
    endif
  endif
endif

# Check if NCCL is available, include if so, for multi-GPU training
ifeq ($(NO_MULTI_GPU), 1)
  $(info → Multi-GPU (NCCL) is manually disabled)
else
  ifneq ($(OS), Windows_NT)
    # Detect if running on macOS or Linux
    ifeq ($(SHELL_UNAME), Darwin)
      $(info ✗ Multi-GPU on CUDA on Darwin is not supported, skipping NCCL support)
    else ifeq ($(shell dpkg -l | grep -q nccl && echo "exists"), exists)
      $(info ✓ NCCL found, OK to train with multiple GPUs)
      NVCC_FLAGS += -DMULTI_GPU
      NVCC_LDLIBS += -lnccl
    else
      $(info ✗ NCCL is not found, disabling multi-GPU support)
      $(info ---> On Linux you can try install NCCL with `sudo apt install libnccl2 libnccl-dev`)
    endif
  endif
endif

# Attempt to find and include OpenMPI on the system
OPENMPI_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi
OPENMPI_LIB_PATH = $(OPENMPI_DIR)/lib/
OPENMPI_INCLUDE_PATH = $(OPENMPI_DIR)/include/
ifeq ($(NO_USE_MPI), 1)
  $(info → MPI is manually disabled)
else ifeq ($(shell [ -d $(OPENMPI_LIB_PATH) ] && [ -d $(OPENMPI_INCLUDE_PATH) ] && echo "exists"), exists)
  $(info ✓ MPI enabled)
  NVCC_INCLUDES += -I$(OPENMPI_INCLUDE_PATH)
  NVCC_LDFLAGS += -L$(OPENMPI_LIB_PATH)
  NVCC_LDLIBS += -lmpi
  NVCC_FLAGS += -DUSE_MPI
else
  $(info ✗ MPI not found)
endif

# Precision settings, default to bf16 but ability to override
PRECISION ?= BF16
VALID_PRECISIONS := FP32 FP16 BF16
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error Invalid precision $(PRECISION), valid precisions are $(VALID_PRECISIONS))
endif
ifeq ($(PRECISION), FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION), FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

# PHONY means these targets will always be executed
.PHONY: all train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu profile_gpt2cu

# Add targets
TARGETS = train_gpt2 test_gpt2 train_gpt2_btm test2 eval_gpt2_btm eval_gpt2_btm_f32 train_gpt2_btm2 eval_gpt2_btm2_f32 eval_gpt2_btm_f32.wasm

# Conditional inclusion of CUDA targets
ifeq ($(NVCC),)
    $(info ✗ nvcc not found, skipping GPU/CUDA builds)
else
    $(info ✓ nvcc found, including GPU/CUDA support)
    TARGETS += train_gpt2cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu $(NVCC_CUDNN)
endif

$(info ---------------------------------------------)

all: $(TARGETS)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

train_gpt2_btm: train_gpt2_btm.c
	$(CC) $(CFLAGS) $(FP_STRICT) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) -lcrypto $(OUTPUT_FILE)

eval_gpt2_btm: eval_gpt2_btm.c
	$(CXX) $(CFLAGS) -DLLMC_PFLOAT $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) -lmimalloc -lcrypto -lmpfr $(OUTPUT_FILE)

# ----------------------------------------------------------------------------
# deterministic float verifier (native + wasm32-wasi), see doc/btm-proof-of-useful-work.md
# strict IEEE float semantics: -O2 only, never -Ofast/-ffast-math/-march=native --
# determinism is the whole point. The wasm toolchain paths can be overridden, e.g.
#   make eval_gpt2_btm_f32.wasm WASI_CLANG=clang WASI_SYSROOT=/opt/wasi/sysroot
#
# THE INVARIANT: every float operation in these binaries must be a single IEEE-754
# single-precision operation, rounded once. Two things can break it, and only one of
# them was previously guarded.
#
#   -Ofast/-ffast-math/-march=native   reassociate and relax rounding. Excluded above.
#   FMA CONTRACTION                    `a += b * c` is ONE statement, so C lets the
#                                      compiler fuse it into a multiply-add that rounds
#                                      ONCE where the source rounds twice. Different
#                                      result, same source. gcc and clang default to
#                                      -ffp-contract=fast, so this is opt-OUT.
#
# x86-64 was safe only by accident: -O2 with no -march targets the baseline ISA, which
# has no FMA instruction to contract into. Measured on this source:
#
#   gcc   -O2                                      0 FMA instructions
#   gcc   -O2 -march=haswell                       2
#   gcc   -O3 -march=haswell -ffp-contract=off     0
#   clang --target=aarch64 -O2                     1   <-- no special flags at all
#   clang --target=aarch64 -O2 -ffp-contract=off   0
#
# aarch64's BASELINE includes fmadd, so an ARM node built with plain -O2 computes
# different weights than an x86 node -- a fork, with no flag to blame. FP_STRICT is
# therefore mandatory on every target below, native and wasm alike, and must not be
# dropped for any reason. It changes nothing on baseline x86-64 (no FMA was emitted
# there to begin with), so the chain and every prior measurement remain valid.
#
# The MINER (train_gpt2_btm_*) carries FP_STRICT too, but it is NOT sufficient there and
# must not be read as making the miner deterministic. The miner builds with CFLAGS:
#
#   cc -g3 -Ofast ... -march=native -fopenmp -DOMP
#
# -Ofast implies -ffast-math, which on x86 links crtfastmath.o -- and that sets FTZ and
# DAZ in MXCSR for the WHOLE PROCESS. -ffp-contract=off does not undo it. Measured:
#
#   gcc -O2                        MXCSR=0x1f80  FTZ=0  DAZ=0
#   gcc -O2 -ffp-contract=off      MXCSR=0x1f80  FTZ=0  DAZ=0
#   gcc -Ofast                     MXCSR=0x9fc0  FTZ=1  DAZ=1
#   gcc -Ofast -ffp-contract=off   MXCSR=0x9fc0  FTZ=1  DAZ=1   <-- flag does not help
#
# So the miner still flushes denormals where the verifier does not, and -march=native
# additionally makes FMA available on any Haswell-or-later host. This does NOT fork the
# chain: the miner's output is int8 deltas, which the verifier applies to its OWN base
# state with its own arithmetic, so a record stays valid either way. What breaks is the
# miner's PREDICTION of its own score -- it rebuilds the base state with different float
# behaviour, so the improvement it measures is not the improvement the network will
# measure, and near the gate threshold that is the difference between submitting a record
# that passes and one that does not. Lost work, not a fork.
#
# Closing it means dropping -Ofast (and -march=native) for the miner, which costs
# training throughput -- already the binding constraint at ~45 min per 480-step record
# against a 16 min slot. Left as a deliberate open item rather than silently traded away.
FP_STRICT = -ffp-contract=off

WASI_CLANG ?= /usr/lib/llvm/21/bin/clang
WASI_SYSROOT ?= $(HOME)/.local/share/wasi/sysroot
WASI_BUILTINS ?= $(HOME)/.local/share/wasi/lib/wasi/libclang_rt.builtins-wasm32.a

eval_gpt2_btm_f32: eval_gpt2_btm_f32.c
	$(CC) -O2 $(FP_STRICT) $(INCLUDES) $^ -lm $(OUTPUT_FILE)

eval_gpt2_btm_f32.wasm: eval_gpt2_btm_f32.c
	$(WASI_CLANG) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -O2 $(FP_STRICT) $^ -o $@ -nodefaultlibs -lc $(WASI_BUILTINS)

# v2: LoRA structured-delta miner + verifier (llmc/btmrecord.h format)
train_gpt2_btm2: train_gpt2_btm2.c
	$(CC) $(CFLAGS) $(FP_STRICT) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

eval_gpt2_btm2_f32: eval_gpt2_btm2_f32.c
	$(CC) -O2 $(FP_STRICT) $(INCLUDES) $^ -lm $(OUTPUT_FILE)

eval_gpt2_btm2_f32.wasm: eval_gpt2_btm2_f32.c
	$(WASI_CLANG) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -O2 $(FP_STRICT) $^ -o $@ -nodefaultlibs -lc $(WASI_BUILTINS)

# toy MoE: E=4 experts, k=2, depth 6 (llmc/btmmoe.h)
train_gpt2_btm_moe: train_gpt2_btm_moe.c
	$(CC) $(CFLAGS) $(FP_STRICT) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

eval_gpt2_btm_moe_f32: eval_gpt2_btm_moe_f32.c
	$(CC) -O2 $(FP_STRICT) $(INCLUDES) $^ -lm $(OUTPUT_FILE)

eval_gpt2_btm_moe_f32.wasm: eval_gpt2_btm_moe_f32.c
	$(WASI_CLANG) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -O2 $(FP_STRICT) $^ -o $@ -nodefaultlibs -lc $(WASI_BUILTINS)

# trunk: C=768, V=16384, fine experts H=384, k=4, E=32 -- 269M params, 1.004 GiB in f32.
# E=64 (1.849 GiB) also fit the verifier's 1.984 GiB cap, but left only ~81 MiB free;
# E=32 leaves ~1004 MiB, enough to batch ~16 scoring windows and cut the gate's noise by
# 4x. E must match between the miner and the verifier: it sizes the router entry and
# therefore the framing of every record.
train_gpt2_btm_trunk: train_gpt2_btm_trunk.c
	$(CC) $(CFLAGS) $(FP_STRICT) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

eval_gpt2_btm_trunk_f32: eval_gpt2_btm_trunk_f32.c
	$(CC) -O2 $(FP_STRICT) $(INCLUDES) $^ -lm $(OUTPUT_FILE)

eval_gpt2_btm_trunk_f32.wasm: eval_gpt2_btm_trunk_f32.c
	$(WASI_CLANG) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -O2 $(FP_STRICT) $^ -o $@ -nodefaultlibs -lc $(WASI_BUILTINS)

eval_gpt2_btm_trunk_f32_simd.wasm: eval_gpt2_btm_trunk_f32.c
	$(WASI_CLANG) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -O2 -msimd128 $(FP_STRICT) $^ -o $@ -nodefaultlibs -lc $(WASI_BUILTINS)

# The ON-CHAIN module: exports verify(), imports nothing but chain.*.
#
# Differs from the .wasm targets above in two flags, and both are load-bearing:
#   -nostartfiles   omits crt1-command.o. Without it the WASI startup object is linked
#                   even with --no-entry, and it drags in __main_void -> args_get,
#                   _Exit -> proc_exit and dlmalloc -> sbrk -> abort. That alone is the
#                   difference between 3 WASI imports and 0.
#   --no-entry      no _start; the node calls the exported verify() directly.
# -lc stays: static linking is per OBJECT, so with no stdio/file/env/time calls in the
# source the only members pulled in are the math ones (expf, logf, sinf, cosf, tanhf),
# which are pure computation and import nothing. Keeping them IN the module is also what
# makes them deterministic across nodes.
#
# Verify after building, because a stray import is silent until the node refuses to
# instantiate:
#   strings -a eval_gpt2_btm_trunk_f32_algo.wasm | grep -c __imported_wasi   # must be 0
# -msimd128 is NOT optional here. The gas limits are per opcode CLASS: a scalar build
# charges each multiply-accumulate to GC_FADD, whose cap is 512M, and one forward over 63
# tokens is ~4.38e9 of them -- an order of magnitude over. The SIMD128 path charges
# GC_SIMD instead (cap 5e9) at four lanes per op. The kernel is written so both produce
# bit-identical results (see btmk_dot), so this is a gas decision, not a numerics one.
eval_gpt2_btm_trunk_f32_algo.wasm: eval_gpt2_btm_trunk_f32.c
	$(WASI_CLANG) --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) -O2 -msimd128 $(FP_STRICT) -DBTM_ONCHAIN $^ -o $@ \
	    -nodefaultlibs -nostartfiles -lc $(WASI_BUILTINS) \
	    -Wl,--no-entry -Wl,--export=verify
	@n=$$(strings -a $@ | grep -c __imported_wasi || true); \
	 if [ "$$n" != "0" ]; then \
	   echo "ERROR: $@ has $$n WASI imports; the node provides none and will refuse it:"; \
	   strings -a $@ | grep __imported_wasi | sed 's/.*preview1_/  /' | sort -u; \
	   rm -f $@; exit 1; \
	 fi; \
	 echo "$@: 0 WASI imports; chain imports:"; \
	 strings -a $@ | grep -E '^slot_block' | sort -u | sed 's/^/  chain./' 

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

test2: test2.c
	$(CXX) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) -lmpfr $(OUTPUT_FILE)

$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_INCLUDES) -o $@

train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2cu: test_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2fp32cu: test_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

profile_gpt2cu: profile_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -lineinfo $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS)  $(CUDA_OUTPUT_FILE)

clean:
	$(REMOVE_FILES) $(TARGETS)
	$(REMOVE_BUILD_OBJECT_FILES)
