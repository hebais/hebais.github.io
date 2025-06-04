+++
date = '2025-05-28T11:10:51+08:00'
draft = false
title = 'TritonAMD'
+++


This is a note from: 

[kernel-development-optimizations-with-triton-on-amd](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)

# Modules

1. Frontend module of AMD Triton Compiler
2. Optimizer module of AMD Triton Compiler
3. Machine Code Generation


## Frontend

TODO

## Optimizer

Optimization Passes:
1. MLIR general optimization passes, like CSE, DCE, Inlining, etc.
2. GPU specific optimization passes, like Pipeline, Prefetch, Matmul accelerate, Coalesce, etc.
3. Vender specific GPU optimization passes, for example Nvidia provides TMA, Async Dot, etc. AMD provides OptimizeLDSUsage, BlockPingpong, etc.

### ttir (Triton IR)
`make_ttir_funtion()`
hardware independent
Passes: inline optimization/CSE/Canonicalization/DCE/Loop Invariant Code Motion/Loop Unroll

### ttgir (Triton GPU IR)
`make_ttgir()`
hardware dependent
designed for GPU platform to boost its performance
Passes: AMD GPU Matmul/Block PingPong/Pipeline etc

### llir (LLVM-IR)
`make_llir()`
IR-level optimization and AMD GPU specific optimization

## Machine Code Generation
AMDGCN assembly generation (`make_amdgcn()`) and AMD hsaco ELF file generation.


# Instrution

PTX: 并行线程执行 (Parallel Thread eXecution) ISA
[ptx](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/parallel-thread-execution/index.html)

SASS: Streaming ASSembler
[cuda binary utils](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/cuda-binary-utilities/index.html)

# AMD Architecture

Southern Island: Graphics Core Next (GCN) 架构


# CUDA GPU
[compute capability](https://developer.nvidia.cn/cuda-gpus#compute)