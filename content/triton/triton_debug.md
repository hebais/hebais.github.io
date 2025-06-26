+++
date = '2025-06-25T11:07:29+08:00'
draft = true
title = 'Triton Debug'
+++

# example

```python
import triton
import torch
import triton.language as tl
from triton.backends.compiler import GPUTarget
import os

# Set environment variables for IR dumping
os.environ["TRITON_DUMP_IR"] = "1"
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/thuang/.cache/triton"

@triton.jit
def add_kernel(x_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    off_x = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off_x < n_elements
    x = tl.load(x_ptr + off_x, mask)
    tl.store(z_ptr + off_x, x + 10.0, mask)
    return

def add(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

if __name__ == "__main__":
    # Compile the kernel to generate IR
    src = triton.compiler.ASTSource(
        fn=add_kernel,
        signature={"0": "*fp32", "1": "*fp32", "2": "i32", "3": "i32"},
        constants={"BLOCK_SIZE": 64}
    )
    
    # Use CUDA target with compute capability 80
    target = GPUTarget("cuda", 80, 32)
    
    try:
        output = triton.compile(src, target=target)
        print("Kernel compiled successfully!")
        print("Generated IR should be available in the cache directory")
    except Exception as e:
        print(f"Compilation failed: {e}")
        print("This might be due to missing CUDA libraries or PyTorch")
```

# triton-opt

```bash
triton-opt add_kernel.ttir -convert-triton-to-tritongpu='target=cuda:80'

# print axis analysis
triton-opt add.ttgir -test-print-alignment

# process with coalesce
triton-opt add.ttgir -tritongpu-coalesce

# lower to llir
triton-translate add-opt.ttgir -target=llvmir
```

# ttgir
```bash
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#loc = loc("/home/thuang/github/hebais.github.io/content/triton/kernel_add.py":15:0)
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> loc("/home/thuang/github/hebais.github.io/content/triton/kernel_add.py":15:0), %arg1: !tt.ptr<f32> loc("/home/thuang/github/hebais.github.io/content/triton/kernel_add.py":15:0), %arg2: i32 loc("/home/thuang/github/hebais.github.io/content/triton/kernel_add.py":15:0), %arg3: i32 loc("/home/thuang/github/hebais.github.io/content/triton/kernel_add.py":15:0)) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+01> : tensor<64xf32, #blocked> loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c64_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<64xi32, #blocked> loc(#loc5)
    %4 = arith.addi %3, %2 : tensor<64xi32, #blocked> loc(#loc5)
    %5 = tt.splat %arg2 : i32 -> tensor<64xi32, #blocked> loc(#loc6)
    %6 = arith.cmpi slt, %4, %5 : tensor<64xi32, #blocked> loc(#loc6)
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked> loc(#loc7)
    # 计算该op需要访问的地址
    %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked> loc(#loc7)
    %9 = tt.load %8, %6 : tensor<64x!tt.ptr<f32>, #blocked> loc(#loc8)
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked> loc(#loc9)
    %11 = tt.addptr %10, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked> loc(#loc9)
    %12 = arith.addf %9, %cst : tensor<64xf32, #blocked> loc(#loc10)
    tt.store %11, %12, %6 : tensor<64x!tt.ptr<f32>, #blocked> loc(#loc11)
    tt.return loc(#loc12)
  } loc(#loc)
} loc(#loc)
```