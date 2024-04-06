//cgeist gemm.cu -O3 -function=* --resource-dir=$LLVM_BUILD_DIR/lib/clang/18 --cuda-gpu-arch=sm_75 -S -raise-scf-to-affine --cuda-path=/usr/local/cuda-11.8 -emit-cuda
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z6matmulPfS_S__kernel94332987900336(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel attributes {gpu.known_block_size = array<i32: 16, 16, 1>, gpu.known_grid_size = array<i32: 19, 13, 1>, nvvm.maxntidx = 256 : index, rocdl.max_flat_work_group_size = 256 : index} {
      %c300 = arith.constant 300 : index
      %c1 = arith.constant 1 : index
      %c400 = arith.constant 400 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c300_i32 = arith.constant 300 : i32
      %c200_i32 = arith.constant 200 : i32
      %0 = gpu.block_id  y
      %1 = arith.index_cast %0 : index to i32
      %2 = gpu.block_dim  y
      %3 = arith.index_cast %2 : index to i32
      %4 = arith.muli %1, %3 : i32
      %5 = gpu.thread_id  y
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.addi %4, %6 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = gpu.block_id  x
      %10 = arith.index_cast %9 : index to i32
      %11 = gpu.block_dim  x
      %12 = arith.index_cast %11 : index to i32
      %13 = arith.muli %10, %12 : i32
      %14 = gpu.thread_id  x
      %15 = arith.index_cast %14 : index to i32
      %16 = arith.addi %13, %15 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.cmpi slt, %7, %c200_i32 : i32
      %19 = arith.cmpi slt, %16, %c300_i32 : i32
      %20 = arith.andi %18, %19 : i1
      scf.if %20 {
        %21 = arith.muli %8, %c400 : index
        %22 = scf.for %arg3 = %c0 to %c400 step %c1 iter_args(%arg4 = %cst) -> (f32) {
          %25 = arith.addi %arg3, %21 : index
          %26 = memref.load %arg0[%25] : memref<?xf32>
          %27 = arith.muli %arg3, %c300 : index
          %28 = arith.addi %27, %17 : index
          %29 = memref.load %arg1[%28] : memref<?xf32>
          %30 = arith.mulf %26, %29 : f32
          %31 = arith.addf %arg4, %30 : f32
          scf.yield %31 : f32
        }
        %23 = arith.muli %8, %c300 : index
        %24 = arith.addi %23, %17 : index
        memref.store %22, %arg2[%24] : memref<?xf32>
      }
      gpu.return
    }
  }
  func.func @_Z6matmulPfS_S_(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c13 = arith.constant 13 : index
    %c19 = arith.constant 19 : index
    %c320000_i64 = arith.constant 320000 : i64
    %c480000_i64 = arith.constant 480000 : i64
    %c240000_i64 = arith.constant 240000 : i64
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<1xmemref<?xf32>>
    %alloca_0 = memref.alloca() : memref<1xmemref<?xf32>>
    %alloca_1 = memref.alloca() : memref<1xmemref<?xf32>>
    %0 = "polygeist.memref2pointer"(%alloca_1) : (memref<1xmemref<?xf32>>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
    %2 = call @cudaMalloc(%1, %c320000_i64) : (memref<?xmemref<?xi8>>, i64) -> i32
    %3 = "polygeist.memref2pointer"(%alloca_0) : (memref<1xmemref<?xf32>>) -> !llvm.ptr
    %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
    %5 = call @cudaMalloc(%4, %c480000_i64) : (memref<?xmemref<?xi8>>, i64) -> i32
    %6 = "polygeist.memref2pointer"(%alloca) : (memref<1xmemref<?xf32>>) -> !llvm.ptr
    %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
    %8 = call @cudaMalloc(%7, %c240000_i64) : (memref<?xmemref<?xi8>>, i64) -> i32
    %9 = memref.load %alloca_1[%c0] : memref<1xmemref<?xf32>>
    %10 = "polygeist.memref2pointer"(%9) : (memref<?xf32>) -> !llvm.ptr
    %11 = "polygeist.pointer2memref"(%10) : (!llvm.ptr) -> memref<?xi8>
    %12 = "polygeist.memref2pointer"(%arg0) : (memref<?xf32>) -> !llvm.ptr
    %13 = "polygeist.pointer2memref"(%12) : (!llvm.ptr) -> memref<?xi8>
    %14 = call @cudaMemcpy(%11, %13, %c320000_i64, %c1_i32) : (memref<?xi8>, memref<?xi8>, i64, i32) -> i32
    %15 = memref.load %alloca_0[%c0] : memref<1xmemref<?xf32>>
    %16 = "polygeist.memref2pointer"(%15) : (memref<?xf32>) -> !llvm.ptr
    %17 = "polygeist.pointer2memref"(%16) : (!llvm.ptr) -> memref<?xi8>
    %18 = "polygeist.memref2pointer"(%arg1) : (memref<?xf32>) -> !llvm.ptr
    %19 = "polygeist.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi8>
    %20 = call @cudaMemcpy(%17, %19, %c480000_i64, %c1_i32) : (memref<?xi8>, memref<?xi8>, i64, i32) -> i32
    %21 = memref.load %alloca_1[%c0] : memref<1xmemref<?xf32>>
    %22 = memref.load %alloca_0[%c0] : memref<1xmemref<?xf32>>
    %23 = memref.load %alloca[%c0] : memref<1xmemref<?xf32>>
    gpu.launch_func  @__polygeist_gpu_module::@_Z6matmulPfS_S__kernel94332987900336 blocks in (%c19, %c13, %c1) threads in (%c16, %c16, %c1)  args(%21 : memref<?xf32>, %22 : memref<?xf32>, %23 : memref<?xf32>)
    %24 = "polygeist.memref2pointer"(%arg2) : (memref<?xf32>) -> !llvm.ptr
    %25 = "polygeist.pointer2memref"(%24) : (!llvm.ptr) -> memref<?xi8>
    %26 = memref.load %alloca[%c0] : memref<1xmemref<?xf32>>
    %27 = "polygeist.memref2pointer"(%26) : (memref<?xf32>) -> !llvm.ptr
    %28 = "polygeist.pointer2memref"(%27) : (!llvm.ptr) -> memref<?xi8>
    %29 = call @cudaMemcpy(%25, %28, %c240000_i64, %c2_i32) : (memref<?xi8>, memref<?xi8>, i64, i32) -> i32
    %30 = memref.load %alloca_1[%c0] : memref<1xmemref<?xf32>>
    %31 = "polygeist.memref2pointer"(%30) : (memref<?xf32>) -> !llvm.ptr
    %32 = "polygeist.pointer2memref"(%31) : (!llvm.ptr) -> memref<?xi8>
    %33 = call @cudaFree(%32) : (memref<?xi8>) -> i32
    %34 = memref.load %alloca_0[%c0] : memref<1xmemref<?xf32>>
    %35 = "polygeist.memref2pointer"(%34) : (memref<?xf32>) -> !llvm.ptr
    %36 = "polygeist.pointer2memref"(%35) : (!llvm.ptr) -> memref<?xi8>
    %37 = call @cudaFree(%36) : (memref<?xi8>) -> i32
    %38 = memref.load %alloca[%c0] : memref<1xmemref<?xf32>>
    %39 = "polygeist.memref2pointer"(%38) : (memref<?xf32>) -> !llvm.ptr
    %40 = "polygeist.pointer2memref"(%39) : (!llvm.ptr) -> memref<?xi8>
    %41 = call @cudaFree(%40) : (memref<?xi8>) -> i32
    return
  }
  func.func private @cudaMalloc(memref<?xmemref<?xi8>>, i64) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @cudaMemcpy(memref<?xi8>, memref<?xi8>, i64, i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @cudaFree(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}