//block comment below
// mlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -reconcile-unrealized-casts --convert-to-llvm matmul.mlir | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=/mnt/d/LLVM/Polygeist/llvm-project/build/lib/libmlir_c_runner_utils.so


func.func @matmul(%A: memref<2088x2048xf32>, %B: memref<2048x2048xf32>, %C: memref<2088x2048xf32>) {
  affine.for %arg3 = 0 to 2088 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        //i,k
        %a = affine.load %A[%arg3, %arg5] : memref<2088x2048xf32>
        //k, j
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf32>
        //i,j
        %ci = affine.load %C[%arg3, %arg4] : memref<2088x2048xf32>
        %p = arith.mulf %a, %b : f32
        %co = arith.addf %ci, %p : f32
        //store the result in C[i,j]
        affine.store %co, %C[%arg3, %arg4] : memref<2088x2048xf32>
      }
    }
  }
  return
}

func.func @main(){
    %reps = index.constant 5

    %A = memref.alloc() : memref<2088x2048xf32>
    %B = memref.alloc() : memref<2048x2048xf32>
    %C = memref.alloc() : memref<2088x2048xf32>
    %cf1 = llvm.mlir.constant(1.00000e+00 : f32) : f32

    //initialize matrices A and B, C with constant 1
    linalg.fill ins (%cf1 : f32) outs(%A : memref<2088x2048xf32>)
    linalg.fill ins (%cf1: f32) outs(%B: memref<2048x2048xf32>)
    linalg.fill ins (%cf1: f32) outs(%C: memref<2088x2048xf32>)

    %t_start = call @rtclock() : () -> (f32)
    affine.for %ti = 0 to %reps {
      linalg.fill ins (%cf1 : f32) outs(%C : memref<2088x2048xf32>)
      func.call @matmul(%A, %B, %C) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
    }
    %t_end = call @rtclock() : () -> (f32)
    //constant 0 for rows, 1` for columns in calculating the dimensions of matrices M*K, K*N, M*N
    %c0 = index.constant 0
    %c1 = index.constant 1
    //rows of C
    %M = memref.dim %C, %c0 : memref<2088x2048xf32>
    //columns of C
    %N = memref.dim %C, %c1 : memref<2088x2048xf32>
    //cols of A
    %K = memref.dim %A, %c1 : memref<2088x2048xf32>
    //time taken
    %t = arith.subf %t_end, %t_start : f32
    %f1 = arith.muli %M, %N : index
    %f2 = arith.muli %K, %M : index
    // 2*M*N*K
    %c2 = index.constant 2
    %f3 = arith.muli %c2, %f2 : index
    %num_flops = arith.muli %reps, %f3 : index
    %num_flops_i = arith.index_cast %num_flops : index to i32
    %num_flops_f = arith.sitofp %num_flops_i : i32 to f32
    %flops = arith.divf %num_flops_f, %t : f32
    //print the time taken
    call @printF32(%t) : (f32) -> ()
    call @printNewline() : () -> ()
    //print the number of flops
    call @printFlops(%flops) : (f32) -> ()

    memref.dealloc %A : memref<2088x2048xf32>
    memref.dealloc %B : memref<2048x2048xf32>
    memref.dealloc %C : memref<2088x2048xf32>

    return

}

func.func private @printNewline() -> ()
func.func private @printF32(f32) -> ()
func.func private @printFlops(f32) -> ()
func.func private @rtclock() -> f32
