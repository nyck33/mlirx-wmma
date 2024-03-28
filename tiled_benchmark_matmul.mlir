//created with mlir-opt -affine-loop-tile benchmark_matmul.mlir > tiled_benchmark_matmul.mlir
//source benchmark_matmul.mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 4)>
#map2 = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @matmul(%arg0: memref<2088x2048xf64>, %arg1: memref<2048x2048xf64>, %arg2: memref<2088x2048xf64>) {
    //M_B step
    affine.for %arg3 = 0 to 2088 step 4 {
      //N_B step
      affine.for %arg4 = 0 to 2048 step 4 {
        //K_B step
        affine.for %arg5 = 0 to 2048 step 4 {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              affine.for %arg8 = #map(%arg5) to #map1(%arg5) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<2088x2048xf64>
                %1 = affine.load %arg1[%arg8, %arg7] : memref<2048x2048xf64>
                %2 = affine.load %arg2[%arg6, %arg7] : memref<2088x2048xf64>
                %3 = arith.mulf %0, %1 : f64
                %4 = arith.addf %2, %3 : f64
                affine.store %4, %arg2[%arg6, %arg7] : memref<2088x2048xf64>
              }
            }
          }
        }
      }
    }
    return
  }
  func.func @main() {
    %idx5 = index.constant 5
    %alloc = memref.alloc() : memref<2088x2048xf64>
    %alloc_0 = memref.alloc() : memref<2048x2048xf64>
    %alloc_1 = memref.alloc() : memref<2088x2048xf64>
    %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    linalg.fill ins(%0 : f64) outs(%alloc : memref<2088x2048xf64>)
    linalg.fill ins(%0 : f64) outs(%alloc_0 : memref<2048x2048xf64>)
    linalg.fill ins(%0 : f64) outs(%alloc_1 : memref<2088x2048xf64>)
    %1 = call @rtclock() : () -> f64
    affine.for %arg0 = 0 to %idx5 {
      affine.for %arg1 = #map(%arg0) to #map2(%arg0) {
        linalg.fill ins(%0 : f64) outs(%alloc_1 : memref<2088x2048xf64>)
        func.call @matmul(%alloc, %alloc_0, %alloc_1) : (memref<2088x2048xf64>, memref<2048x2048xf64>, memref<2088x2048xf64>) -> ()
      }
    }
    %2 = call @rtclock() : () -> f64
    %idx0 = index.constant 0
    %idx1 = index.constant 1
    %dim = memref.dim %alloc_1, %idx0 : memref<2088x2048xf64>
    %dim_2 = memref.dim %alloc_1, %idx1 : memref<2088x2048xf64>
    %dim_3 = memref.dim %alloc, %idx1 : memref<2088x2048xf64>
    %3 = arith.subf %2, %1 : f64
    %4 = arith.muli %dim, %dim_2 : index
    %5 = arith.muli %dim_3, %dim : index
    %idx2 = index.constant 2
    %6 = arith.muli %idx2, %5 : index
    %7 = arith.muli %idx5, %6 : index
    %8 = arith.index_cast %7 : index to i64
    %9 = arith.sitofp %8 : i64 to f64
    %10 = arith.divf %9, %3 : f64
    call @printF64(%3) : (f64) -> ()
    call @printNewline() : () -> ()
    call @printFlops(%10) : (f64) -> ()
    memref.dealloc %alloc : memref<2088x2048xf64>
    memref.dealloc %alloc_0 : memref<2048x2048xf64>
    memref.dealloc %alloc_1 : memref<2088x2048xf64>
    return
  }
  func.func private @printNewline()
  func.func private @printF64(f64)
  func.func private @printFlops(f64)
  func.func private @rtclock() -> f64
}

