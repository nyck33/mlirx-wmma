#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 464, 2088)>
#map2 = affine_map<(d0) -> (d0 + 512)>
#map3 = affine_map<(d0) -> (d0 + 16)>
#map4 = affine_map<(d0) -> (d0 + 5)>
module {
  func.func @matmul(%arg0: memref<2088x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<2088x2048xf32>) {
    affine.for %arg3 = 0 to 2088 step 464 {
      affine.for %arg4 = 0 to 2048 step 512 {
        affine.for %arg5 = 0 to 2048 step 16 {
          affine.for %arg6 = #map(%arg3) to min #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to #map2(%arg4) {
              affine.for %arg8 = #map(%arg5) to #map3(%arg5) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<2088x2048xf32>
                %1 = affine.load %arg1[%arg8, %arg7] : memref<2048x2048xf32>
                %2 = affine.load %arg2[%arg6, %arg7] : memref<2088x2048xf32>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.addf %2, %3 : f32
                affine.store %4, %arg2[%arg6, %arg7] : memref<2088x2048xf32>
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
    %alloc = memref.alloc() : memref<2088x2048xf32>
    %alloc_0 = memref.alloc() : memref<2048x2048xf32>
    %alloc_1 = memref.alloc() : memref<2088x2048xf32>
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    linalg.fill ins(%0 : f32) outs(%alloc : memref<2088x2048xf32>)
    linalg.fill ins(%0 : f32) outs(%alloc_0 : memref<2048x2048xf32>)
    linalg.fill ins(%0 : f32) outs(%alloc_1 : memref<2088x2048xf32>)
    %1 = call @rtclock() : () -> f32
    affine.for %arg0 = 0 to %idx5 step 464 {
      affine.for %arg1 = #map(%arg0) to #map4(%arg0) {
        linalg.fill ins(%0 : f32) outs(%alloc_1 : memref<2088x2048xf32>)
        func.call @matmul(%alloc, %alloc_0, %alloc_1) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
      }
    }
    %2 = call @rtclock() : () -> f32
    %idx0 = index.constant 0
    %idx1 = index.constant 1
    %dim = memref.dim %alloc_1, %idx0 : memref<2088x2048xf32>
    %dim_2 = memref.dim %alloc_1, %idx1 : memref<2088x2048xf32>
    %dim_3 = memref.dim %alloc, %idx1 : memref<2088x2048xf32>
    %3 = arith.subf %2, %1 : f32
    %4 = arith.muli %dim, %dim_2 : index
    %5 = arith.muli %dim_3, %dim : index
    %idx2 = index.constant 2
    %6 = arith.muli %idx2, %5 : index
    %7 = arith.muli %idx5, %6 : index
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.divf %9, %3 : f32
    call @printF32(%3) : (f32) -> ()
    call @printNewline() : () -> ()
    call @printFlops(%10) : (f32) -> ()
    memref.dealloc %alloc : memref<2088x2048xf32>
    memref.dealloc %alloc_0 : memref<2048x2048xf32>
    memref.dealloc %alloc_1 : memref<2088x2048xf32>
    return
  }
  func.func private @printNewline()
  func.func private @printF32(f32)
  func.func private @printFlops(f32)
  func.func private @rtclock() -> f32
}

