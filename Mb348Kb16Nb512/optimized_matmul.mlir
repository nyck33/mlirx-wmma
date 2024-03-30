#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 348)>
#map2 = affine_map<(d0) -> (d0 + 512)>
#map3 = affine_map<(d0) -> (d0 + 16)>
module {
  func.func @matmul(%arg0: memref<2088x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<2088x2048xf32>) {
    %alloc = memref.alloc() : memref<2088x2048xf32>
    affine.for %arg3 = 0 to 2088 {
      affine.for %arg4 = 0 to 2048 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<2088x2048xf32>
        affine.store %0, %alloc[%arg3, %arg4] : memref<2088x2048xf32>
      }
    }
    %alloc_0 = memref.alloc() : memref<2048x2048xf32>
    affine.for %arg3 = 0 to 2048 {
      affine.for %arg4 = 0 to 2048 {
        %0 = affine.load %arg1[%arg3, %arg4] : memref<2048x2048xf32>
        affine.store %0, %alloc_0[%arg3, %arg4] : memref<2048x2048xf32>
      }
    }
    %alloc_1 = memref.alloc() : memref<2088x2048xf32>
    affine.for %arg3 = 0 to 2088 {
      affine.for %arg4 = 0 to 2048 {
        %0 = affine.load %arg2[%arg3, %arg4] : memref<2088x2048xf32>
        affine.store %0, %alloc_1[%arg3, %arg4] : memref<2088x2048xf32>
      }
    }
    //#map = affine_map<(d0) -> (d0)>
    //#map1 = affine_map<(d0) -> (d0 + 348)>
    //#map2 = affine_map<(d0) -> (d0 + 512)>
    //#map3 = affine_map<(d0) -> (d0 + 16)>
    //tileA : 348 * 16, tileB : 16 * 512, tileC : 348 * 512
    affine.for %arg3 = 0 to 2088 step 348 {
      affine.for %arg4 = 0 to 2048 step 512 {
        affine.for %arg5 = 0 to 2048 step 16 {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) {//348
            affine.for %arg7 = #map(%arg4) to #map2(%arg4) {//512
              affine.for %arg8 = #map(%arg5) to #map3(%arg5) {//16
                //348, 16
                %0 = affine.load %alloc[%arg6, %arg8] : memref<2088x2048xf32>
                // 16, 512
                %1 = affine.load %alloc_0[%arg8, %arg7] : memref<2048x2048xf32>
                // 348, 512
                %2 = affine.load %alloc_1[%arg6, %arg7] : memref<2088x2048xf32>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.addf %2, %3 : f32
                
                affine.store %4, %alloc_1[%arg6, %arg7] : memref<2088x2048xf32>
              }
            }
          }
        }
      }
    }
    affine.for %arg3 = 0 to 2088 {
      affine.for %arg4 = 0 to 2048 {
        %0 = affine.load %alloc_1[%arg3, %arg4] : memref<2088x2048xf32>
        affine.store %0, %arg2[%arg3, %arg4] : memref<2088x2048xf32>
      }
    }
    memref.dealloc %alloc_1 : memref<2088x2048xf32>
    memref.dealloc %alloc_0 : memref<2048x2048xf32>
    memref.dealloc %alloc : memref<2088x2048xf32>
    return
  }
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c2088 = arith.constant 2088 : index
    %c1 = arith.constant 1 : index
    %c2048 = arith.constant 2048 : index
    %cst = arith.constant 0x4C232000 : f32
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %alloc = memref.alloc() : memref<2088x2048xf32>
    %alloc_0 = memref.alloc() : memref<2048x2048xf32>
    %alloc_1 = memref.alloc() : memref<2088x2048xf32>
    scf.for %arg0 = %c0 to %c2088 step %c1 {
      scf.for %arg1 = %c0 to %c2048 step %c1 {
        memref.store %0, %alloc[%arg0, %arg1] : memref<2088x2048xf32>
      }
    }
    scf.for %arg0 = %c0 to %c2048 step %c1 {
      scf.for %arg1 = %c0 to %c2048 step %c1 {
        memref.store %0, %alloc_0[%arg0, %arg1] : memref<2048x2048xf32>
      }
    }
    scf.for %arg0 = %c0 to %c2088 step %c1 {
      scf.for %arg1 = %c0 to %c2048 step %c1 {
        memref.store %0, %alloc_1[%arg0, %arg1] : memref<2088x2048xf32>
      }
    }
    %1 = call @rtclock() : () -> f32
    affine.for %arg0 = 0 to 5 step 348 {
      affine.for %arg1 = 0 to 5 {
        scf.for %arg2 = %c0 to %c2088 step %c1 {
          scf.for %arg3 = %c0 to %c2048 step %c1 {
            memref.store %0, %alloc_1[%arg2, %arg3] : memref<2088x2048xf32>
          }
        }
        func.call @matmul(%alloc, %alloc_0, %alloc_1) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
      }
    }
    %2 = call @rtclock() : () -> f32
    %3 = arith.subf %2, %1 : f32
    %4 = arith.divf %cst, %3 : f32
    call @printF32(%3) : (f32) -> ()
    call @printNewline() : () -> ()
    call @printFlops(%4) : (f32) -> ()
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

