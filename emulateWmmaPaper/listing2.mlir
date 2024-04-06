#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 64)>
#map2 = affine_map<(d0) -> (d0 + 128)>
module {
    // Shared memory buffers for A and B.
    memref.global "private" @b_smem_global : memref<64x136xf16, 3>
    memref.global "private" @a_smem_global : memref<128x72xf16, 3>
    func @main() {

         // Allocate memory for A, B on host using half precision
        %hA = memref.alloc() : memref<8192x8192xf16>
        %hB = memref.alloc() : memref<8192x8192xf16>
        // Allocate memory for output matrix C on host using single precision
        %hC = memref.alloc() : memref<8192x8192xf32>

        // Define constants used in the program
        %f1 = arith.constant 1.0e+00 : f16 // Constant value 1.0 of type half-precision float
        %f0 = arith.constant 0.0e+00 : f32 // Constant value 0.0 of type single-precision float

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index

        %c8192 = arith.constant 8192 : index

        //initialize the input matrices with ones
        scf.for %arg0 = %c0 to %c8192 step %c1 {
            scf.for %arg1 = %c0 to %c8192 step %c1 {
               memref.store %f1, %hA[%arg0, %arg1] : memref<8192x8192xf16>
            }
        }

        //now initialize hB with ones
        scf.for %arg0 = %c0 to %c8192 step %c1 {
            scf.for %arg1 = %c0 to %c8192 step %c1 {
               memref.store %f1, %hB[%arg0, %arg1] : memref<8192x8192xf16>
            }
        }

        //now initialize hC with zeros
        scf.for %arg0 = %c0 to %c8192 step %c1 {
            scf.for %arg1 = %c0 to %c8192 step %c1 {
               memref.store %f0, %hC[%arg0, %arg1] : memref<8192x8192xf32>
            }
        }
        // Asynchronous operations token
        %token = gpu.wait async

        // Allocate device memory for matrices A, B, and C asynchronously
        %A, %tokenA = gpu.alloc async [%token] () : memref<8192x8192xf16>
        %B, %tokenB = gpu.alloc async [%token] () : memref<8192x8192xf16>
        %C, %tokenC = gpu.alloc async [%token] () : memref<8192x8192xf32>

        // Copy A and B from host to device asynchronously
        %copyA = gpu.memcpy async [%token] %A, %hA : memref<8192x8192xf16>, memref<8192x8192xf16>
        %copyB = gpu.memcpy async [%token] %B, %hB : memref<8192x8192xf16>, memref<8192x8192xf16>
        // Copy C from host to device asynchronously (if initialization on host is needed)
        %copyC = gpu.memcpy async [%token] %C, %hC : memref<8192x8192xf32>, memref<8192x8192xf32>

        //define the gpu.launch blocks for the kernel

        affine.for %i = 0 to 8192 step 128 {
            affine.for %j = 0 to 8192 step 128 {
                // References to shared memory buffers.
                %b_smem = memref.get_global @b_smem_global : memref<64x136xf16, 3>
                %a_smem = memref.get_global @a_smem_global : memref<128x72xf16, 3>

                affine.for %k = 0 to 8192 step 64 {
                    // Copy loop for B.
                    affine.for %copykk = #map0(%k) to #map1(%k) {
                        affine.for %copyjj = #map0(%j) to #map2(%j) {
                            %11 = affine.load %B[%copykk, %copyjj] : memref<8192x8192xf16>
                            affine.store %11, %b_smem[%copykk - %k, %copyjj - %j] : memref<64x136xf16, 3>
                        }
                    }
                    // Copy loop for A.
                    affine.for %copyii = #map0(%i) to #map2(%i) {
                        affine.for %copykk = #map0(%k) to #map1(%k) {
                            %11 = affine.load %A[%copyii, %copykk] : memref<8192x8192xf16>
                            affine.store %11, %a_smem[%copyii - %i, %copykk - %k] : memref<128x72xf16, 3>
                        }
                    }
                    affine.for %ii = 0 to 128 step 64 {
                        affine.for %jj = 0 to 128 step 32 {
                            affine.for %kk = 0 to 64 step 32 {
                                affine.for %kkk = 0 to 32 step 16 {
                                    affine.for %iii = 0 to 64 step 16 {
                                        affine.for %jjj = 0 to 32 step 16 {
                                            ...
                                            %a = gpu.subgroup_mma_load_matrix %a_smem[%11, %12] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                                            %b = gpu.subgroup_mma_load_matrix %b_smem[%12, %14] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                                            %c = gpu.subgroup_mma_load_matrix %C[%16, %17] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                                            %res = gpu.subgroup_mma_compute %a, %b, %c : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                                            gpu.subgroup_mma_store_matrix %res, %C[%16, %17] {leadDimension = 8192 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>

                                             // Print success message to indicate successful execution
                                            gpu.printf "Success\n"
                                            // CHECK: Success
                                            gpu.terminator
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Deallocate device memory for matrices A, B, and C asynchronously
        %zA = gpu.dealloc async [%token] %A_device : memref<8192x8192xf16>
        %zB = gpu.dealloc async [%token] %B_device : memref<8192x8192xf16>
        %zC = gpu.dealloc async [%token] %C_device : memref<8192x8192xf32>

        // Wait for all asynchronous operations to complete
        gpu.wait [%token]
        return
    }
}