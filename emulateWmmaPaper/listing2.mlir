#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 64)>
#map2 = affine_map<(d0) -> (d0 + 128)>
module {
    // Shared memory buffers for A and B.
    memref.global "private" @b_smem_global : memref<64x136xf16, 3>
    memref.global "private" @a_smem_global : memref<128x72xf16, 3>
    func.func @main() {
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

        //define block and grid xyz where each block is 16 x 16 threads
        //grid is 8 x 8 so that covers the 128 * 128 tiles of C
        %blockX = arith.constant 128 : index
        %blockY = arith.constant 128 : index
        %blockZ = arith.constant 1 : index
        %gridX = arith.constant 8 : index
        %gridY = arith.constant 8 : index
        %gridZ = arith.constant 1 : index

        affine.for %i = 0 to 8192 step 128 { //128 rows of tile C/A
            affine.for %j = 0 to 8192 step 128 {//128 cols of tile C/B
                // References to shared memory buffers.
                %b_smem = memref.get_global @b_smem_global : memref<64x136xf16, 3>
                %a_smem = memref.get_global @a_smem_global : memref<128x72xf16, 3>
                //main k-loop
                affine.for %k = 0 to 8192 step 64 {  //64 is cols A tile and rows B tile 
                    // Copy loop for B.
                    affine.for %copykk = #map0(%k) to #map1(%k) {//k to k + 64
                        affine.for %copyjj = #map0(%j) to #map2(%j) {//j to j + 128
                            %11 = affine.load %B[%copykk, %copyjj] : memref<8192x8192xf16>
                            affine.store %11, %b_smem[%copykk - %k, %copyjj - %j] : memref<64x136xf16, 3>
                        }
                    }
                    // Copy loop for A. done in this order to preserve %11 to use it to index the shared memory
                    affine.for %copyii = #map0(%i) to #map2(%i) {//i to i + 128
                        affine.for %copykk = #map0(%k) to #map1(%k) {//k to k + 64
                            %11 = affine.load %A[%copyii, %copykk] : memref<8192x8192xf16>
                            //what is the value of %11 here?  It is the value of A at %copyii, %copykk
                            //copyii - i to index into the shared memory of size 128 * 72, the padded section remains as 
                            affine.store %11, %a_smem[%copyii - %i, %copykk - %k] : memref<128x72xf16, 3>
                        }
                    }
                    //copied so iterate
                    affine.for %ii = 0 to 128 step 64 {//rows A tile
                        affine.for %jj = 0 to 128 step 32 {//cols B tile
                            affine.for %kk = 0 to 64 step 32 {//cols A tile
                                affine.for %kkk = 0 to 32 step 16 {//2 steps
                                    affine.for %iii = 0 to 64 step 16 {//4 steps
                                        affine.for %jjj = 0 to 32 step 16 {//2 steps
                                            //define the gpu.launch blocks for the kernel 
                                            gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gridX, %grid_y = %gridY, %grid_z = %gridZ) threads(%tx, %ty, %tz) in (%block_x = %blockX, %block_y = %blockY, %block_z = %blockZ) {
                                                   //A tile 128 * 72, from top left at %11, %12
                                                    %a = gpu.subgroup_mma_load_matrix %a_smem[%11, %12] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                                                    // B tile is 64 * 136, from top left at %12, %14
                                                    %b = gpu.subgroup_mma_load_matrix %b_smem[%12, %14] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                                                    //C matrix is 8192 * 8192, from top left at %16, %17, the 128 * 128 tile
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
        }

        // Deallocate device memory for matrices A, B, and C asynchronously
        %zA = gpu.dealloc async [%token] %A : memref<8192x8192xf16>
        %zB = gpu.dealloc async [%token] %B : memref<8192x8192xf16>
        %zC = gpu.dealloc async [%token] %C : memref<8192x8192xf32>

        // Wait for all asynchronous operations to complete
        gpu.wait [%token]
        return
    }
}
