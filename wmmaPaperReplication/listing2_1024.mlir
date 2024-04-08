//mlir-opt listing2_1024.mlir  | mlir-opt -lower-affine -convert-scf-to-cf -test-lower-to-nvvm="host-bare-ptr-calling-convention=1 kernel-bare-ptr-calling-convention=1 cubin-chip=sm_75 cubin-format=fatbin" -mlir-print-debuginfo | mlir-cpu-runner   --shared-libs=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build/lib/libmlir_cuda_runtime.so   --shared-libs=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build/lib/libmlir_runner_utils.so   --entry-point-result=void -shared-libs=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build/lib/libmlir_c_runner_utils.so

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 64)>
#map2 = affine_map<(d0) -> (d0 + 128)>

//new affine maps
// Affine maps for local indexing within shared memory tiles
#localA_row = affine_map<(ii, iii) -> (ii + iii)>
#localA_col = affine_map<(kk, kkk) -> (kk + kkk)>
#localB_col = affine_map<(jj, jjj) -> (jj + jjj)>

// Affine maps for calculating global indices for storing into matrix C
#globalC_row = affine_map<(i, ii, iii) -> (i + ii + iii)>
#globalC_col = affine_map<(j, jj, jjj) -> (j + jj + jjj)>
module {
    // Shared memory buffers for A and B.
    memref.global "private" @b_smem_global : memref<64x136xf16, 3>
    memref.global "private" @a_smem_global : memref<128x72xf16, 3>
    func.func @main() {
        // Allocate memory for A, B on host using half precision
        %hA = memref.alloc() : memref<1024x1024xf16>
        %hB = memref.alloc() : memref<1024x1024xf16>
        // Allocate memory for output matrix C on host using single precision
        %hC = memref.alloc() : memref<1024x1024xf32>

        // Define constants used in the program
        %f1 = arith.constant 1.0e+00 : f16 // Constant value 1.0 of type half-precision float
        %f0 = arith.constant 0.0e+00 : f32 // Constant value 0.0 of type single-precision float

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index

        %c1024 = arith.constant 1024 : index

        //initialize the input matrices with ones
        scf.for %arg0 = %c0 to %c1024 step %c1 {
            scf.for %arg1 = %c0 to %c1024 step %c1 {
                memref.store %f1, %hA[%arg0, %arg1] : memref<1024x1024xf16>
            }
        }

        //now initialize hB with ones
        scf.for %arg0 = %c0 to %c1024 step %c1 {
            scf.for %arg1 = %c0 to %c1024 step %c1 {
                memref.store %f1, %hB[%arg0, %arg1] : memref<1024x1024xf16>
            }
        }

        //now initialize hC with zeros
        scf.for %arg0 = %c0 to %c1024 step %c1 {
            scf.for %arg1 = %c0 to %c1024 step %c1 {
                memref.store %f0, %hC[%arg0, %arg1] : memref<1024x1024xf32>
            }
        }
        // Asynchronous operations token
        %token = gpu.wait async

        // Allocate device memory for matrices A, B, and C asynchronously
        %A, %tokenA = gpu.alloc async [%token] () : memref<1024x1024xf16>
        %B, %tokenB = gpu.alloc async [%token] () : memref<1024x1024xf16>
        %C, %tokenC = gpu.alloc async [%token] () : memref<1024x1024xf32>

        // Copy A and B from host to device asynchronously
        %copyA = gpu.memcpy async [%token] %A, %hA : memref<1024x1024xf16>, memref<1024x1024xf16>
        %copyB = gpu.memcpy async [%token] %B, %hB : memref<1024x1024xf16>, memref<1024x1024xf16>
        // Copy C from host to device asynchronously (if initialization on host is needed)
        %copyC = gpu.memcpy async [%token] %C, %hC : memref<1024x1024xf32>, memref<1024x1024xf32>

        //define block and grid xyz where each block is 16 x 16 threads
        //thread block granular to warp operations (256 threads per block)
        %blockX = arith.constant 16 : index
        %blockY = arith.constant 16 : index
        %blockZ = arith.constant 1 : index
        %gridX = arith.constant 64 : index // Adjusted for 1024x1024 matrix size
        %gridY = arith.constant 64 : index // Adjusted for 1024x1024 matrix size
        %gridZ = arith.constant 1 : index

        %t_start = call @rtclock() : () -> (f32)

        %reps = index.constant 5
        //define the gpu.launch blocks for the kernel, grid (2,4,1), block (16,16,1) for 64*32 result tile covered by threads
        affine.for %ti = 0 to %reps{
            gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gridX, %grid_y = %gridY, %grid_z = %gridZ) threads(%tx, %ty, %tz) in (%block_x = %blockX, %block_y = %blockY, %block_z = %blockZ) {

            affine.for %i = 0 to 1024 step 128 { //128 rows of tile C/A
                affine.for %j = 0 to 1024 step 128 {//128 cols of tile C/B
                
                    // References to shared memory buffers.
                    %b_smem = memref.get_global @b_smem_global : memref<64x136xf16, 3>
                    %a_smem = memref.get_global @a_smem_global : memref<128x72xf16, 3>
                    //main k-loop
                    affine.for %k = 0 to 1024 step 64 {  //64 is cols A tile and rows B tile 
                        // Copy loop for B tile
                        affine.for %copykk = #map0(%k) to #map1(%k) {//k to k + 64
                            affine.for %copyjj = #map0(%j) to #map2(%j) {//j to j + 128
                                %11 = affine.load %B[%copykk, %copyjj] : memref<1024x1024xf16>
                                affine.store %11, %b_smem[%copykk - %k, %copyjj - %j] : memref<64x136xf16, 3>
                            }
                        }
                        // Copy loop for A tile
                        affine.for %copyii = #map0(%i) to #map2(%i) {//i to i + 128
                            affine.for %copykk = #map0(%k) to #map1(%k) {//k to k + 64
                                %11 = affine.load %A[%copyii, %copykk] : memref<1024x1024xf16>
                                //%11 is the value of A at %copyii, %copykk
                                //copyii - i to index into the shared memory of size 128 * 72, the padded section remains as 
                                affine.store %11, %a_smem[%copyii - %i, %copykk - %k] : memref<128x72xf16, 3>
                            }
                        }
                        //copied so iterate over the tiles
                        affine.for %ii = 0 to 128 step 64 {//rows A tile
                            affine.for %jj = 0 to 128 step 32 {//cols B tile
                                affine.for %kk = 0 to 64 step 32 {//cols A tile
                                //iterate the 64 * 32 A minitile, 32 * 32 B minitile
                                    affine.for %kkk = 0 to 32 step 16 {//2 steps
                                        affine.for %iii = 0 to 64 step 16 {//4 steps
                                            affine.for %jjj = 0 to 32 step 16 {//2 steps

                                                // Assuming %ii, %jj, %kk are the local loop variables for sub-tile indexing
                                                %localA_row = affine.apply #localA_row(%ii, %iii) // Local row index within %a_smem
                                                %localA_col = affine.apply #localA_col(%kk, %kkk) // Local column index within %a_smem
                                                %localB_col = affine.apply #localB_col(%jj, %jjj) // Local column index within %b_smem

                                                // Calculate global indices for storing the result into matrix C
                                                %globalC_row = affine.apply #globalC_row(%i, %ii, %iii)
                                                %globalC_col = affine.apply #globalC_col(%j, %jj, %jjj)
                                                
                                                //A tile 128 * 72 load 16 * 16 fragment into warp matrix
                                                %a = gpu.subgroup_mma_load_matrix %a_smem[%localA_row, %localA_col] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                                                
                                                // B tile is 64 * 136, load 16 * 16 fragment into warp matrix
                                                %b = gpu.subgroup_mma_load_matrix %b_smem[%localA_col, %localB_col] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                                            
                                                //C matrix is 1024 * 1024, load 16 * 16 fragment into warp matrix
                                                %c = gpu.subgroup_mma_load_matrix %C[%localA_row, %localB_col] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                                                %res = gpu.subgroup_mma_compute %a, %b, %c : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                                                //stores the warp tile into global result matrix so need %16 and %17 to be global
                                                gpu.subgroup_mma_store_matrix %res, %C[%globalC_row, %globalC_col] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>

                                                    
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            // Print success message to indicate successful execution
            //gpu.printf "Success\n"
            // CHECK: Success
            gpu.terminator
        }
        }
        

        %t_end = call @rtclock() : () -> (f32)

        %t0 = index.constant 0
        %t1 = index.constant 1
        //rows of C
        %M = memref.dim %C, %t0 : memref<1024x1024xf32>
        //cols of C
        %N = memref.dim %C, %t1 : memref<1024x1024xf32>
        //cols of A
        %K = memref.dim %A, %t1 : memref<1024x1024xf16> 
        //time taken
        %time = arith.subf %t_end, %t_start : f32
        //flops
        %prod1 = arith.muli %M, %N : index
        %prod2 = arith.muli %prod1, %K : index
        //2 * M * N * K
        %t2 = index.constant 2 
        %f3 = arith.muli %t2, %prod2 : index
        %num_flops = arith.muli %reps, %f3 : index
        %num_flops_i = arith.index_cast %num_flops : index to i32
        %num_flops_f = arith.sitofp %num_flops_i : i32 to f32
        %flops = arith.divf %num_flops_f, %time : f32

        // Print the time taken and the number of flops
        call @printF32(%time) : (f32) -> ()
        call@printNewline() : () -> ()
        call @printFlops(%flops) : (f32) -> ()

        // Deallocate device memory for matrices A, B, and C asynchronously
        %zA = gpu.dealloc async [%token] %A : memref<1024x1024xf16>
        %zB = gpu.dealloc async [%token] %B : memref<1024x1024xf16>
        %zC = gpu.dealloc async [%token] %C : memref<1024x1024xf32>

        // Wait for all asynchronous operations to complete
        gpu.wait [%token]
        return
    }

    func.func private @printNewline() -> ()
    func.func private @printF32(f32) -> ()
    func.func private @printFlops(f32) -> ()
    func.func private @rtclock() -> f32

        
    }

