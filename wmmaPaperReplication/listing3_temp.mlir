//mlir-opt /mnt/d/LLVM/NewPolygeistDir/llvm-project/mlir/test/Integration/GPU/CUDA/TensorCore/wmmaPaperReplication/listing2.mlir  | mlir-opt -lower-affine -convert-scf-to-cf -test-lower-to-nvvm="host-bare-ptr-calling-convention=1 kernel-bare-ptr-calling-convention=1 cubin-chip=sm_75 cubin-format=fatbin" -mlir-print-debuginfo -mlir-print-ir-before-all | mlir-cpu-runner   --shared-libs=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build/lib/libmlir_cuda_runtime.so   --shared-libs=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build/lib/libmlir_runner_utils.so   --entry-point-result=void

//mlir-opt listing3_temp.mlir  | mlir-opt -lower-affine -convert-scf-to-cf -test-lower-to-nvvm="host-bare-ptr-calling-convention=1 kernel-bare-ptr-calling-convention=1 cubin-chip=sm_75 cubin-format=fatbin" -mlir-print-debuginfo | mlir-cpu-runner   --shared-libs=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build/lib/libmlir_cuda_runtime.so   --shared-libs=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build/lib/libmlir_runner_utils.so   --entry-point-result=void

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 64)>
#map2 = affine_map<(d0) -> (d0 + 128)>

//new affine maps for listing 3
#map0_3 = affine_map<(d0, d1) -> (d0 + d1)>
#map_add16_to_row = affine_map<(d0) -> (d0 + 16)>
#map_add16_to_col = affine_map<(d1) -> (d1 + 16)>

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
        //thread block granular to warp operations (256 threads per block)
        %blockX = arith.constant 16 : index
        %blockY = arith.constant 16 : index
        %blockZ = arith.constant 1 : index
        %gridX = arith.constant 512 : index
        %gridY = arith.constant 512 : index
        %gridZ = arith.constant 1 : index

        //define the gpu.launch blocks for the kernel, grid (2,4,1), block (16,16,1) for 64*32 result tile covered by threads
        gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gridX, %grid_y = %gridY, %grid_z = %gridZ) threads(%tx, %ty, %tz) in (%block_x = %blockX, %block_y = %blockY, %block_z = %blockZ) {

            // Thread block ‘i‘ loop.
            affine.for %i = 0 to 8192 step 128 {//outermost 2 loops mapped to thread block
                // Thread block ‘j‘ loop.
                affine.for %j = 0 to 8192 step 128 {
                    %b_smem = memref.get_global @b_smem_global : memref<64x136xf16, 3>
                    %a_smem = memref.get_global @a_smem_global : memref<128x72xf16, 3>
                    // Warp ‘ii‘ loop.
                    affine.for %ii = 0 to 128 step 64 {//ii, jj mapped to warps
                        // Warp ‘jj‘ loop.
                        affine.for %jj = 0 to 128 step 32 {
                            // Hoisted loads on C.
                            //added to get to tile location i = matrix, ii = tile, iii = 16* 16 warp tile
                            %11 = affine.apply #map0_3(%i, %ii)
                            %12 = affine.apply #map0_3(%j, %jj)
                            //load the first 16*16 warp tile of C, unrolled so increment rows, cols by step size so iterating the 64 * 32 C subtile that's the result of mamtmul on A subtile of 64*32, B subtile of 32*32

                            %c_reg_0 = gpu.subgroup_mma_load_matrix %C[%11, %12] {leadDimension = 8192 : index} :
                                memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                            //in lieu of the jjj or kkk loop, + 16 to get the next tile
                            %col_idx_inc = affine.apply #map_add16_to_col(%12)
                            %c_reg_1 = gpu.subgroup_mma_load_matrix %C[%11, %col_idx_inc] {leadDimension = 8192 : index} :
                                memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                            
                            %row_idx_inc1 = affine.apply #map_add16_to_row(%11)
                            %c_reg_2 = gpu.subgroup_mma_load_matrix %C[%row_idx_inc1, %12] {leadDimension = 8192 : index} :
                                memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                            //schedule to increment the rows and cols  (16, 16), (32, 0), (32, 16), (48, 0), (48, 16)
                            
                            %c_reg_3 = gpu.subgroup_mma_load_matrix %C[%row_idx_inc1, %col_idx_inc] {leadDimension = 8192 : index} :
                                memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                            //increment rows by 32, cols by 0
                            %row_idx_inc2 = affine.apply #map_add16_to_row(%row_idx_inc1)
                            %c_reg_4 = gpu.subgroup_mma_load_matrix %C[%row_idx_inc2, %12] {leadDimension = 8192: index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp"> 

                            
                            %c_reg_5 = gpu.subgroup_mma_load_matrix %C[%row_idx_inc2, %col_idx_inc] {leadDimension = 8192 : index} :
                                memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                            
                            %row_idx_inc3 = affine.apply #map_add16_to_row(%row_idx_inc2)
                            %c_reg_6 = gpu.subgroup_mma_load_matrix %C[%row_idx_inc3, %12] {leadDimension = 8192 : index} :
                                memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                            %c_reg_7 = gpu.subgroup_mma_load_matrix %C[%row_idx_inc3, %col_idx_inc] {leadDimension = 8192: index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp"> 

                            // Main ‘k‘-loop with loaded C operand as iter_args. thread block mapping, use all the c_regs as iter args
                            //refer to https://discourse.llvm.org/t/how-does-affine-yield-work-into-an-array/78215
                            //need a res array of 8 elements, ie. 8 warp tiles in each dimension of 128 * 128 C tile 
                            %res:8 = affine.for %k = 0 to 8192 step 64 iter_args(%c_in_0 = %c_reg_0, %c_in_1 = %c_reg_1, %c_in_2 = %c_reg_2, %c_in_3 = %c_reg_3, %c_in_4 = %c_reg_4, %c_in_5 = %c_reg_5, %c_in_6 = %c_reg_6, %c_in_7 = %c_reg_7) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">,!gpu.mma_matrix<16x16xf32, "COp">,!gpu.mma_matrix<16x16xf32, "COp">,!gpu.mma_matrix<16x16xf32, "COp">,!gpu.mma_matrix<16x16xf32, "COp">,!gpu.mma_matrix<16x16xf32, "COp">,!gpu.mma_matrix<16x16xf32, "COp">) {
                                //load from a shared memory to 16*16 warp tile, repeat 8 times for all the c_regs, make sure to increment the c_in_0, c_in_1, c_in_2, c_in_3, c_in_4, c_in_5, c_in_6, c_in_7 and the variables %a, %b, %c_res like %a_1, %b_1, %c_res_1, %a_2, %b_2, %c_res_2, etc.
                                //missing the iii,jjj, kkk indexing here so need to figure it out
                                //todo: indexing into shared memory needs fixing remembering the iii, jjj, kkk 
                                //%iii = %ii step 16
                                //%jjj = %jj step 16
                                //%kkk = 
                                %a = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_0] {leadDimension = 72 : index} :
                                    memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">

                                %b = gpu.subgroup_mma_load_matrix %b_smem[%c_in_0, %jj] {leadDimension = 136 : index} :
                                    memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                                %c_res = gpu.subgroup_mma_compute %a, %b, %c_in_0 :
                                    !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                                %a_1 = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_1] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                                %b_1 = gpu.subgroup_mma_load_matrix %b_smem[%c_in_1, %jj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                                %c_res_1 = gpu.subgroup_mma_compute %a_1, %b_1, %c_in_1 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                                %a_2 = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_2] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                                %b_2 = gpu.subgroup_mma_load_matrix %b_smem[%c_in_2, %jj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                                %c_res_2 = gpu.subgroup_mma_compute %a_2, %b_2, %c_in_2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                                %a_3 = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_3] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                                %b_3 = gpu.subgroup_mma_load_matrix %b_smem[%c_in_3, %jj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                                %c_res_3 = gpu.subgroup_mma_compute %a_3, %b_3, %c_in_3 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                                %a_4 = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_4] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                                %b_4 = gpu.subgroup_mma_load_matrix %b_smem[%c_in_4, %jj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                                %c_res_4 = gpu.subgroup_mma_compute %a_4, %b_4, %c_in_4 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                                %a_5 = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_5] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                                %b_5 = gpu.subgroup_mma_load_matrix %b_smem[%c_in_5, %jj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                                %c_res_5 = gpu.subgroup_mma_compute %a_5, %b_5, %c_in_5 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                                %a_6 = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_6] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                                %b_6 = gpu.subgroup_mma_load_matrix %b_smem[%c_in_6, %jj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                                %c_res_6 = gpu.subgroup_mma_compute %a_6, %b_6, %c_in_6 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                                %a_7 = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_7] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                                %b_7 = gpu.subgroup_mma_load_matrix %b_smem[%c_in_7, %jj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                                %c_res_7 = gpu.subgroup_mma_compute %a_7, %b_7, %c_in_7 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                                ...
                                // Rewriting the yield operation to include 8 matrix tiles as per the unrolled operations above.
                                // Each %c_res_n variable represents a computed matrix tile from the unrolled operations.
                                // These are the results of the matrix multiplication for each tile, which we now yield from the loop.
                                // The variables %c_res, %c_res_1, ..., %c_res_7 represent the 8 computed matrix tiles.
                                // We yield these tiles to pass them as results of the current iteration of the affine loop.
                                // Main ‘k‘-loop yielding the results of the current iteration, yield the c warp tile for all the above unrolled ops as %104 affine.yield %104, %107 ... : !gpu.mma_matrix<16x16xf32, "COp">, 
                                //!gpu.mma_matrix<16x16xf32, "COp">... 
                                affine.yield %c_warp_tile0, %c_warp_tile1, %c_warp_tile2, %c_warp_tile3, %c_warp_tile4, %c_warp_tile5, %c_warp_tile6, %c_warp_tile7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
                            }
                            // Hoisted stores on C.
                            //todo: fix the indexing here too to use affine maps
                            gpu.subgroup_mma_store_matrix %res#0, %C[%11, %12] {leadDimension = 8192 : index} :
                                !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>

                            //store all the other yielded matrices
                            gpu.subgroup_mma_store_matrix %res#1, %C[%11, %12 + 16] {leadDimension = 8192 : index} :
                                !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
                            gpu.subgroup_mma_store_matrix %res#2, %C[%11 + 16, %12] {leadDimension = 8192 : index} :
                                !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
                            gpu.subgroup_mma_store_matrix %res#3, %C[%11 + 16, %12 + 16] {leadDimension = 8192 : index} :
                                !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
                            gpu.subgroup_mma_store_matrix %res#4, %C[%11 + 32, %12] {leadDimension = 8192 : index} :
                                !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
                            
                            gpu.subgroup_mma_store_matrix %res#5, %C[%11 + 32, %12 + 16] {leadDimension = 8192 : index} :
                                !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
                            
                            gpu.subgroup_mma_store_matrix %res#6, %C[%11 + 48, %12] {leadDimension = 8192 : index} : 
                                !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
                            
                            gpu.subgroup_mma_store_matrix %res#7, %C[%11 + 48, %12 + 16] {leadDimension = 8192 : index} : 
                                !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>

                        }
                    }
                }
            }
            // Print success message to indicate successful execution
            //gpu.printf "Success\n"
            // CHECK: Success
            gpu.terminator
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


