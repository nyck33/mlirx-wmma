//Listing 3: Affine matmul after loop unrolling and invariant load-store hoisting.
//Now that we have generated the WMMA operations, we do the following IR transformations:
//• permute the outermost six loops to go from (i, j, k, ii, jj, kk) order to (i, j, ii, jj, k, kk) order. This later
//helps in mapping compute loops to GPU compute hierarchy. Additionally, it also helps in moving invariant
//load-store operations on C to the most outermost position possible.
//• permute the innermost three loops to go form (i, j, k) to (k, i, j). This represents warp level MMA operation
//as an outer product and enhances ILP, as pointed out by Bhaskaracharya et al. [4].
//• fully unroll the innermost three loops.

#map0 = affine_map<(d0, d1) -> (d0 + d1)>
#map_add16_to_row = affine_map<(d0) -> (d0 + 16)>
#map_add16_to_col = affine_map<(d1) -> (d1 + 16)>

//maps for unrolling kk, iii, jjj, kkk loop, use i,j,k in variable names
//input first kk then increment as needed
#map_unroll_kk = affine_map<(kk) -> (kk + 32)>
#map_unroll_iii = affine_map<(iii) -> (iii + 16)>
#map_unroll_jjj = affine_map<(jjj) -> (jjj + 16)>
#map_unroll_kkk = affine_map<(kkk) -> (kkk + 16)>

#map_kk_plus_kkk = affine_map<(kk, kkk) -> (kk + kkk)>


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
                // Hoisted loads on C for 64*32 tile of C, ie. C is made of 2*4 of these
                %11_0 = affine.apply #map0_3(%i, %ii) 
                %12_0 = affine.apply #map0_3(%j, %jj) 

                %11_1 = affine.apply #map_add16_to_row(%11)
                %11_2 = affine.apply #map_add16_to_row(%11_1)
                %11_3 = affine.apply #map_add16_to_row(%11_2)
                
                %12_1 = affine.apply #map_add16_to_col(%12)
                
                //naming convention is row col on 64*64 grid of 16*16 warp tiles making up 128*128 tile
                %c_reg_0_0 = gpu.subgroup_mma_load_matrix %C[%11_0, %12_0] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                %c_reg_0_1 = gpu.subgroup_mma_load_matrix %C[%11_0, %12_1] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                %c_reg_1_0 = gpu.subgroup_mma_load_matrix %C[%11_1, %12_0] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                %c_reg_1_1 = gpu.subgroup_mma_load_matrix %C[%11_1, %12_1] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                %c_reg_2_0 = gpu.subgroup_mma_load_matrix %C[%11_2, %12_0] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                %c_reg_2_1 = gpu.subgroup_mma_load_matrix %C[%11_2, %12_1] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                %c_reg_3_0 = gpu.subgroup_mma_load_matrix %C[%11_3, %12_0] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

                %c_reg_3_1 = gpu.subgroup_mma_load_matrix %C[%11_3, %12_1] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                //end of top left 64*32 subtile in 128*128 tile

                //todo: need 64 registers for C, since I"m filling 128 * 128 tile with 8*8 warp tiles

                // Main ‘k‘-loop with loaded C operand as iter_args. thread block mapping, use all the c_regs as iter args
                //refer to https://discourse.llvm.org/t/how-does-affine-yield-work-into-an-array/78215
                %res:64 = affine.for %k = 0 to 8192 step 64 iter_args(%c_in_0_0 = %c_reg_0_0, %c_in_0_1 = %c_reg_0_1, %c_in_1_0 = %c_reg_1_0, %c_in_1_1 = %c_reg_1_1, %c_in_2_0 = %c_reg_2_0, %c_in_2_1 = %c_reg_2_1, %c_in_3_0 = %c_reg_3_0, %c_in_3_1 = %c_reg_3_1) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">){
                    //when kk shifts you accumulate both into th e same c tile in the same c register
                    %kk_0 = arith.constant 0 : index
                    %kk_32 = %kk + 32 : index
                    //64 step 16
                    %iii_0 = arith.constant 0 : index
                    %iii_16 = %iii + 16 : index
                    %iii_32 = %iii + 32 : index
                    %iii_48 = %iii + 48 : index
                    //32 step 16
                    %jjj_0 = arith.constant 0 : index
                    %jjj_16 = %jjj + 16 : index
                    //32 step 16
                    %kkk_0 = arith.constant 0 : index
                    %kkk_16 = %kkk + 16 : index
                    
                    //annotate with coordinates in shared memory warp tile loaded from 
                    %a_0_0 = gpu.subgroup_mma_load_matrix %a_smem[%iii, %kkk] {leadDimension = 72 : index} :memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                    %b_0_0 = gpu.subgroup_mma_load_matrix %b_smem[%kkk, %jjj] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                    //registers are named by coordinates, 64 locations
                    %c_res_0_0 = gpu.subgroup_mma_compute %a_0_0, %b_0_0, %c_in_0_0 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                    //increment kkk, same c reg accumulation
                    %a_0_1 = gpu.subgroup_mma_load_matrix %a_smem[%iii, %kkk_16] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                    %b_1_0 = gpu.subgroup_mma_load_matrix %b_smem[%kkk_16, %jjj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                    %c_res_1 = gpu.subgroup_mma_compute %a_0_1, %b_1_0, %c_in_0 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                    //reset kkk to 0, increment jjj
                    %a_0_0_16_0 = gpu.subgroup_mma_load_matrix %a_smem[%iii, %kkk] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                    %b_0_0_16_0 = gpu.subgroup_mma_load_matrix %b_smem[%kkk, %jjj_16] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                    %c_res_2 = gpu.subgroup_mma_compute %a_0_0_16_0, %b_0_0_16_0, %c_in_1 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                    //increment kkk, same creg
                    %a_0_0_16_16 = gpu.subgroup_mma_load_matrix %a_smem[%iii, %kkk_16] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                    %b_0_0_16_16 = gpu.subgroup_mma_load_matrix %b_smem[%kkk_16, %jjj_16] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                    %c_res_3 = gpu.subgroup_mma_compute %a_0_0_16_16, %b_0_0_16_16, %c_in_3 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

                    //kkk and jjj back to 0, increment iii
                    %a_0_16_0_0 = gpu.subgroup_mma_load_matrix %a_smem[%iii_16, %kkk] {leadDimension = 72 : index} : !gpu.mma_matrix<16x16xf16, "AOp">
                    %b_0_16_0_0 = gpu.subgroup_mma_load_matrix %b_smem[%kkk, %jjj] {leadDimension = 136 : index} : !gpu.mma_matrix<16x16xf16, "BOp">
                    %c_res_4 = gpu.subgroup_mma_compute %a_0_16_0_0, %b_0_16_0_0, %c_in_4 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

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
                // Hoisted stores on C so index subtile and tile
                //todo: fix the indexing here too to use affine maps
                //unrolled are %kk, %iii, %jjj, %kkk
                //input first kk then increment as needed
                
                //#map_unroll_kk = affine_map<(kk) -> (kk + 32)>
                //#map_unroll_iii = affine_map<(iii) -> (iii + 16)>
                //#map_unroll_jjj = affine_map<(jjj) -> (jjj + 16)>
                //#map_unroll_kkk = affine_map<(kkk) -> (kkk + 16)
                //#map_kk_plus_kkk = affine_map<(kk, kkk) -> (kk + kkk)>

                %kk_st = arith.constant 0 : index
                %iii_st = arith.constant 0 : index
                %jjj_st = arith.constant 0 : index
                %kkk_st = arith.constant 0 : index

                //new maps to be defined at top of file
                //%ii_plus_iii_st = %ii + %iii_st
                //%jj_plus_jjj_st = %jj + %jjj_st
                //%kk_plus_kkk_st = %kk + %kkk_st
                //index into C at the warp tile level within subtile
                gpu.subgroup_mma_store_matrix %res#0, %C[%ii_plus_iii_st, %jj_plus_jjj_st] {leadDimension = 8192 : index} :
                    !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>

                //store all the other yielded matrices
                //increment by kkk, within the tile dims of 64*32, 32*32 = 64 * 32 

                gpu.subgroup_mma_store_matrix %res#1, %C[%ii+%iii+%kkk, %jj+ %jjj + %kkk] {leadDimension = 8192 : index} :
                    !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
                %kkk = 0
                %jjj += 16
                //%kkk back to 0, increment %jjj so next col of B, same row of A, next col of C
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
