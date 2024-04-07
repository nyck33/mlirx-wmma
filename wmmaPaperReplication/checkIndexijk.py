# Constants
size = 8192
tile_size_i = 128
tile_size_j = 128
tile_size_k = 64
subtile_size_ii = 64
subtile_size_jj = 32
subtile_size_kk = 32

# Tile dimensions
tile_a_dims = (128, 72)  # 128 rows, 72 columns (with padding)
tile_b_dims = (64, 136)  # 64 rows, 136 columns (with padding)
tile_c_dims = (128, 128) # 128 rows, 128 columns

import numpy as np
#want matrix of all zeros or 1's of dims 8192 x 8192
mat = np.zeros((8192, 8192))

# Function to calculate indices based on affine maps
def calculate_indices(i, j, k, ii, jj, kk, iii, jjj, kkk):
    local_idx_a_row = ii + iii
    local_idx_a_col = kk + kkk
    local_idx_b_col = jj + jjj
    global_idx_a_row = i + local_idx_a_row
    global_idx_a_col = k + local_idx_a_col
    global_idx_b_col = j + local_idx_b_col
    # only want top left corner of 
    global_idx_c_row = i + ii + iii
    global_idx_c_col = j + jj + jjj
    return local_idx_a_row, local_idx_a_col, local_idx_b_col, global_idx_a_row, global_idx_a_col, global_idx_b_col, global_idx_c_row, global_idx_c_col

# Nested loops to emulate the MLIR loop structure
for i in range(0, size, tile_size_i):  # (0, 8192, 128)
    for j in range(0, size, tile_size_j):  # (0, 8192, 128)
        for k in range(0, size, tile_size_k):  # (0, 8192, 64)
            for ii in range(0, tile_size_i, subtile_size_ii):  # (0, 128, 64)
                for jj in range(0, tile_size_j, subtile_size_jj):  # (0, 128, 32)
                    for kk in range(0, tile_size_k, subtile_size_kk):  # (0, 64, 32)
                        for iii in range(0, subtile_size_ii, 16):  # (0, 64, 16)
                            for jjj in range(0, subtile_size_jj, 16):  # (0, 32, 16)
                                for kkk in range(0, subtile_size_kk, 16):  # (0, 32, 16)
                                    # Calculate indices using affine maps
                                    local_idx_a_row, local_idx_a_col, local_idx_b_col, global_idx_a_row, global_idx_a_col, global_idx_b_col, global_idx_c_row, global_idx_c_col = calculate_indices(i, j, k, ii,jj,kk,iii, jjj, kkk)

                                    # 8192 * 8192 matrix is tiled, A matrix is 128 * 64, B matrix is 64 * 128, C matrix is 128 * 128
                                    # A matrix tile is further tiled into subtiles of 64 * 32 and B tiles into subtiles of 32 * 32 which multplied make res matrix of 64 * 32
                                    # A matrix subtile of 64 * 32 is further tiled into 16 * 16 subtiles, and B subtile of 32 * 32 is further tiled into 16 * 16 subtiles for the warp matrix multiplication

                                    # check the warp tile being loaded for A is within the A tile's boundaries <= [128, 72] in x, y
                                    assert local_idx_a_row <= 128 and local_idx_a_col <= 64, f"A warp tile out of A tile's boundary: local_idx_a_row={local_idx_a_row}, local_idx_a_col={local_idx_a_col}"

                                    # check the warp tile being loaded for B is within the B tile's boundaries <= [64, 128] in x, y
                                    assert local_idx_b_col <= 128, f"B warp tile out of B tile's boundary: local_idx_b_col={local_idx_b_col}"

                                    # check the warp tile being loaded for C is within the C tile's boundaries basically <= [128,128] in x, y
                                    assert global_idx_c_row <= 8192 and global_idx_c_col <= 8192, f"C warp tile out of C tile's boundary: global_idx_c_row={global_idx_c_row}, global_idx_c_col={global_idx_c_col}"

                                    # check the A warp tile's global indices are within the global boundaries of A matrix <= [8192, 8192] in x, y       
                                    assert global_idx_a_row <= 8192 and global_idx_a_col <= 8192, f"A warp tile's global indices out of A matrix's global boundary: global_idx_a_row={global_idx_a_row}, global_idx_a_col={global_idx_a_col}"

                                    # check the B warp tile's global indices are within the global boundaries of B matrix <= [8192, 8192] in x, y
                                    assert global_idx_b_col <= 8192, f"B warp tile's global indices out of B matrix's global boundary: global_idx_b_col={global_idx_b_col}"

                                    # fill mat with 1's
                                    mat[global_idx_c_row, global_idx_c_col] = 1

# assert matrix is all 1's for every 16th row and every 16th column using numpy
assert np.all(mat[::16, ::16] == 1), "Matrix is not all 1's at specified indices"
                                    
