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

# Function to calculate indices based on affine maps
def calculate_indices(i, j, k, ii, jj, kk):
    idx_a_row = i + ii * 64 + kk * 16
    idx_a_col = k + kk * 32
    idx_b_col = j + jj * 32 + kk * 16
    idx_c_row = i + ii * 64 + kk * 16
    idx_c_col = j + jj * 32 + kk * 16
    return idx_a_row, idx_a_col, idx_b_col, idx_c_row, idx_c_col

# Nested loops to emulate the MLIR loop structure
for i in range(0, size, tile_size_i):  # Global row index for tile C (and A)
    for j in range(0, size, tile_size_j):  # Global column index for tile C (and B)
        for k in range(0, size, tile_size_k):  # Shared dimension for A and B
            for ii in range(0, tile_size_i, subtile_size_ii):  # Local row index within tile A
                for jj in range(0, tile_size_j, subtile_size_jj):  # Local column index within tile B
                    for kk in range(0, tile_size_k, subtile_size_kk):  # Shared local dimension for A and B
                        for kkk in range(0, subtile_size_kk, 16):  # Sub-tile within tile A and B
                            for iii in range(0, subtile_size_ii, 16):  # Sub-tile within tile A
                                for jjj in range(0, subtile_size_jj, 16):  # Sub-tile within tile B
                                    # Calculate indices using affine maps
                                    idx_a_row, idx_a_col, idx_b_col, idx_c_row, idx_c_col = calculate_indices(i, j, k, ii + iii, jj + jjj, kk + kkk)

                                    # Local indices within tiles A and B
                                    local_idx_a_row = idx_a_row - i
                                    local_idx_a_col = idx_a_col - k
                                    local_idx_b_col = idx_b_col - j

                                    # Assert that local indices are within tile boundaries
                                    assert 0 <= local_idx_a_row < tile_a_dims[0] and 0 <= local_idx_a_col < tile_a_dims[1], f"Local index out of tile A boundaries: A[{local_idx_a_row}, {local_idx_a_col}]"
                                    assert 0 <= local_idx_b_col < tile_b_dims[1], f"Local index out of tile B boundaries: B[, {local_idx_b_col}]"

                                    # Assert that global indices are within the global matrix
                                    assert 0 <= idx_a_row < size and 0 <= idx_a_col < size, f"Global index out of boundaries for A: A[{idx_a_row}, {idx_a_col}]"
                                    assert 0 <= idx_b_col < size, f"Global index out of boundaries for B: B[, {idx_b_col}]"
                                    assert 0 <= idx_c_row < size and 0 <= idx_c_col < size, f"Global index out of boundaries for C: C[{idx_c_row}, {idx_c_col}]"

                                    print(f"A[{idx_a_row}, {idx_a_col}], B[, {idx_b_col}], C[{idx_c_row}, {idx_c_col}]")
