# Python script to generate all 32 MLIR code permutations for matrix multiplication

def generate_mlir_code():
    mlir_code = ""
    mlir_code_arr = []
    for kk in range(2):
        for iii in range(4):  # 0, 1, 2, 3
            for jjj in range(2):  # 0, 1
                for kkk in range(2):  # 0, 1
                    iii_index = iii * 16
                    jjj_index = jjj * 16
                    kkk_index = kkk * 16
                    a_load = f"%a_{iii}_{kkk} = gpu.subgroup_mma_load_matrix %a_smem[%iii_{iii_index}, %kkk_{kkk_index}] {{leadDimension = 72 : index}} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, \"AOp\">\n"
                    b_load = f"%b_{kkk}_{jjj} = gpu.subgroup_mma_load_matrix %b_smem[%kkk_{kkk_index}, %jjj_{jjj_index}] {{leadDimension = 136 : index}} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, \"BOp\">\n"
                    c_compute = f"%c_res_{iii}_{jjj}_{kkk} = gpu.subgroup_mma_compute %a_{iii}_{kkk}, %b_{kkk}_{jjj}, %c_in_{iii*2+jjj} : !gpu.mma_matrix<16x16xf16, \"AOp\">, !gpu.mma_matrix<16x16xf16, \"BOp\"> -> !gpu.mma_matrix<16x16xf32, \"COp\">\n\n"
                    mlir_code += a_load + b_load + c_compute
                    mlir_code_arr.append(mlir_code)
        return mlir_code_arr, mlir_code

mlir_code_arr, mlir_code = generate_mlir_code()
print(len(mlir_code_arr))
print(mlir_code)
