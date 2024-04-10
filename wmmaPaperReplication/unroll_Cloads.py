mlir_code = ""
mlir_lines = []
for i in range(8):
    for j in range(8):
        reg_index = i * 8 + j
        mlir_code += f'''
                    %c_reg_{reg_index} = gpu.subgroup_mma_load_matrix %C[%11_{i}, %12_{j}] {{leadDimension = 8192 : index}} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                    '''
        mlir_lines.append(mlir_code)

print(f'len mlir_lines: {len(mlir_lines)}')
print(mlir_code)



