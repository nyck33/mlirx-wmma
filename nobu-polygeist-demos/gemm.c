//PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
cgeist simple_matmul.cu -function=matmul --cuda-path=/usr/local/cuda-11.8 --cuda-gpu-arch=sm_75

#define N 200
#define M 300
#define K 400
#define DATA_TYPE float

void matmul(DATA_TYPE A[N][K], DATA_TYPE B[K][M], DATA_TYPE C[N][M]) {
  int i, j, k;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}