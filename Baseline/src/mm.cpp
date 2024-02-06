
#define BUFFER_SIZE_M 1024
#define BUFFER_SIZE_K 1024
#define BUFFER_SIZE_N 1024
#define DATA_SIZE 1024
#define iter 1

// TRIPCOUNT identifier
const unsigned int c_lens = DATA_SIZE * DATA_SIZE;
const unsigned int c_len = DATA_SIZE * DATA_SIZE;




extern "C" {
void matrix_multiply(const unsigned int *in0, // Read-Only matrix 0
          const unsigned int *in1, // Read-Only matrix 1
          unsigned int *out0,// Output Result
          int M,                    // Size in integer
          int K,                    // Size in integer
          int N                    // Size in integer
          ) {

#pragma HLS INTERFACE m_axi port = in0 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = out0 offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = in0 bundle = control
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = out0 bundle = control

#pragma HLS INTERFACE s_axilite port = M bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control



unsigned int v0_buffer[BUFFER_SIZE_M][BUFFER_SIZE_K];   // Local memory to store vector0
unsigned int v1_buffer[BUFFER_SIZE_K][BUFFER_SIZE_N];   // Local memory to store vector1
unsigned int vout_buffer[BUFFER_SIZE_M][BUFFER_SIZE_N]; // Local Memory to store result
unsigned int temp_sum[BUFFER_SIZE_N];

for(int iteration = 0; iteration < iter; iteration++) {
  // Per iteration of this loop perform M*K*N matrix multiply

//Read0
  for(int seq = 0, i = 0, j = 0; seq < M * K; seq++){
  #pragma HLS LOOP_TRIPCOUNT min = c_lens max = c_lens
  #pragma HLS PIPELINE II = 1
    if(j == BUFFER_SIZE_K){
      j = 0;
      i++;
    }
    v0_buffer[i][j] = in0[seq];
    j++;
  }

//Read1
  for(int seq = 0, i = 0, j = 0; seq < K * N; seq++){
  #pragma HLS LOOP_TRIPCOUNT min = c_lens max = c_lens
  #pragma HLS PIPELINE II = 1
    if(j == BUFFER_SIZE_N){
      j = 0;
      i++;
    }
    v1_buffer[i][j] = in1[seq];
    j++;
  }

//Calculate
  for(int i = 0; i < BUFFER_SIZE_M; i++){
  #pragma HLS LOOP_TRIPCOUNT min = c_len max = c_len

    for(int j = 0; j < BUFFER_SIZE_N; j++){
    #pragma HLS LOOP_TRIPCOUNT min = c_len max = c_len
      
      unsigned int result = 0;
      for(int k = 0; k < BUFFER_SIZE_K; k++){
      #pragma HLS LOOP_TRIPCOUNT min = c_len max = c_len

        result += v0_buffer[i][k] * v1_buffer[k][j];
        if (k == BUFFER_SIZE_K - 1)
          vout_buffer[i][j] = result;

      }
    }
  }

//Write
  for(int seq = 0, i = 0, j = 0; seq < M * N; seq++){
  #pragma HLS LOOP_TRIPCOUNT min = c_lens max = c_lens
  #pragma HLS PIPELINE II = 1
    if(j == BUFFER_SIZE_N){
      j = 0;
      i++;
    }
    out0[seq] = vout_buffer[i][j];
    j++;
  }



}
}
}



