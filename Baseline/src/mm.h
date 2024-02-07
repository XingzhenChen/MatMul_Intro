extern "C" {
void matrix_multiply(unsigned int *in0, // Read-Only matrix 0
          unsigned int *in1, // Read-Only matrix 1
          unsigned int *out0,// Output Result
          int M,                    // Size in integer
          int K,                    // Size in integer
          int N                    // Size in integer
          );

}