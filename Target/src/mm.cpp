#include "ap_int.h"
#include <inttypes.h>
#include <stdio.h>


#define BUFFER_SIZE_M 1024
#define BUFFER_SIZE_K 1024
#define BUFFER_SIZE_N 1024
#define DATA_SIZE 1024
#define iter 1
#define tile_size 128

// TRIPCOUNT identifier
const unsigned int c_lens = 128;
const unsigned int c_len = 128;

union fp_int{
  unsigned int integer;
  float fp;
};

void Read(bool enable, 
          const ap_uint<512> in[],
          ap_uint<512> buffer[tile_size][tile_size / 16],
          int row_begin,
          int column_begin
              ){
  if(enable){
    printf("Read[%d][%d]\r\n",row_begin,column_begin);
    for(int i = 0; i < tile_size; i++){
      for(int j = 0; j < tile_size / 16; j++){
      #pragma HLS PIPELINE II = 1
        buffer[i][j] = in[(row_begin + i) * (DATA_SIZE / 16) + (column_begin / 16 + j)];
      }
    }
  }

}


void Calculate( bool enable,
                ap_uint<512> LHS_buffer[tile_size][tile_size / 16],
                ap_uint<512> RHS_buffer[tile_size][tile_size / 16], 
                ap_uint<512> OUT_buffer[tile_size][tile_size / 16]
                ){
  if(enable){ //kij loop method
    printf("Calculate\r\n");
    for(int k = 0; k < tile_size; k++){
        for(int i = 0; i < tile_size; i++){
            #pragma HLS PIPELINE II = 1
            ap_uint<512> LHS = LHS_buffer[i][k / 16];
            fp_int A;
            A.integer = LHS(32 * ((k % 16) + 1) - 1, 32 * (k % 16));
            for(int j = 0; j < tile_size; j+=16){// j = 0 ~ 15 
                #pragma HLS unroll
                ap_uint<512> RHS = RHS_buffer[k][j / 16];
                ap_uint<512> Output = OUT_buffer[i][j / 16];
                for(int m = 0; m < 16; m++){
                    #pragma HLS unroll
                    fp_int temp;
                    temp.integer = Output(32 * (m + 1) - 1, 32 * m);
                    fp_int B;
                    B.integer = RHS(32 * (m + 1) - 1, 32 * m);
                    float AB = A.fp * B.fp;
                    temp.fp += AB;
                    Output(32 * (m + 1) - 1, 32 * m) = temp.integer;
                }
                OUT_buffer[i][j / 16] = Output;
            }
        }
    }
  }
}


void Write_To_Temp( bool enable,
                    ap_uint<512> buffer[][tile_size / 16],
                    ap_uint<512> temp[][tile_size / 16]
                    ){
    if(enable){
        printf("Write_To_Temp\r\n");
        for(int i = 0; i < tile_size; i++){
        #pragma HLS pipeline II = 1
            for(int j = 0; j < tile_size / 16; j++){
            #pragma HLS unroll
                ap_uint<512> Buffer = buffer[i][j];
                buffer[i][j] = 0;    //init buffer
                ap_uint<512> Temp = temp[i][j];
                for(int m = 0; m < 16; m++){
                #pragma HLS unroll
                    fp_int A;
                    A.integer = Buffer(32 * (m + 1) - 1, 32 * m);
                    fp_int B;
                    B.integer = Temp(32 * (m + 1) - 1, 32 * m);
                    B.fp = B.fp + A.fp;
                    Temp(32 * (m + 1) - 1, 32 * m) = B.integer;
                }
                temp[i][j] = Temp;
            }
        }
    }

}

void Write_To_Memory( bool enable,
                      ap_uint<512> out[],
                      ap_uint<512> buffer[][tile_size / 16],
                      int row_begin,
                      int column_begin
                      ){

  if(enable){
    printf("Write_To_Memory[%d][%d]\r\n",row_begin,column_begin);
    for(int i = 0; i < tile_size; i++){
      for(int j = 0; j < tile_size / 16; j++){
      #pragma HLS PIPELINE II = 1
        out[(row_begin + i) * (DATA_SIZE / 16) + (column_begin / 16 + j)] = buffer[i][j];
        buffer[i][j] = 0;  //init buffer
      }

    }
  }

}

void Init(  bool enable,
            ap_uint<512> buffer[][tile_size / 16]
            ){
  for(int i = 0; i < tile_size; i++){
    for(int j = 0; j < tile_size / 16; j++){
    #pragma HLS pipeline II = 1
        buffer[i][j] = 0;
    }
  }
}




extern "C" {
void matrix_multiply(ap_uint<512> *in0, // Read-Only matrix 0
                     ap_uint<512> *in1, // Read-Only matrix 1
                     ap_uint<512> *out0,// Output Result
                     int M0,                    // Size in integer
                     int K0,                    // Size in integer
                     int N0,                    // Size in integer
                     int MagicFlag
                     ) {

#pragma HLS INTERFACE m_axi port = in0 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = out0 offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = in0 bundle = control
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = out0 bundle = control

#pragma HLS INTERFACE s_axilite port = M0 bundle = control
#pragma HLS INTERFACE s_axilite port = K0 bundle = control
#pragma HLS INTERFACE s_axilite port = N0 bundle = control

#pragma HLS INTERFACE s_axilite port = MagicFlag bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control



// unsigned int v0_buffer_first[tile_size][tile_size];   // Local memory to store left
// unsigned int v1_buffer_first[tile_size][tile_size];   // Local memory to store right
// unsigned int vout_buffer_first[tile_size][tile_size]; // Local Memory to store result
// unsigned int temp_sum[tile_size][tile_size];    // Local Memory to store sum of result

// unsigned int v0_buffer_second[tile_size][tile_size];   // Local memory to store left
// unsigned int v1_buffer_second[tile_size][tile_size];   // Local memory to store right
// unsigned int vout_buffer_second[tile_size][tile_size]; // Local Memory to store result

// #pragma HLS array_partition variable=v0_buffer_first dim=2
// #pragma HLS array_partition variable=v1_buffer_first dim=2
// #pragma HLS array_partition variable=v0_buffer_second dim=2
// #pragma HLS array_partition variable=v1_buffer_second dim=2

// #pragma HLS array_partition variable=vout_buffer_first dim=2
// #pragma HLS array_partition variable=vout_buffer_second dim=2
// #pragma HLS array_partition variable=temp_sum dim=2

ap_uint<512> LHS_buffer_PI[tile_size][tile_size / 16];   // Local memory to store left
ap_uint<512> RHS_buffer_PI[tile_size][tile_size / 16];   // Local memory to store right
ap_uint<512> OUT_buffer_PI[tile_size][tile_size / 16]; // Local Memory to store result

ap_uint<512> LHS_buffer_PO[tile_size][tile_size / 16];   // Local memory to store left
ap_uint<512> RHS_buffer_PO[tile_size][tile_size / 16];   // Local memory to store right
ap_uint<512> OUT_buffer_PO[tile_size][tile_size / 16]; // Local Memory to store result

ap_uint<512> temp_sum[tile_size][tile_size / 16];    // Local Memory to store sum of result



#pragma HLS array_partition variable=RHS_buffer_PI dim=2
#pragma HLS array_partition variable=OUT_buffer_PI dim=2

#pragma HLS array_partition variable=RHS_buffer_PO dim=2
#pragma HLS array_partition variable=OUT_buffer_PO dim=2

#pragma HLS array_partition variable=temp_sum dim=2

bool enable_Read = true;
bool enable_Comp = true;
bool enable_Wtemp = true;
bool enable_Wmem = true;

if(MagicFlag == 1){
  enable_Read = true;
  enable_Comp = true;
  enable_Wtemp = true;
  enable_Wmem = true;
}
else if(MagicFlag == 2){
  enable_Read = true;
  enable_Comp = false;
  enable_Wtemp = false;
  enable_Wmem = false;
}
else if(MagicFlag == 3){
  enable_Read = false;
  enable_Comp = true;
  enable_Wtemp = false;
  enable_Wmem = false;
}
else if(MagicFlag == 4){
  enable_Read = false;
  enable_Comp = false;
  enable_Wtemp = true;
  enable_Wmem = true;
}

Init(enable_Read,LHS_buffer_PI);
Init(enable_Read,RHS_buffer_PI);
Init(enable_Read,OUT_buffer_PI);
Init(enable_Read,LHS_buffer_PO);
Init(enable_Read,RHS_buffer_PO);
Init(enable_Read,OUT_buffer_PO);
Init(enable_Read,temp_sum);
printf("enter kernel\r\n");


for(int stage = 0, I = 0, J = 0, K = 0; stage < (DATA_SIZE / tile_size) * (DATA_SIZE / tile_size) * (DATA_SIZE / tile_size) + 2; stage++){
    printf(">>>>>>>>>>Start stage:%d<<<<<<<<<<<<\n",stage);
    if(stage % 2 == 0){ //stage=0 2 4 6 ... 512
        Read(enable_Read && stage >= 0 && stage <= 510, in0, LHS_buffer_PI, I, K);
        Read(enable_Read && stage >= 0 && stage <= 510, in1, RHS_buffer_PI, K, J);
        Calculate(enable_Comp && stage >= 2 && stage <= 512, LHS_buffer_PO, RHS_buffer_PO, OUT_buffer_PO);
        // if(enable_Comp && stage >= 2 && stage <= 512){
        //     printf("Calculate To [OUT_buffer_PO]\t\n");
        // }
        Write_To_Temp(enable_Wtemp && stage >= 2 && stage <= 512, OUT_buffer_PI, temp_sum);
        // if(enable_Wtemp && stage >= 2 && stage <= 512){
        //     printf("Write [OUT_buffer_PI] To [temp_sum]\t\n");
        // }
    }
    else{                //stage=1 3 5 7 ... 513
        Read(enable_Read && stage >= 1 && stage <= 511, in0, LHS_buffer_PO, I, K);
        Read(enable_Read && stage >= 1 && stage <= 511, in1, RHS_buffer_PO, K, J);
        Calculate(enable_Comp && stage >= 1 && stage <= 511, LHS_buffer_PI, RHS_buffer_PI, OUT_buffer_PI);
        // if(enable_Comp && stage >= 1 && stage <= 511){
        //     printf("Calculate To [OUT_buffer_PI]\t\n");
        // }
        Write_To_Temp(enable_Wtemp && stage >= 3 && stage <= 513, OUT_buffer_PO, temp_sum);
        // if(enable_Wtemp && stage >= 3 && stage <= 513){
        //     printf("Write [OUT_buffer_PO] To [temp_sum]\t\n"); 
        // }


        Write_To_Memory(enable_Wmem && (stage - 1) % (DATA_SIZE / tile_size) == 0 && stage >= 9, out0, temp_sum,
                              ((stage - 1) % ((DATA_SIZE / tile_size ) * (DATA_SIZE / tile_size )) == 0) ? I - tile_size : I, 
                              ((stage - 1) % ((DATA_SIZE / tile_size ) * (DATA_SIZE / tile_size )) == 0) ? DATA_SIZE - tile_size : J - tile_size);

    }
    //compute I K J:
    K += tile_size;
    if(K == DATA_SIZE){
        K = 0;
        J += tile_size;
        if(J == DATA_SIZE){
            J = 0;
            I += tile_size;
        }
    }
}

/*
/////////////////////////////
int stage = 0;
for(int I = 0; I < DATA_SIZE; I = I + tile_size){

  for(int J = 0; J < DATA_SIZE; J = J + tile_size){

    for(int K = 0; K < DATA_SIZE;K = K + tile_size ){

      if(stage == 0){ //stage=0

        Read(enable_Read, in0, LHS_buffer_PI, I, K);
        Read(enable_Read, in1, RHS_buffer_PI, K, J);
        stage++;

      }
      else{
        if(stage == 1){ //stage=1
          Read(enable_Read, in0, LHS_buffer_PO, I, K);
          Read(enable_Read, in1, RHS_buffer_PO, K, J);
          Calculate(enable_Comp, LHS_buffer_PI, RHS_buffer_PI, OUT_buffer_PI);
          stage++;

          
        }
        else{

          if(stage % 2 == 0){ //stage=2 4 6 .. 510

            Read(enable_Read, in0, LHS_buffer_PI, I, K);
            Read(enable_Read, in1, RHS_buffer_PI, K, J);
            Calculate(enable_Comp, LHS_buffer_PO, RHS_buffer_PO, OUT_buffer_PO);
            Write_To_Temp(enable_Wtemp, OUT_buffer_PI, temp_sum);
            stage++;


          }
          else{
            if(stage % 2 == 1){ //stage=3 5 7 .. 511
              Read(enable_Read, in0, LHS_buffer_PO, I, K);
              Read(enable_Read, in1, RHS_buffer_PO, K, J);
              Calculate(enable_Comp, LHS_buffer_PI, RHS_buffer_PI, OUT_buffer_PI);
              Write_To_Temp(enable_Wtemp, OUT_buffer_PO, temp_sum);
              if((stage - 1) % (DATA_SIZE / tile_size) == 0){
                if((stage - 1) % ((DATA_SIZE / tile_size ) * (DATA_SIZE / tile_size )) == 0){
                  Write_To_Memory(enable_Wmem, out0, temp_sum, I - tile_size, DATA_SIZE - tile_size);
                }
                else{
                  Write_To_Memory(enable_Wmem, out0, temp_sum, I, J - tile_size);
                }
              }
              stage++;
              
              
            }
          }
        }
      }
 
    }

  }

}


if(stage == (DATA_SIZE / tile_size) * (DATA_SIZE / tile_size) * (DATA_SIZE / tile_size)){ //stage=512

  Calculate(enable_Comp, LHS_buffer_PO, RHS_buffer_PO, OUT_buffer_PO);
  Write_To_Temp(enable_Wtemp, OUT_buffer_PI, temp_sum);
  stage++;

}

if(stage == (DATA_SIZE / tile_size) * (DATA_SIZE / tile_size) * (DATA_SIZE / tile_size) + 1){ //stage=513

  Write_To_Temp(enable_Wtemp, OUT_buffer_PO, temp_sum);
  Write_To_Memory(enable_Wmem, out0, temp_sum, DATA_SIZE - tile_size, DATA_SIZE - tile_size);
  stage++;

}
*/

}
}



