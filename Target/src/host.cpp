
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <math.h>
#include <string>



// This is used for the PL Kernels
#include "xrt/experimental/xrt_kernel.h"


using namespace std;


const int M=1024;
const int K=1024;
const int N=1024;


int main(int argc, char** argv) {



    int iter = 1;
    int verify = 1;
    int MagicFlag = 1;

    char* xclbinFilename;
    xclbinFilename=argv[1];

    if(argc == 5) {
        
        if (sscanf (argv[2], "%i", &iter) != 1) {
            fprintf(stderr, "error - not an integer");
        }
        if (sscanf (argv[3], "%i", &verify) != 1) {
            fprintf(stderr, "error - not an integer");
        }
        if (sscanf (argv[4], "%i", &MagicFlag) != 1) {
            fprintf(stderr, "error - not an integer");
        }
    }





    //////////////////////////////////////////
    //Open xclbin
    //Open a Device
    //Load compiled kernel binary (or XCLBIN) onto the device
    //////////////////////////////////////////
    
    unsigned int dev_index = 0;
    auto device = xrt::device(dev_index); //device index=0
	auto uuid = device.load_xclbin(xclbinFilename);
    auto dhdl = xrtDeviceOpenFromXcl(device);
     if(!dhdl){
	printf("Device open error\n");
    }
    std::cout << "Device open successful\n";
    auto matrix_multiply = xrt::kernel(device, uuid, "matrix_multiply");



    std::vector<float> DataInput0(M * K, 1);
    std::vector<float> DataInput1(K * N, 1);
    std::vector<float> final_result_sw(M * N, 1);

    srand (time(0));
   for(int i=0; i < M; i++) {
        for(int j=0; j < K; j++){
            DataInput0[i * K + j] = (float)(rand()%20);
        }
   }
    std::cout << "chech if it is random.\n";
    for(int i = 0; i < 10; i++){
        std::cout << DataInput0[i] << "||";
    }
    std::cout  << endl;

    for(int i=0; i < K; i++) {
        for(int j=0; j < N; j++){
            DataInput1[i * N + j] = (float)(rand()%20);
        }
   }
    std::cout << "srand generate successful\n";

    //Allocate input mem
    auto in_bohdl0 = xrt::bo(device, M * K * sizeof(float), matrix_multiply.group_id(0));
    auto in_bomapped0 = in_bohdl0.map<float*>();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++){
            in_bomapped0 [i * K + j]= DataInput0[i * K + j];
        }  
    } 
    std::cout << "in_bomapped0 successful\n";

    auto in_bohdl1 = xrt::bo(device, K * N * sizeof(float), matrix_multiply.group_id(0));
    auto in_bomapped1 = in_bohdl1.map<float*>();
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++){
            in_bomapped1 [i * N + j]= DataInput1[i * N + j];
        }
    }
    std::cout << "in_bomapped1 successful\n";
    // sync input memory
    in_bohdl0.sync(XCL_BO_SYNC_BO_TO_DEVICE , M * K * sizeof(float),0);
    in_bohdl1.sync(XCL_BO_SYNC_BO_TO_DEVICE , K * N * sizeof(float),0);
    std::cout << "sync input memory successful\n";

    //Allocate output buffer
    auto out_bohdl = xrt::bo(device, M * N * sizeof(float), matrix_multiply.group_id(0));
    auto out_bomapped = out_bohdl.map<float*>();
    std::cout << "Allocate output buffer successful\n";

    std::cout << "Kernel run\n";
    xrt::run mm_run0;
    //
    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);
    auto kernel_start = std::chrono::high_resolution_clock::now();
    
    for (int i=0;i<iter;i++){
        // start input kernels run handles
        mm_run0 = matrix_multiply(in_bohdl0, in_bohdl1, out_bohdl, M, K, N, MagicFlag);
        

        mm_run0.wait();
    }
    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    kernel_time_in_sec = kernel_time.count();

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Total time is: "<< kernel_time_in_sec / iter << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;


    // sync output memory
    out_bohdl.sync(XCL_BO_SYNC_BO_FROM_DEVICE , M * N * sizeof(float),/*OFFSET=*/ 0);

    std::cout << "sync output memory successful\n";
    ////////////////////////////////////////////
    //// Comparing the execution data to the golden data
    ////////////////////////////////////////////
    
    if(verify){
        std::cout << "Begin to calculate sw result.\n";
        for (int i=0; i < M; i++) {
            for (int j=0; j < N; j++){
                
                float result = 0;
                for (int k=0; k < K; k++){

                    result += in_bomapped0[i * K + k] * in_bomapped1[k * N + j];
                    final_result_sw[i * N + j] = result;
                }
            }
        }
        std::cout << "calculate sw result successful\n";

        int errorCount = 0;  
        for (int row = 0; row < M; row++) {
            for (int column = 0; column < N; column++){
                if(!(abs(out_bomapped[row * N + column] - final_result_sw[row * N + column]) < 0.00001) ){
                    printf("Error found out_bomapped[%d][%d]!=final_result_sw[%d][%d], %g!=%g \n", row,column,row,column,out_bomapped[row * N + column],final_result_sw[row * N + column]);
                    errorCount++;
                }
            }
        }
        if (errorCount)
            printf("Test failed with %d errors\n", errorCount);
        else
            printf("TEST PASSED\n");
    }


    
    //////////////////////////////////////////
    // clean up XRT
    //////////////////////////////////////////

    std::cout << "Releasing remaining XRT objects...\n";
    
    return 0;

}
