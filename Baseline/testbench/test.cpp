#include "../src/mm.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#define DATA_SIZE 64
int main(){
    unsigned int* LHS = (unsigned int*) malloc(sizeof(unsigned int) * DATA_SIZE * DATA_SIZE);
    unsigned int* RHS = (unsigned int*) malloc(sizeof(unsigned int) * DATA_SIZE * DATA_SIZE);
    unsigned int* OUT = (unsigned int*) malloc(sizeof(unsigned int) * DATA_SIZE * DATA_SIZE);
    unsigned int* GoldenOUT = (unsigned int*) malloc(sizeof(unsigned int) * DATA_SIZE * DATA_SIZE)


    //generate random data
    srand (time(0));
    for(int i = 0; i < DATA_SIZE * DATA_SIZE; i++){
        LHS[i] = rand()%5;
        RHS[i] = rand()%5;
    }
    for (int i = 0; i < DATA_SIZE; i++){
        for (int j = 0; j < DATA_SIZE; j++){
            GoldenOUT[i * DATA_SIZE + j] = 0;

            for (int k = 0; k < DATA_SIZE; k++){
                GoldenOUT[i * DATA_SIZE + j] += LHS[i * DATA_SIZE + k] * RHS[k * DATA_SIZE + j];
            }
            
        }
    }

    //launch kernel
    matrix_multiply(LHS, RHS, OUT, DATA_SIZE, DATA_SIZE, DATA_SIZE);
    

    //test result data
    int count = 0;
    printf("#######################\n")
    for(int i = 0; i < DATA_SIZE * DATA_SIZE; i++){
        if (OUT[i]!=GoldenOUT[i]){
            count++;
            printf("Error found: %d, %d %d\n", i, OUT[i], GoldenOUT[i]);
        }
    }
    printf("Error Num: %d\n", count);
    printf("#######################\n")

    //free allocated memory
    free(GoldenOUT);
    free(OUT);
    free(LHS);
    free(RHS);



    return 0;
}