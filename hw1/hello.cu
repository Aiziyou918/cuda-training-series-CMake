#include <stdio.h>

__global__ void hello(){
  int blockId = blockIdx.x;
  int threadId = threadIdx.x;
  printf("Hello from block: %u, thread: %u\n", blockId, threadId);
}

int main(){

  hello<<<2, 2>>>();
  cudaDeviceSynchronize();
}

