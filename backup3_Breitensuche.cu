/**
 * compile: nvcc Breintensuche.cu
 * 
*/




#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <queue>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include<unistd.h>


#define MAX_QUEUE_SIZE  25
#define BLOCK_SIZE 256
#define MAX_NODES_PER_BLOCK 32
#define QUEUE_SIZE 256

using namespace std;

class CSR_Format {
    public:
    int *C;
    int cSize;
    int *R;
    int rSize;
};

void run();
CSR_Format readGraph(string path);
void printGraph(CSR_Format cage15);
void checkCudaError(cudaError_t error, const char* message);
void readLine(string line,  int *source, int *destination ,string deli);
void swapQueues(int **d_inQ, int **d_outQ, int *s_counterIn, int *s_counterOut);
void breitensucheCUDA(int startingNode, const int *C, const int *R, int rSize,  const int cSize, int *distance);


__device__ int dequeue();
__device__ void enqueue(int value);
__global__ void breitensucheGPU(
        int startingNode, const int *C,
        const int *R, const int rSize, 
        const int cSize, int *distance,
        int *s_counterIn, int *s_counterOut,
        int *s_inQ, int *s_outQ
    );


__device__ int queueDev[QUEUE_SIZE];
__device__ int head = 0;
__device__ int tail = 0;


//--------------------------------------------------------------------------------------------
int main() {

    run();

    return 0;
}


void run(){

    CSR_Format  cage15 = readGraph("Beispielgraph.mtx");
    //CSR_Format  cage15 = readGraph("cage15/cage15.mtx");
    cout<< "The graph has been read successfully\n";

    int startingNode = 0;   
    int sizeInt = sizeof(int);
    int distinationSize = sizeInt*(cage15.rSize-1);
    int rSize = cage15.rSize * sizeInt;
    int cSize = cage15.cSize * sizeInt;

    int *dev_R;
    int *dev_C;
    int *dev_Distance;
    
    //device allocation
    cudaMalloc((void**)&dev_R, rSize);
    cudaMalloc((void**)&dev_C, cSize);
    cudaMalloc((void **)&dev_Distance, distinationSize);

    
    //host allocation
    int *distanceCage15 = (int*) malloc(distinationSize);
    fill(distanceCage15, distanceCage15 + cage15.rSize, INT_MAX);
    distanceCage15[startingNode] = 0;

    //
    cudaMemcpy(dev_R, cage15.R, rSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, cage15.C, cSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Distance, distanceCage15, distinationSize, cudaMemcpyHostToDevice);


    breitensucheCUDA(
        startingNode, dev_C, dev_R, cSize, 
        rSize, dev_Distance
    );

    cudaMemcpy(distanceCage15, dev_Distance, distinationSize, cudaMemcpyDeviceToHost);

    for (int i=0; i< cage15.rSize-1; ++i){
        printf("distance %d = %d\n", i, distanceCage15[i]);
    }
    printf("\n");

    // printGraph(cage15);

    free(distanceCage15);

    cudaFree(dev_C);
    cudaFree(dev_R);
    cudaFree(dev_Distance);

}




void breitensucheCUDA(
    int startingNode, const int *C,
    const int *R, int rSize, 
    const int cSize, int *distance
){
   
    const int MAX_THREADS_PER_BLOCK = 1024;
    int numThreads = 0;
    int numBlocks = 0;
    
    int *s_inQ;  
    int *s_outQ;
     
    int s_counterIn = 1;
    int s_counterOut = 0;
    int sizeInt = sizeof(int);

    s_inQ = (int*) malloc(MAX_QUEUE_SIZE * sizeInt);
    s_outQ = (int*) malloc(MAX_QUEUE_SIZE * sizeInt);
    s_inQ[0] = startingNode;


    int *d_counterIn, *d_counterOut;
    int *d_inQ, *d_outQ;

    checkCudaError(cudaMalloc(&d_counterIn, sizeInt), "cudaMalloc d_counterIn failed");
    checkCudaError(cudaMalloc(&d_counterOut, sizeInt), "cudaMalloc d_counterOut failed");
    checkCudaError(cudaMalloc(&d_inQ, MAX_QUEUE_SIZE * sizeInt), "cudaMalloc d_inQ failed");
    checkCudaError(cudaMalloc(&d_outQ, MAX_QUEUE_SIZE * sizeInt), "cudaMalloc d_outQ failed");

    //kann nur einmal kopiert werden
    checkCudaError(cudaMemcpy(d_inQ, s_inQ, MAX_QUEUE_SIZE * sizeInt, cudaMemcpyHostToDevice), 
        "cudaMemcpy s_inQ => d_inQ faild"
    );

    while (s_counterIn > 0) {
        numThreads = min(s_counterIn, 4);
        numBlocks = (s_counterIn + numThreads-1) / numThreads;
        
        //sollte geguckt werden, ob man s_countIn übergeben
        cudaMemcpy(d_counterIn, &s_counterIn, sizeInt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_counterOut, &s_counterOut, sizeInt, cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_outQ, s_outQ, MAX_QUEUE_SIZE * sizeInt, cudaMemcpyHostToDevice);
        
        if (numBlocks == 0){
            numBlocks++;
        }
        breitensucheGPU<<<numBlocks, numThreads>>>(startingNode, C, R, 
            rSize, cSize, distance, d_counterIn, d_counterOut, d_inQ, d_outQ
        );
        cudaDeviceSynchronize();
    
        cudaMemcpy(&s_counterIn, d_counterIn, sizeInt, cudaMemcpyDeviceToHost);
        cudaMemcpy(&s_counterOut, d_counterOut, sizeInt, cudaMemcpyDeviceToHost);
        

        swapQueues(&d_inQ, &d_outQ, &s_counterIn, &s_counterOut);
        
        //sleep(1);
    }

    cudaFree(d_counterIn);
    cudaFree(d_counterOut);
    cudaFree(d_inQ);
    cudaFree(d_outQ);
}


__global__ void breitensucheGPU(
    int startingNode, const int *C,
        const int *R, const int rSize, const int cSize, int *distance,
        int *s_counterIn, int *s_counterOut, int *s_inQ, int *s_outQ
    ){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < *s_counterIn) {
        int current_node = s_inQ[tid];

        for (int i = R[current_node]; i < R[current_node + 1]; i++) {   
            int new_node = C[i];
            
            if (distance[new_node] == INT_MAX) {
                distance[new_node] = distance[current_node] + 1;                
                int pos = atomicAdd(s_counterOut, 1);                
                s_outQ[pos] = new_node;
            }            
        }
    }
    __syncthreads();
}




void printGraph(CSR_Format cage15){
    printf("\n");

    for (int i =0; i< cage15.rSize-1; i++){
        printf("%d = ",i);
        for(int j=cage15.R[i]; j<cage15.R[i+1]; j++){
            printf("%d  ", cage15.C[j]);
        }        
        printf("\n");
    }
    printf("\n");
}



//read current line and store it in source and destination
void readLine(string line,  int *source, int *destination ,string deli = " ")
{
    int start = 0;
    int end = line.find(deli);
    *destination = stoi(line.substr(start, end - start));
    start = end + deli.size();
    end = line.find(deli, start);
    *source = stoi(line.substr(start, end - start));
}



//read graph in Matrix Market format
// format: <destination> <source> <weight>
// lines have to be sorted by source ascending
CSR_Format readGraph(string path) {
    ifstream Preparation(path);
    string line;
    int lineCounter = 0;
    int max=0;
    int source;
    int destination;
    int sizeInt = sizeof(int);

    //get number of nodes and edges
    while(getline(Preparation,line)) {
        if (line[0] =='%') {
            continue;
        }
        lineCounter++;
        readLine(line, &source, &destination);
        if(max < source) {max=source;}
        if(max < destination) {max=destination;}         
    }

    //preparation
    int cSize = lineCounter; //number of lines = number of edges = size of C
    int *C = (int*)malloc(sizeInt*cSize); 
    int cCounter=0;

    int rSize = max+2; // max+1  nodes => size of R is max+2
    int *R = (int*)malloc(sizeInt*rSize);       
    int rCounter=0;
    R[rCounter]=0;
    rCounter++;
    ifstream Graph(path);
    int lastSource=0;
    lineCounter = 0;

    //main part
    //for each line
    while(getline(Graph,line)) {
        if (line[0] =='%') {
            continue;
        }
        //read current line and store it in source and destination
        readLine(line,&source,&destination);

        //transform into CSR format
        while(lastSource<source) {
            R[rCounter] = lineCounter;
            rCounter++;
            lastSource++;
        }
        //here is lastSource == source, R up to date
        C[cCounter] = destination;
        cCounter++;


        lineCounter++;
    }
    CSR_Format result = CSR_Format();
    result.C=C;
    result.cSize=cSize;
    result.R=R;
    result.rSize=rSize; 
    return result;
}


void swapQueues(int **d_inQ, int **d_outQ, int *s_counterIn, int *s_counterOut) {
    
    int *temp;
    temp = *d_outQ;
    *d_outQ = *d_inQ;
    *d_inQ = temp;

    *s_counterIn = *s_counterOut;
    *s_counterOut = 0;   
}

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


__device__ void enqueue(int value) {
    int next_tail = (tail + 1) % QUEUE_SIZE;
    if (next_tail == head) {
        // Queue is full, do something (e.g., return an error)
    } else {
        queueDev[tail] = value;
        tail = next_tail;
    }
}

__device__ int dequeue() {
    if (head == tail) {
        // Queue is empty, do something (e.g., return an error)
        return -1;
    } else {
        int value = queueDev[head];
        head = (head + 1) % QUEUE_SIZE;
        return value;
    }
}