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
#include <memory>


#define MAX_QUEUE_SIZE  1000000
#define BLOCK_SIZE 256
#define MAX_NODES_PER_BLOCK 32
#define QUEUE_SIZE 256
#define MAX_THREADS_PER_BLOCK 1024

using namespace std;

class CSR_Format {
    public:
    int *C;
    int cSize;
    int *R;
    int rSize;
};

typedef struct {
    long unsigned rSize;
    long unsigned cSize;
    int *dev_R;
    int *dev_C;
    int *dev_Distance;
} GraphData;


struct QueueData {
    int *inQ;
    int *outQ;
    int *counterIn;
    int *counterOut;
};

void run();
CSR_Format readGraph(string path);
void printGraph(CSR_Format cage15);
void printDistance(const int *distances, const int size);
void checkCudaError(cudaError_t error, const char* message);
void readLine(string line,  int *source, int *destination ,string deli);
void swapQueues(int **d_inQ, int **d_outQ, int *s_counterIn, int *s_counterOut);
void breitensucheCUDA(int startingNode, GraphData graphData, QueueData queueData, int sizeInt);

__device__ int dequeue();
__device__ void enqueue(int value);
__global__ void breitensucheGPU(
        int startingNode, GraphData graphData, QueueData queueData);


__device__ int queueDev[QUEUE_SIZE];
__device__ int head = 0;
__device__ int tail = 0;


//--------------------------------------------------------------------------------------------
int main() {

    run();

    return 0;
}

void allocateDeviceMemory(CSR_Format graph, GraphData& graphData, QueueData& queueData) {
    const int sizeInt = sizeof(int);
    const int distinationSize = sizeInt * (graph.rSize - 1);
    const int maxQueueSize = MAX_QUEUE_SIZE;

    // Allocate device memory for the graph data
    cudaMalloc((void**)&graphData.dev_R, graph.rSize * sizeInt);
    cudaMalloc((void**)&graphData.dev_C, graph.cSize * sizeInt);
    cudaMalloc((void**)&graphData.dev_Distance, distinationSize);

    // Allocate device memory for the queue data
    checkCudaError(cudaMalloc(&queueData.inQ, maxQueueSize * sizeInt), "cudaMalloc d_inQ failed");
    checkCudaError(cudaMalloc(&queueData.outQ, maxQueueSize * sizeInt), "cudaMalloc d_outQ failed");
    checkCudaError(cudaMalloc(&queueData.counterIn, sizeInt), "cudaMalloc d_counterIn failed");
    checkCudaError(cudaMalloc(&queueData.counterOut, sizeInt), "cudaMalloc d_counterOut failed");

}

void copyDataHostToDevice(GraphData& graphData, const CSR_Format& csrFormat, const unique_ptr<int[]>& distanceCage15) {
    const int sizeInt = sizeof(int);
    const int distinationSize = sizeInt * (csrFormat.rSize - 1);

    cudaMemcpyAsync(graphData.dev_R, csrFormat.R, csrFormat.rSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(graphData.dev_C, csrFormat.C, csrFormat.cSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(graphData.dev_Distance, distanceCage15.get(), distinationSize, cudaMemcpyHostToDevice);
}


void copyDataDeviceToHost(unique_ptr<int[]>& distanceCage15, const GraphData& graphData, const CSR_Format& csrFormat) {
    const int sizeInt = sizeof(int);
    const int distinationSize = sizeInt * (csrFormat.rSize - 1);

    cudaMemcpyAsync(distanceCage15.get(), graphData.dev_Distance, distinationSize, cudaMemcpyDeviceToHost);
}

GraphData allocateGraphData(CSR_Format &graph, int sizeInt) {
    GraphData data;
    data.rSize = graph.rSize * sizeInt;
    data.cSize = graph.cSize * sizeInt;
    cudaMalloc((void**)&data.dev_R, data.rSize);
    cudaMalloc((void**)&data.dev_C, data.cSize);
    cudaMalloc((void**)&data.dev_Distance, (graph.rSize - 1) * sizeInt);
    return data;
}

QueueData allocateQueueData(int queueSize, int sizeInt){
    QueueData devQueueData;
    // Allocate device memory for the queue data
    checkCudaError(cudaMalloc(&devQueueData.inQ, queueSize * sizeInt), "cudaMalloc d_inQ failed");
    checkCudaError(cudaMalloc(&devQueueData.outQ, queueSize * sizeInt), "cudaMalloc d_outQ failed");
    checkCudaError(cudaMalloc(&devQueueData.counterIn, sizeInt), "cudaMalloc d_counterIn failed");
    checkCudaError(cudaMalloc(&devQueueData.counterOut, sizeInt), "cudaMalloc d_counterOut failed");
    return devQueueData;
}

void freeGraphData(GraphData &graphData){
    cudaFree(graphData.dev_C);
    cudaFree(graphData.dev_R);
    cudaFree(graphData.dev_Distance);
}

void freeQueueData(QueueData &queueData){
    cudaFree(queueData.counterIn);
    cudaFree(queueData.counterOut);
    cudaFree(queueData.inQ);
    cudaFree(queueData.outQ);
}

int *initializeStartQueue(int startingNode, int sizeInt) {
    int *s_inQ = (int*)malloc(MAX_QUEUE_SIZE * sizeInt);
    s_inQ[0] = startingNode;

    return s_inQ;
}

void run(){

    CSR_Format  cage15 = readGraph("Beispielgraph.mtx");
    // CSR_Format  cage15 = readGraph("cage15/cage15.mtx");
    cout<< "The graph has been read successfully\n";

    const int startingNode = 0;   
    const int sizeInt = sizeof(int);

    GraphData graphData = allocateGraphData(cage15, sizeInt);
    QueueData devQueueData = allocateQueueData(MAX_QUEUE_SIZE, sizeInt);
    
    
    //host allocation
    unique_ptr<int[]> distanceCage15(new int[cage15.rSize - 1]);
    fill(distanceCage15.get(), distanceCage15.get() + cage15.rSize - 1, INT_MAX);
    distanceCage15[startingNode] = 0;


    copyDataHostToDevice(graphData, cage15, distanceCage15);


    breitensucheCUDA(startingNode, graphData, devQueueData, sizeInt);

    copyDataDeviceToHost(distanceCage15, graphData, cage15);

    printDistance(distanceCage15.get(), 10);

    freeGraphData(graphData);
    freeQueueData(devQueueData);
    

}


void breitensucheCUDA(int startingNode, GraphData graphData, QueueData queueData, int sizeInt){

    int numThreads = 0;
    int numBlocks = 0;
    int s_counterIn = 1;
    int s_counterOut = 0;

    int *s_inQ = initializeStartQueue(startingNode, sizeInt);  
    int *s_outQ;
     

    s_outQ = (int*) malloc(MAX_QUEUE_SIZE * sizeInt);

    checkCudaError(cudaMemcpyAsync(queueData.inQ, s_inQ, MAX_QUEUE_SIZE * sizeInt, cudaMemcpyHostToDevice), 
        "cudaMemcpy s_inQ => d_inQ faild"
    );

    while (s_counterIn > 0) {
        numThreads = min(s_counterIn, MAX_THREADS_PER_BLOCK);
        numBlocks = (s_counterIn + numThreads-1) / numThreads;
        
        //sollte geguckt werden, ob man s_countIn übergeben
        cudaMemcpyAsync(queueData.counterIn, &s_counterIn, sizeInt, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(queueData.counterOut, &s_counterOut, sizeInt, cudaMemcpyHostToDevice);
        
        // cudaMemcpyAsync(queueData.outQ, s_outQ, MAX_QUEUE_SIZE * sizeInt, cudaMemcpyHostToDevice);
        
        if (numBlocks == 0){
            numBlocks++;
        }
        breitensucheGPU<<<numBlocks, numThreads>>>(startingNode, graphData, queueData);
        cudaDeviceSynchronize();
    
        cudaMemcpyAsync(&s_counterIn, queueData.counterIn, sizeInt, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(&s_counterOut, queueData.counterOut, sizeInt, cudaMemcpyDeviceToHost);
        

        swapQueues(&queueData.inQ, &queueData.outQ, &s_counterIn, &s_counterOut);
        
        //sleep(1);
    }

}


__global__ void breitensucheGPU(
    int startingNode, GraphData graphData, QueueData queueData ){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < *queueData.counterIn) {
        int current_node = queueData.inQ[tid];

        for (int i = graphData.dev_R[current_node]; i < graphData.dev_R[current_node + 1]; i++) {   
            int new_node = graphData.dev_C[i];
            
            if (graphData.dev_Distance[new_node] == INT_MAX) {
                graphData.dev_Distance[new_node] = graphData.dev_Distance[current_node] + 1;                
                int pos = atomicAdd(queueData.counterOut, 1);                
                queueData.outQ[pos] = new_node;
            }            
        }
    }
    __syncthreads();
}


void printDistance(const int *distances, const int size){
    for (int i=0; i< size; ++i){
        printf("distance %d = %d\n", i, distances[i]);
    }
    printf("\n");
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