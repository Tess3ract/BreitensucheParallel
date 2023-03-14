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
void checkHostAllocation(void *ptr, const char* ptrName);
void readLine(string line,  int *source, int *destination ,string deli);
void swapQueues(int **d_inQ, int **d_outQ, int *s_counterIn, int *s_counterOut);
void breitensucheCUDA(int startingNode, GraphData graphData, QueueData queueData, int sizeInt);

void freeQueueData(QueueData &queueData);
void freeGraphData(GraphData &graphData);
QueueData allocateQueueData(int queueSize, int sizeInt);
GraphData allocateGraphData(CSR_Format &graph, int sizeInt);
void copyDataDeviceToHost(unique_ptr<int[]>& distanceCage15, const GraphData& graphData, const CSR_Format& csrFormat);
void copyDataHostToDevice(GraphData& graphData, const CSR_Format& csrFormat, const unique_ptr<int[]>& distanceCage15);


__global__ void breitensucheGPU(int startingNode, GraphData graphData, QueueData queueData);
__device__ int queueDev[QUEUE_SIZE];
__device__ int head = 0;
__device__ int tail = 0;


//--------------------------------------------------------------------------------------------
int main() {

    run();

    return 0;
}


void run(){

    // CSR_Format  cage15 = readGraph("Beispielgraph.mtx");
    CSR_Format  cage15 = readGraph("cage15/cage15.mtx");
    cout<< "The graph has been read successfully\n";

    const int startingNode = 1;   
    const int sizeInt = sizeof(int);

    GraphData graphData = allocateGraphData(cage15, sizeInt);
    QueueData devQueueData = allocateQueueData(MAX_QUEUE_SIZE, sizeInt);
    
    //host allocation
    unique_ptr<int[]> distanceCage15(new int[cage15.rSize - 1]);
    fill(distanceCage15.get(), distanceCage15.get() + cage15.rSize - 1, INT_MAX);
    distanceCage15[startingNode] = 0;


    copyDataHostToDevice(graphData, cage15, distanceCage15);


    auto start = chrono::high_resolution_clock::now(); //save start time
    breitensucheCUDA(startingNode, graphData, devQueueData, sizeInt);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    double duration_mili = duration.count()/60000.0;
    cout <<  "Die Laufzeit der Funktion ist " << duration_mili/1000.0 << " Mikrosekunden.\n";

    copyDataDeviceToHost(distanceCage15, graphData, cage15);

    printDistance(distanceCage15.get(), 15);

    freeGraphData(graphData);
    freeQueueData(devQueueData);
    

}

void breitensucheCUDA(int startingNode, GraphData graphData, QueueData queueData, int sizeInt){

    int numThreads = 0;
    int numBlocks = 0;
    
    int *dev_inQ = queueData.inQ;
    // Kopiert den Startknoten in die dev_inQ auf das Gerät. dev_inQ[0] = sartingNode
    checkCudaError(cudaMemcpyAsync(dev_inQ, &startingNode, sizeInt, cudaMemcpyHostToDevice), 
        "cudaMemcpy startingNode => d_inQ failed"
    );

    // Setzt den Eingangs- und Ausgangszähler
    *queueData.counterIn = 1;
    *queueData.counterOut = 0;

    // Solange es noch Knoten in der Eingangswarteschlange gibt
    while (*queueData.counterIn > 0) {
        // Berechnet die Anzahl der Threads und Blöcke
        numThreads = min(*queueData.counterIn, MAX_THREADS_PER_BLOCK);
        numBlocks = (*queueData.counterIn + numThreads-1) / numThreads;

        // Führt die Breitensuche auf dem Gerät aus
        breitensucheGPU<<<numBlocks, numThreads>>>(startingNode, graphData, queueData);
        // Synchronisiert die Geräteausführung
        checkCudaError(cudaDeviceSynchronize(), "CUDA kernel failed to synchronize");

        // Vertauscht die Eingangs- und Ausgangswarteschlange sowie setzt die counterIn und counterOut
        swapQueues(&queueData.inQ, &queueData.outQ, queueData.counterIn, queueData.counterOut);
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


void copyDataHostToDevice(GraphData& graphData, const CSR_Format& csrFormat, const unique_ptr<int[]>& distanceCage15) {
    const int sizeInt = sizeof(int);
    const int distinationSize = sizeInt * (csrFormat.rSize - 1);

    cudaMemcpyAsync(graphData.dev_R, csrFormat.R, csrFormat.rSize * sizeInt, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(graphData.dev_C, csrFormat.C, csrFormat.cSize * sizeInt, cudaMemcpyHostToDevice);
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
    checkCudaError(cudaMalloc((void**)&data.dev_R, data.rSize), "cudaMalloc data.dev_R faild");
    checkCudaError(cudaMalloc((void**)&data.dev_C, data.cSize), "cudaMalloc data.dev_C faild");
    checkCudaError(cudaMalloc((void**)&data.dev_Distance, (graph.rSize - 1) * sizeInt), "cudaMalloc data.dev_Distance faild");    
    return data;
}

QueueData allocateQueueData(int queueSize, int sizeInt){
    QueueData devQueueData;
    // Allocate device memory for the queue data
    checkCudaError(cudaMalloc(&devQueueData.inQ, queueSize * sizeInt), "cudaMalloc d_inQ failed");
    checkCudaError(cudaMalloc(&devQueueData.outQ, queueSize * sizeInt), "cudaMalloc d_outQ failed");

    checkCudaError(cudaMallocManaged(&devQueueData.counterIn, sizeInt), "cudaMalloc d_counterIn failed");
    checkCudaError(cudaMemset(devQueueData.counterIn, 0, sizeInt), "cudaMemset devQueueData.counterIn failed");
    checkCudaError(cudaMallocManaged(&devQueueData.counterOut, sizeInt), "cudaMalloc d_counterOut failed");
    checkCudaError(cudaMemset(devQueueData.counterOut, 0, sizeInt), "cudaMemset devQueueData.counterOut failed");
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

void checkHostAllocation(void *ptr, const char* ptrName){
    if (!ptr) {
        fprintf(stderr, "Error: host memory allocation failed for %s.\n", ptrName);
        exit(EXIT_FAILURE);
    }
}