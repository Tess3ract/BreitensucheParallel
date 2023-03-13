/**
 * compile: n
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

CSR_Format readGraph(string path);
void printGraph(CSR_Format cage15);
void readLine(string line,  int *source, int *destination ,string deli);



__device__ int queueDev[QUEUE_SIZE];
__device__ int head = 0;
__device__ int tail = 0;

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


__device__ void swapQueues(int *s_inQ, int *s_outQ, int &s_counterIn, int &s_counterOut) {
    int tmp[MAX_QUEUE_SIZE];

    memcpy(tmp, s_inQ, s_counterIn * sizeof(int));
    memcpy(s_inQ, s_outQ, s_counterOut * sizeof(int));
    memcpy(s_outQ, tmp, s_counterIn * sizeof(int));
    s_counterIn = s_counterOut;
    s_counterOut = 0;
}

__global__ void breitensucheGPU(int startingNode, const int *C, const int *R, const int rSize, const int cSize, int *distance) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_inQ[MAX_QUEUE_SIZE];
    __shared__ int s_outQ[MAX_QUEUE_SIZE];
    __shared__ int s_counterIn;
    __shared__ int s_counterOut;
    __shared__ int s_iteration;

    if (tid == 0) {
        s_inQ[0] = startingNode;
        s_counterIn = 1;
        distance[startingNode] = 0;
        s_iteration = 0;
    }

    while (true) {
        int items_processed_by_thread = 0;
        for (int i = tid; i < s_counterIn; i += blockDim.x * gridDim.x) {
            const int current_node = s_inQ[i];
            for (int j = R[current_node]; j < R[current_node + 1]; j++) {
                const int new_node = C[j];
                if (distance[new_node] == INT_MAX) {
                    distance[new_node] = distance[current_node] + 1;
                    const int pos = atomicAdd(&s_counterOut, 1);
                    s_outQ[pos] = new_node;
                    items_processed_by_thread++;
                }
            }
        }

        
        
        // Anzahl alle Threads in alle Blocks
        const int threads_in_block = blockDim.x * gridDim.x;

        const int total_threads = threads_in_block * gridDim.y;
        const int threads_to_sync = min(s_counterIn, total_threads);
        //Wenn GPU nicht zu viel Resource hat?????
        const int threads_synced = __syncthreads_count(threads_to_sync);
        if (threads_synced == 0) break;

        if (tid == 0) {
            s_iteration++;
            swapQueues(s_inQ, s_outQ, s_counterIn, s_counterOut);
        }
        __syncthreads();
    }
}



//--------------------------------------------------------------------------------------------
int main() {

    CSR_Format  cage15 = readGraph("Beispielgraph.mtx");
    //CSR_Format  cage15 = readGraph("cage15/cage15.mtx");
    cout<< "The graph has been read successfully\n";
    int startingNode = 0;

    int *dev_R;
    int *dev_C;
    int *dev_Distance;
    

    int blockSize = 1;
    // int threadNum = 15;
    // int N; // number of nodes in the graph


    int sizeInt = sizeof(int);
    int distinationSize = sizeInt*(cage15.rSize-1);
    int rSize = cage15.rSize * sizeInt;
    int cSize = cage15.cSize * sizeInt;
    //device allocation
    cudaMalloc((void**)&dev_R, rSize);
    cudaMalloc((void**)&dev_C, cSize);
    cudaMalloc((void **)&dev_Distance, distinationSize);

    
    //host allocation
    int *distanceCage15 = (int*) malloc(distinationSize);

    for (int j=0; j<=cage15.rSize-1; j++){
        distanceCage15[j] = INT_MAX;
    }
    distanceCage15[startingNode] = 0;

    cudaMemcpy(dev_R, cage15.R, rSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, cage15.C, cSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Distance, distanceCage15, distinationSize, cudaMemcpyHostToDevice);

    printf(":========== %d\n", cage15.rSize);
    breitensucheGPU<<<2, 8>>>(
        startingNode, dev_C, dev_R, cSize, 
        rSize, dev_Distance
    );
    cudaDeviceSynchronize();

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

    return 0;
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
    int *C = (int*)malloc(sizeof(int)*cSize); 
    int cCounter=0;

    int rSize = max+2; // max+1  nodes => size of R is max+2
    int *R = (int*)malloc(sizeof(int)*rSize);       
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





// #define BLOCK_SIZE 256
// #define MAX_NODES_PER_BLOCK 32

// __device__ void enqueue(int node, int* queue, int& queueSize, int* distance)
// {
//     // Use an atomic operation to update the distance array to avoid race conditions
//     atomicMin(&distance[node], distance[queue[queueSize-1]]+1);
//     queue[queueSize++] = node;
// }

// __global__ void breitensucheGPU(int startingNode, int* C, int* R, int C_size, int R_size, int* distance, int* queue)
// {
//     __shared__ int sC[MAX_NODES_PER_BLOCK];
//     __shared__ int sR[MAX_NODES_PER_BLOCK+1];

//     int tIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     int queueSize = 0;
//     if (tIndex == 0) {
//         for (int i = 0; i < R_size-1; i++) {
//             distance[i] = INT_MAX;
//         }
//         distance[startingNode] = 0;
//         queue[queueSize++] = startingNode;
//     }
//     __syncthreads();

//     int sQueueSize = 0;
//     int sQueue[MAX_NODES_PER_BLOCK];
//     sQueue[sQueueSize++] = queue[tIndex];
//     while (sQueueSize > 0) {
//         // Load the current node and its neighbors into shared memory
//         int currentNode = sQueue[--sQueueSize];
//         if (threadIdx.x < MAX_NODES_PER_BLOCK) {
//             sC[threadIdx.x] = C[R[currentNode]+threadIdx.x];
//             if (threadIdx.x == MAX_NODES_PER_BLOCK-1) {
//                 sR[threadIdx.x+1] = R[currentNode+1];
//             }
//         }
//         __syncthreads();

//         // Explore the neighbors of the current node
//         for (int i = 0; i < sR[threadIdx.x+1]-sR[threadIdx.x]; i++) {
//             int neighbor = sC[i];
//             if (distance[neighbor] == INT_MAX) {
//                 enqueue(neighbor, queue, queueSize, distance);
//             }
//         }
//         __syncthreads();
//     }
// }

// __host__ void breitensucheCUDA(int startingNode, int* C, int* R, int C_size, int R_size, int* distance, int* queue)
// {
//     int numBlocks = (R_size-1 + BLOCK_SIZE-1) / BLOCK_SIZE;
//     breitensucheGPU<<<numBlocks, BLOCK_SIZE>>>(startingNode, C, R, C_size, R_size, distance, queue);
//     cudaDeviceSynchronize();
// }


// __host__ void breitensucheCUDA(
//     int startingNode, const int *C,
//         const int *R, int rSize, 
//         const int cSize, int *distance
//     )
// {

//     const int MAX_THREADS_PER_BLOCK = 1024;
//     int numThreads = min(rSize, MAX_THREADS_PER_BLOCK);
//     int numBlocks = (rSize-1 + numThreads-1) / numThreads;

//     breitensucheGPU<<<numBlocks, numThreads>>>(startingNode, C, R, 
//         rSize, cSize, distance
//     );
//     cudaDeviceSynchronize();
// }