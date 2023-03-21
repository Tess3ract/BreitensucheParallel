/**
 * @brief CUDA-Breitensuche auf einem Graphen
 *
 * Diese Implementierung führt eine Breitensuche auf einem Graphen durch, der in der Matrix-Markt-Dateiformat gespeichert ist.
 * Die Implementierung verwendet CUDA und führt die Breitensuche auf der GPU aus.
 * Die Distanz jedes Knotens zum Startknoten wird in einem Array gespeichert, welches am Ende ausgegeben wird.
 *
 * @author Moritz Niederer und Sayed Mustafa Sajadi
 * @date 24.03.2023
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
#define MAX_THREADS_PER_BLOCK 2

using namespace std;

// Definition einer Klasse zur Verwendung des CSR-Formats.
class CSR_Format {
    public:
    int *C;
    int cSize;
    int *R;
    int rSize;
};

// Definition einer Struktur zur Verwendung der Daten auf der GPU.
typedef struct {
    long unsigned rSize;
    long unsigned cSize;
    int *dev_R;
    int *dev_C;
    int *dev_Distance;
} GraphData;

// Definition einer Struktur zur Verwendung der Warteschlangendaten auf der GPU.
struct QueueData {
    int *inQ;
    int *outQ;
    int *counterIn;
    int *counterOut;
};

// Funktionsprototypen.
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


// CUDA-Kernel zum Durchführen der Breitensuche.
__global__ void breitensucheGPU(int startingNode, GraphData graphData, QueueData queueData);


//--------------------------------------------------------------------------------------------
int main() {

    run();

    return 0;
}


/**
 * @brief Liest einen Graphen aus einer Datei ein, initialisiert die benötigten Datenstrukturen 
 *        und führt eine Breitensuche auf dem Graphen unter Verwendung von CUDA durch.
 * 
 * Die Funktion liest einen Graphen aus einer Matrix Market-Datei ein, initialisiert benötigte Datenstrukturen 
 * und führt eine Breitensuche auf dem Graphen unter Verwendung von CUDA durch. Die Breitensuche wird solange 
 * durchgeführt, bis alle erreichbaren Knoten vom Startknoten aus besucht wurden. Die Distanz jedes Knotens zum 
 * Startknoten wird in einem Array gespeichert, welches am Ende ausgegeben wird.
 * 
 * @return void
 */
void run(){

    // Lese den Graphen aus einer Datei in CSR_Format ein.
    // CSR_Format  cage15 = readGraph("Beispielgraph.mtx");
    CSR_Format  cage15 = readGraph("cage15/cage15.mtx");
    cout<< "The graph has been read successfully\n";

    // Startknoten der Breitensuche festlegen.
    const int startingNode = 1;   
    const int sizeInt = sizeof(int);

    // Reserviere Speicher für Graphdaten und Warteschlangendaten auf der GPU.
    GraphData graphData = allocateGraphData(cage15, sizeInt);
    QueueData devQueueData = allocateQueueData(MAX_QUEUE_SIZE, sizeInt);
    
    // Reserviere Speicher für die Distanzinformationen auf dem Host.
    unique_ptr<int[]> distanceCage15(new int[cage15.rSize - 1]);
    // Setze alle Distanzen auf "unendlich" (INT_MAX).
    fill(distanceCage15.get(), distanceCage15.get() + cage15.rSize - 1, INT_MAX);
    // Setze die Distanz zum Startknoten auf 0.
    distanceCage15[startingNode] = 0;

    // Kopiere alle relevanten Daten vom Host auf die GPU.
    copyDataHostToDevice(graphData, cage15, distanceCage15);

    // Starte die Zeitmessung.
    auto start = chrono::high_resolution_clock::now(); //save start time

    // Führe die Breitensuche auf der GPU aus.
    breitensucheCUDA(startingNode, graphData, devQueueData, sizeInt);

     // Stoppe die Zeitmessung.
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    double duration_mili = duration.count()/60000.0;
    cout <<  "Die Laufzeit der Funktion ist " << duration_mili/1000.0 << " Mikrosekunden.\n";

    // Kopiere die Distanzinformationen von der GPU zurück auf den Host.
    copyDataDeviceToHost(distanceCage15, graphData, cage15);

    // Gib die Distanzinformationen aus.
    printDistance(distanceCage15.get(), 15);

    // Gib den Speicher auf der GPU frei.
    freeGraphData(graphData);
    freeQueueData(devQueueData);
    

}

/**
 * @brief Führt eine Breitensuche auf dem gegebenen Graphen unter Verwendung der CUDA-Technologie durch.
 * 
 * Die Funktion verwendet eine Queue, um die Knoten zu verfolgen, die während der Breitensuche besucht werden sollen.
 * Die Breitensuche wird iterativ durchgeführt, wobei in jeder Iteration die Knoten bearbeitet werden, die sich in der
 * Eingangs-Queue befinden. Die Knoten, die von einem besuchten Knoten erreicht werden können, werden zur Ausgangs-Queue
 * hinzugefügt, um in der nächsten Iteration verarbeitet zu werden.
 *
 * @param startingNode Startknoten für die Breitensuche.
 * @param graphData Die Struktur, die die Graphendaten auf dem Gerät enthält.
 * @param queueData Die Struktur, die die Queue-Daten auf dem Gerät enthält.
 * @param sizeInt Die Größe eines int-Datentyps in Byte.
 */
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

/**
 * @brief Eine CUDA-Kernel-Funktion, die die Breitensuche auf einer Teilmenge des Graphen durchführt.
 * 
 * Die Funktion wird von der breitensucheCUDA-Funktion aufgerufen, um die Breitensuche auf einer Teilmenge des Graphen
 * durchzuführen. Jeder Thread bearbeitet einen Knoten, der sich in der Eingangs-Queue befindet, und fügt seine nicht
 * besuchten Nachbarn zur Ausgangs-Queue hinzu.
 * 
 * @param startingNode Startknoten für die Breitensuche.
 * @param graphData Die Struktur, die die Graphendaten auf dem Gerät enthält.
 * @param queueData Die Struktur, die die Queue-Daten auf dem Gerät enthält.
 */
__global__ void breitensucheGPU(
    int startingNode, GraphData graphData, QueueData queueData )
    {   
        
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

/**
 * Kopiert Graphdaten und Distanzdaten vom Host zum Gerät.
 *
 * @param graphData die Gerätedatenstruktur für den Graphen
 * @param csrFormat die CSR-Datenstruktur für den Graphen
 * @param distanceCage15 der Speicher für die Distanzwerte
 */
void copyDataHostToDevice(GraphData& graphData, const CSR_Format& csrFormat, const unique_ptr<int[]>& distanceCage15) {
    const int sizeInt = sizeof(int);
    const int distinationSize = sizeInt * (csrFormat.rSize - 1);

    cudaMemcpyAsync(graphData.dev_R, csrFormat.R, csrFormat.rSize * sizeInt, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(graphData.dev_C, csrFormat.C, csrFormat.cSize * sizeInt, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(graphData.dev_Distance, distanceCage15.get(), distinationSize, cudaMemcpyHostToDevice);
}

/**
 * Kopiert Distanzdaten vom Gerät zum Host.
 *
 * @param distanceCage15 der Speicher für die Distanzwerte
 * @param graphData die Gerätedatenstruktur für den Graphen
 * @param csrFormat die CSR-Datenstruktur für den Graphen
 */
void copyDataDeviceToHost(unique_ptr<int[]>& distanceCage15, const GraphData& graphData, const CSR_Format& csrFormat) {
    const int sizeInt = sizeof(int);
    const int distinationSize = sizeInt * (csrFormat.rSize - 1);
    cudaMemcpyAsync(distanceCage15.get(), graphData.dev_Distance, distinationSize, cudaMemcpyDeviceToHost);
}


/**
 * @brief Allokiert Geräte-Speicher für Graphdaten (CSR-Format)
 * @param graph CSR-Format des Graphen
 * @param sizeInt Größe eines int-Datentyps
 * @return GraphData-Struktur, die den allokierten Speicher enthält
 */
GraphData allocateGraphData(CSR_Format &graph, int sizeInt) {
    GraphData data;
    data.rSize = graph.rSize * sizeInt;
    data.cSize = graph.cSize * sizeInt;
    checkCudaError(cudaMalloc((void**)&data.dev_R, data.rSize), "cudaMalloc data.dev_R faild");
    checkCudaError(cudaMalloc((void**)&data.dev_C, data.cSize), "cudaMalloc data.dev_C faild");
    checkCudaError(cudaMalloc((void**)&data.dev_Distance, (graph.rSize - 1) * sizeInt), "cudaMalloc data.dev_Distance faild");    
    return data;
}

/**
 * @brief Allokiert Geräte-Speicher für die Queue-Daten der Breitensuche
 * @param queueSize Größe der Queue (in int-Einheiten)
 * @param sizeInt Größe eines int-Datentyps
 * @return QueueData-Struktur, die den allokierten Speicher enthält
 */
QueueData allocateQueueData(int queueSize, int sizeInt){
    QueueData devQueueData;
    // Gerätespeicher für die Queue-Daten allokiert
    checkCudaError(cudaMalloc(&devQueueData.inQ, queueSize * sizeInt), "cudaMalloc d_inQ failed");
    checkCudaError(cudaMalloc(&devQueueData.outQ, queueSize * sizeInt), "cudaMalloc d_outQ failed");

    // Gerätespeicher für die Counter-Daten allokiert
    checkCudaError(cudaMallocManaged(&devQueueData.counterIn, sizeInt), "cudaMalloc d_counterIn failed");
    checkCudaError(cudaMemset(devQueueData.counterIn, 0, sizeInt), "cudaMemset devQueueData.counterIn failed");
    checkCudaError(cudaMallocManaged(&devQueueData.counterOut, sizeInt), "cudaMalloc d_counterOut failed");
    checkCudaError(cudaMemset(devQueueData.counterOut, 0, sizeInt), "cudaMemset devQueueData.counterOut failed");
    return devQueueData;
}


/**
 * @brief Gibt die auf dem Device allokierten Speicher für den Graphen frei
 *
 * @param graphData GraphData Struktur, die den allokierten Speicher enthält
 */
void freeGraphData(GraphData &graphData){
    cudaFree(graphData.dev_C);
    cudaFree(graphData.dev_R);
    cudaFree(graphData.dev_Distance);
}

/**
 * @brief Gibt den auf dem Device allokierten Speicher für die Queue-Daten frei
 *
 * @param queueData QueueData Struktur, die den allokierten Speicher enthält
 */
void freeQueueData(QueueData &queueData){
    cudaFree(queueData.counterIn);
    cudaFree(queueData.counterOut);
    cudaFree(queueData.inQ);
    cudaFree(queueData.outQ);
}

/**
 * @brief Gibt die Distanzwerte eines Graphen aus, die von einer bestimmten Quelle aus berechnet wurden
 * 
 * @param distances Pointer auf ein Array von Distanzwerten
 * @param size Größe des Arrays
 */
void printDistance(const int *distances, const int size){
    for (int i=0; i< size; ++i){
        printf("distance %d = %d\n", i, distances[i]);
    }
    printf("\n");
}

/**
 * @brief Gibt die Kantenliste eines Graphen aus, die im CSR-Format vorliegt.
 * 
 * @param cage15 Kantenliste im CSR-Format
 */
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


/**
 * @brief Liest eine Zeile in einer Datei in Matrix-Markt-Format ein und speichert die Werte in source und destination
 *
 * @param line Eingabezeile in Matrix-Markt-Format (z.B. "1 2 10")
 * @param source Pointer auf den Speicherort, an dem die Quellknoten-ID gespeichert werden soll
 * @param destination Pointer auf den Speicherort, an dem die Zielknoten-ID gespeichert werden soll
 * @param deli Trennzeichen zwischen den Werten (Standardwert: " ")
 */
void readLine(string line,  int *source, int *destination ,string deli = " ")
{
    int start = 0;
    int end = line.find(deli);
    *destination = stoi(line.substr(start, end - start));
    start = end + deli.size();
    end = line.find(deli, start);
    *source = stoi(line.substr(start, end - start));
}



/**
 * @brief Liest eine Graphdatei in Matrix-Markt-Format ein und wandelt sie in den CSR-Format um.
 *        Die Zeilen der Graphdatei müssen nach Quellknoten aufsteigend sortiert sein.
 *
 * @param path Pfad zur Graphdatei in Matrix-Markt-Format
 * @return CSR_Format-Struktur, die den eingelesenen Graphen im CSR-Format enthält.
 */
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


/**
 * @brief Tauscht den Inhalt zweier Queues und aktualisiert die Anzahl der Elemente.
 * 
 * @param d_inQ Ein Zeiger auf den Zeiger des Input-Queues auf dem Device.
 * @param d_outQ Ein Zeiger auf den Zeiger des Output-Queues auf dem Device.
 * @param s_counterIn Ein Zeiger auf die Variable, die die Anzahl der Elemente im Input-Queue enthält.
 * @param s_counterOut Ein Zeiger auf die Variable, die die Anzahl der Elemente im Output-Queue enthält.
 */
void swapQueues(int **d_inQ, int **d_outQ, int *s_counterIn, int *s_counterOut) {
    
    int *temp;
    temp = *d_outQ;
    *d_outQ = *d_inQ;
    *d_inQ = temp;

    *s_counterIn = *s_counterOut;
    *s_counterOut = 0;   
}

/**
 * @brief Prüft, ob der übergebene cudaError_t-Wert erfolgreich ist. Wenn nicht, wird eine Fehlermeldung ausgegeben und das Programm wird beendet.
 * 
 * @param error Der cudaError_t-Wert, der überprüft wird.
 * @param message Eine kurze Nachricht, die beschreibt, welche Operation durchgeführt wurde, als der Fehler aufgetreten ist.
 */
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Funktion zum Überprüfen, ob der Host genügend Speicherplatz für ein Array bereitstellen konnte.
 * 
 * @param ptr Der Pointer zum Array.
 * @param ptrName Der Name des Pointers, um den Fehler auf der Konsole auszugeben.
 */
void checkHostAllocation(void *ptr, const char* ptrName){
    if (!ptr) {
        fprintf(stderr, "Error: host memory allocation failed for %s.\n", ptrName);
        exit(EXIT_FAILURE);
    }
}