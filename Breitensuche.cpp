#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <random>
#include <queue>
#include <fstream>
#include <algorithm>



#define MAX_QUEUE_SIZE  1000000
#define INT_MAX 2147483647

using namespace std;

class CSR_Format {
    public:
    int *C;
    int cSize;
    int *R;
    int rSize;
};

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


//Bedingung für die Nodes...

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

//--------------------------------------------------------------------------------------------
//input is a graph in csr format
void breitensuche(int starting_node,int C[], int R[], int C_size, int R_size, int distance[]) {
    printf("Breitensuche wird gestartet... \n");

    //Initialisierung
    queue<int> Q;
    for(int i=0; i<R_size;i++) {
        distance[i] = INT_MAX;
    }
    distance[starting_node] = 0;
    Q.push(starting_node);

    printf("Distanzen bei Initialisierung: \n");
    for (int i = 0; i< R_size; i++ ) {
        printf("%d," ,distance[i]);
    }
    printf("\n");


    //Start des Algos

    while (!Q.empty()) {
        int current_node = Q.front();
        Q.pop();
        //for all neighbours of current_node
        for(int i=R[current_node]; i<R[current_node+1]; i++) {
            int new_node = C[i];
            if(distance[new_node] == INT_MAX) {
                distance[new_node] = distance[current_node] + 1;
                Q.push(new_node);
            }
        }
    }

}
//--------------------------------------------------------------------------------------------
void breitensucheMulticore(int starting_node,int C[], int R[], int C_size, int R_size, int distance[]) {
    printf("Breitensuche wird gestartet... \n");

    //Initialisierung
    int *inQ = (int*)malloc(sizeof(int)*MAX_QUEUE_SIZE);
    int *outQ = (int*)malloc(sizeof(int)*MAX_QUEUE_SIZE);
    int *help;
    int counterIn = 0; //counters always point to the next free space in array
    int counterOut = 0;
    int privateCounter =0; //private counter for each thread
    int doublesCounter = 0; //to count nodes that are in the Q more than once

    int iteration = 0;
    #pragma omp parallel for default(none)  shared(distance,R_size)
    for(int i=0; i<R_size-1;i++) {
        distance[i] = INT_MAX;
    }
    distance[starting_node] = 0;
    inQ[counterIn] = starting_node;
    counterIn++;

    //Start des Algos

    #pragma omp parallel default(none)  shared(C,R,distance,inQ,outQ,counterIn,counterOut,iteration,help, doublesCounter) private(privateCounter)
    {
        while (counterIn != 0) {
            //----------------------------------------------------------------------------------------hier war ich 

            //chunk size abhängig von counterIn
            //schedule abhängig von counterIn: klein/mittel -> dynamic, groß -> guided
            //all nodes in Queue in parallel
            //counterIn / omp_get_num_threads()
            #pragma omp for private(privateCounter) schedule(dynamic)
            for(int j=0; j < counterIn; j++) {
                int current_node = inQ[j];
                //for all neighbours of node
                //pragma omp for ab hier, mal probieren, wahrscheinlich weniger Parallelität
                for(int i=R[current_node]; i<R[current_node+1]; i++) {
                    int new_node = C[i];
                    if(distance[new_node] == INT_MAX) {
                        //omp_lock_t writelock [16] ....
                        //mit array von locks (nur nötig, wenn doppelte Einträge in Queue)
                        //new node % 32 ergibt locknummer (hashing)
                        //lock anfragen und nochmal distance[new_node] == INT_MAX prüfen
                        distance[new_node] = iteration + 1;
                        #pragma omp atomic capture
                        {
                            privateCounter = counterOut;
                            counterOut++;
                        }
                        outQ[privateCounter] = new_node;
                        //unlock
                    }
                }  
            }
            #pragma omp single
            {
            iteration++;
            // inQ = outQ 
            help = inQ;
            inQ = outQ;
            outQ = help;
            counterIn = counterOut;
            counterOut = 0;
            //---------------------------------------------------------
            if(counterIn>MAX_QUEUE_SIZE){
                printf("Error MAX_QUEUE_SIZE überschritten!");
            }
            //doppelte Knoten in inQ?
            sort(inQ,inQ+counterIn);
            for(int i=0;i<counterIn-1;i++){
                if(inQ[i]==inQ[i+1]){
                    doublesCounter++;
                }
            }

        
            printf("Ende Iteration %d: \n",iteration-1);
            printf("InQ size: %d \n",counterIn);
            printf("Anzahl doppelt eingefügter Knoten %d. \n",doublesCounter);

            
            /*
            printf("Inhalt der InQ: \t");
            for (int i = 0; i< counterIn; i++ ) {
                printf("%d," ,inQ[i]);
            }
            printf("\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
            **/
            }
        }
    }
    
    
}




//--------------------------------------------------------------------------------------------
// Input wie bei Breitensuche + Funktionspointer
void zeitmesser(int starting_node,int C[], int R[], int C_size, int R_size, int distance[], void ( *function )( int, int[], int[], int, int, int[])) {
    auto start = chrono::high_resolution_clock::now(); //save start time
    function(starting_node,C,R,C_size,R_size,distance);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    double duration_mili = duration.count()/60000.0;
    cout <<  "Die Laufzeit der Funktion ist " << duration_mili/1000.0 << " Mikrosekunden.\n";
    // printf("Die Laufzeit der Funktion ist %ld Mikrosekunden. \n", duration.count());
}




//--------------------------------------------------------------------------------------------
int main() {
    //first graph
    int C1[11] = {1,3,0,2,4,4,5,7,8,6,8};
    int R1[10] = {0,2,5,5,6,8,9,9,11,11};
    int distance1[9] = {};

    //second graph
    int C2[24] = {1,2,2,9,4,1,4,5,11,3,7,10,8,8,6,7,1,10,12,9,13,13,14,13};
    int R2[16] = {0,2,4,5,9,10,12,13,14,16,19,21,21,22,23,24};
    int distance2[15] = {};
    //CSR_Format  graph2 = readGraph("Beispielgraph.mtx");
    //execution
    //zeitmesser(0,C1,R1,11,10,distance1,breitensucheMulticore);
    //zeitmesser(0,C2,R2,24,16,distance2,breitensucheMulticore);
    //zeitmesser(0,graph2.C,graph2.R,graph2.cSize,graph2.rSize,distance2 ,breitensucheMulticore);

    CSR_Format  cage15 = readGraph("Beispielgraph.mtx");
    //CSR_Format  cage15 = readGraph("cage15/cage15.mtx");
    cout<< "The graph has been read successfully\n";
    int *distanceCage15 = (int*) malloc(sizeof(int)*cage15.rSize-1);
    zeitmesser(0,cage15.C,cage15.R,cage15.cSize,cage15.rSize,distanceCage15 ,breitensucheMulticore);
    //zeitmesser(0,cage15.C,cage15.R,cage15.cSize,cage15.rSize,distance2 ,breitensucheMulticore);
    
    //result
    printf("Ergebnis der Breitensuche: \n");
    for(int i=0; i< cage15.rSize-1;i++) {
        printf("Knoten Nr. %d \t hat Distanz %d. \n",i,distanceCage15[i]);
    }

    /*
    for(int i=0; i< sizeof(distance1) / sizeof(int);i++) { 
        printf("Knoten Nr. %d \t hat Distanz %d. \n",i,distance1[i]);
    }
    **/
    return 0;
}