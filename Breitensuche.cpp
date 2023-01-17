#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <random>
#include <queue>
#include <list>

#define MAX_QUEUE = 100

using namespace std;


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
    list<int> inQ;
    list<int> outQ;
    int iteration = 0;
    #pragma omp parallel for default(none)  shared(distance,R_size)
    for(int i=0; i<R_size;i++) {
        distance[i] = INT_MAX;
    }
    distance[starting_node] = 0;
    inQ.push_back(starting_node);

    printf("Distanzen bei Initialisierung: \n");
    for (int i = 0; i< R_size; i++ ) {
        printf("%d," ,distance[i]);
    }
    printf("\n");


    //Start des Algos
    //nur Arrays mit max. Breite
    
    while (!inQ.empty()) {

        //copy elements from list to array. All elements will be processed in parallel
        int currentNodes [100] = {}; //hier sollte eigentlich inQ.size() als Größe sein
        int counter = 0;
        while(!inQ.empty()) {  //nicht mehr als threads
            currentNodes [counter] = inQ.front();
            inQ.pop_front();
            counter++;
        }
        //all nodes in Queue in parallel
        #pragma omp parallel for default(none)  shared(C,R,distance,outQ,counter,iteration,currentNodes)
        for(int j=0; j < counter; j++) {
            int current_node = currentNodes[j];
            //for all neighbours of node
            for(int i=R[current_node]; i<R[current_node+1]; i++) {
                int new_node = C[i];
                if(distance[new_node] == INT_MAX) {
                    distance[new_node] = iteration + 1;
                    outQ.push_back(new_node); //hier dann in outArray, und counter++ schützen
                }
            }  
        }
        iteration++;
        // inQ = outQ 
        // keine schöne Lösung: Vielleicht: delete InQ, dann pointer ändern inQ = outQ, dann clear outQ
        // arrays und pointer auf arrays tauschen
        while (!outQ.empty()) {
            inQ.push_back(outQ.front());
            outQ.pop_front();
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
    printf("Die Laufzeit der Funktion ist %d Mikrosekunden. \n", duration);
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
    //execution
    //zeitmesser(0,C1,R1,11,10,distance1,breitensucheMulticore);
    zeitmesser(0,C2,R2,24,16,distance2,breitensucheMulticore);
    
    //result
    printf("Ergebnis der Breitensuche: \n");
    for(int i=0; i< sizeof(distance2) / sizeof(int);i++) {
        printf("Knoten Nr. %d \t hat Distanz %d. \n",i,distance2[i]);
    }

    /*
    for(int i=0; i< sizeof(distance1) / sizeof(int);i++) {
        printf("Knoten Nr. %d \t hat Distanz %d. \n",i,distance1[i]);
    }
    **/

    return 0;
}