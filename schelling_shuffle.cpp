#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <mpi.h>


int main()
{
    const int psize = 4;
    int total_labeled_wantmove = 0;
    int total_wantmove = 0;
    int * wantmove_ones_counts;
    int * wantmove_counts;
    const int max_tries = 5;
    int n_tries = 0;
    int border_locations;
    int * new_counts;
    int max_binsize;
    int remainder;
    int wantmove_local;
    int wantmove_local_labeled;
    int n_houses;
    
    wantmove_ones_counts = (int *)malloc(psize * sizeof(int));  
    wantmove_counts = (int *)malloc(psize * sizeof(int));  

    new_counts = (int *)malloc(psize * sizeof(int));  

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(2,100);
    
    // generate the model situation
    printf("Wantmoves: \n");
    for (int i=0; i<psize; i++)
    {   
        wantmove_local_labeled = distribution(generator);
        wantmove_local = wantmove_local_labeled + distribution(generator);
        
        wantmove_ones_counts[i] = wantmove_local_labeled;
        wantmove_counts[i] = wantmove_local;
        
        total_labeled_wantmove += wantmove_ones_counts[i];
        total_wantmove += wantmove_counts[i];
        
        printf("%d out of %d \n", wantmove_ones_counts[i], wantmove_counts[i]);
    }
    printf("\n");
    printf("Total %d out of %d \n", total_labeled_wantmove, total_wantmove);
    
    //shuffle
    while(n_tries < max_tries)
    {
        n_tries++;
        remainder = total_labeled_wantmove;
        for(int i=0; i<psize-1; i++)
        {
            std::uniform_int_distribution<int> distribution(0, std::min(remainder, wantmove_counts[i]));
            n_houses = distribution(generator);
            remainder -= n_houses;
            new_counts[i] = n_houses;
        }
        printf("%d\n", remainder);
        new_counts[psize - 1] = remainder;
        
        for(int i=0; i<psize; i++)
        {
            printf("%d ", new_counts[i]);
        }
        printf("\n");

        if (remainder < wantmove_counts[psize - 1])
        {
            printf("Success \n");
            break;
        } else {
            printf("Invalid configuration, retrying...\n");
        }
    }
    

}


