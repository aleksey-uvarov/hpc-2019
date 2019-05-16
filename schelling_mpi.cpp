#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <mpi.h>

/// Simlest Schelling model cellular automaton

/// Compile with 'g++ Schelling.c -o schelling.out'
/// Run with './schelling.out'

struct House
{
    // int row;
    // int col;
    int status;
};

// basicly -- square matrix
class City
{
    // linear sizes of rectangular City
    int size_x;
    int size_y;
    
    // all houses row-wise
    House * houses;
    
    // coefficient from [0., 1.]
    double coeff;
    
    // indices of housed whose inhabitants need to move
    int * wantmove;
    
    // quantity and quality of those wanting to move
    int count_wantmove;
    int count_wantmove_label_one;
    
    // rank of process
    int my_rank;
    
    //total number of processes
    int psize;
    
    // data borders
    House * my_left_border;
    House * my_right_border;
    House * left_neighbour_border;
    House * right_neighbour_border;
    
    
    // 
    
public:
    
    City();
    
    // random initialization of houses
    City(const int newsize_x, const int newsize_y, const double newcoeff, const int newrank);
    
    ~City();
    
    // sets house[k].moveflag if inhabitants of k-th house want to relocate
    // returns number of moveflags set
    int EvaluateMove();
    //void Swap(const int first_ind, const int sec_ind);
    void Shuffle();
    void Iterate(const int iterations);
    
    void Swap(const int first_ind, const int sec_ind);
    void LocalShuffle(const int count);
    
    //asks the neigbours about their edge components
    void SendBordersL2R();
    void SendBordersR2L();
    
    void UpdateBorders();
    
    // Your I/O
    void FileDump(const int iteration);
    void Print();
};

City::City()
{
    size_x = 0;
    size_y = 0;
    houses = NULL;
    coeff = 0.;
    wantmove = NULL;
    
    return;
}

City::City(const int newsize_x, const int newsize_y, const double newcoeff, const int newrank)
{
    size_x = newsize_x;
    size_y = newsize_y;
    my_rank = newrank;
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    count_wantmove = 0;
    count_wantmove_label_one = 0;
    
    houses = (House *)malloc(size_x * size_y * sizeof(House));
    my_left_border = (House *)malloc(size_y * sizeof(House));
    my_right_border = (House *)malloc(size_y * sizeof(House));
    left_neighbour_border = (House *)malloc(size_y * sizeof(House));
    right_neighbour_border = (House *)malloc(size_y * sizeof(House));
    
    std::default_random_engine generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, 1);
    
//     generator.seed(clock());
    
    for (int i = 0; i < size_y; ++i)
    {
        for (int j = 0; j < size_x; ++j)
        {
            // houses[i * size + j].row = i;
            // houses[i * size + j].col = j;
            houses[i * size_x + j].status = distribution(generator);
        }
    }
    
    for (int i=0; i<size_y; i++)
    {
        my_left_border[i] = houses[i * size_x];
        my_right_border[i] = houses[i * size_x + size_x - 1];
    }
    
    coeff = newcoeff;
    wantmove = (int *)malloc(size_x * size_y * sizeof(int));
    
    return;
}

City::~City()
{
    if (houses)
    {
        free(houses);
    }
    
    if (wantmove)
    {
        free(wantmove);
    }
    
    return;
}

void City::UpdateBorders()
{
    for (int i=0; i<size_y; i++)
    {
        my_left_border[i] = houses[i * size_x];
        my_right_border[i] = houses[i * size_x + size_x - 1];
    }  
}


int City::EvaluateMove()
{
    int count = 0;
    int vicinity_status;
    int qty_neighbours;
    
    count_wantmove_label_one = 0;
    
    for (int i = 0; i < size_y; ++i)
    {
        for (int j = 0; j < size_x; ++j)
        {
            // check status of all neighbours
            vicinity_status = 0;
            qty_neighbours = 0;
            
            if (i > 0)
            {
                vicinity_status += houses[(i - 1) * size_x + j].status;
                qty_neighbours += 1;
                
                if (j > 0)
                {
                    vicinity_status += houses[(i - 1) * size_x + j - 1].status;
                    qty_neighbours += 1;
                }
                
                if (j < size_x - 1)
                {
                    vicinity_status += houses[(i - 1) * size_x + j + 1].status;
                    qty_neighbours += 1;
                }
            }
            
            if (i < size_y - 1)
            {
                vicinity_status += houses[(i + 1) * size_x + j].status;
                qty_neighbours += 1;
                
                if (j > 0)
                {
                    vicinity_status += houses[(i + 1) * size_x + j - 1].status;
                    qty_neighbours += 1;
                }
                
                if (j < size_y - 1)
                {
                    vicinity_status += houses[(i + 1) * size_x + j + 1].status;
                    qty_neighbours += 1;
                }
            }
            
            if (j > 0)
            {
                vicinity_status += houses[i * size_x + j - 1].status;
                qty_neighbours += 1;
            }
            
            if (j < size_y - 1)
            {
                vicinity_status += houses[i * size_x + j + 1].status;
                qty_neighbours += 1;
            }
            
            // Cross-border checks
            if (j == 0 && my_rank > 0)
            {
                vicinity_status += left_neighbour_border[i].status;
                qty_neighbours += 1;
                if (i>0)
                {
                    vicinity_status += left_neighbour_border[i - 1].status;
                    qty_neighbours += 1;
                }
                else if (i < size_y - 1)
                {
                    vicinity_status += left_neighbour_border[i + 1].status;
                    qty_neighbours += 1;
                }
            }
            
            if (j == size_x - 1 && my_rank < (psize - 1))
            {
                vicinity_status += right_neighbour_border[i].status;
                qty_neighbours += 1;
                if (i>0)
                {
                    vicinity_status += right_neighbour_border[i - 1].status;
                    qty_neighbours += 1;
                }
                else if (i < size_y - 1)
                {
                    vicinity_status += right_neighbour_border[i + 1].status;
                    qty_neighbours += 1;
                }
            }
            
            // OLD BEHAVIOR:
            // if average neighbour status is more than coeff -- need to move
            // set 1 if want to move, 0 otherwise
            
            // NEW BEHAVIOR:
            // if average neigbour status is different from mine by more than coeff
            // then I want to move
            
            // update wantmove list and total count of houses "need to move"
            if (abs(double(vicinity_status) / double(qty_neighbours)
                - double(houses[i * size_x + j].status)) >= coeff)
            {
                wantmove[count] = i * size_x + j;
                ++count;
                if (houses[i * size_x + j].status == 1)
                {
                    count_wantmove_label_one += 1;
                }
            }
        }
    }
    count_wantmove = count;
    return count;
}


// swap statuses
void City::Swap(const int first_ind, const int sec_ind)
{
    int tmp = houses[wantmove[first_ind]].status;
    houses[wantmove[first_ind]].status = houses[wantmove[sec_ind]].status;
    houses[wantmove[sec_ind]].status = tmp;

    return;
}

// Fisher-Yates shuffle
void City::LocalShuffle(const int count)
{
    std::default_random_engine generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, count - 1);

//     generator.seed(clock());

    
    for (int k = 0; k < count; ++k)
    {
        Swap(k, distribution(generator));
    }

    return;
}



// Collect data across processes and relocate households
void City::Shuffle()
{
    int * wantmove_counts;
    int * wantmove_ones_counts;
    int total_wantmove = 0;
    int total_labeled_wantmove = 0;
    double p_labeled;
    double randval;
    int * new_counts;
//     int * new_labeled_counts;
    const int max_tries = 10000;
    int n_tries = 0;
    int remainder;
    int n_houses;
    
    std::default_random_engine generator_2(std::random_device{}());
//     generator_2.seed(clock());


    
    wantmove_counts = (int *)malloc(psize * sizeof(int));
    wantmove_ones_counts = (int *)malloc(psize * sizeof(int));  
    new_counts = (int *)malloc(psize * sizeof(int));
    
    // Gathering wantmove counts and labeled counts
    
    MPI_Gather(&count_wantmove, 1, MPI_INT, wantmove_counts,
               1, MPI_INT, 0, MPI_COMM_WORLD);
    // if (my_rank == 0)
    //   {
    //     printf("Wantmove counts gathered: \n");
    //     for (int i=0; i<psize; i++)
    //       {
    //         printf("%d ", wantmove_counts[i]);
    //       }
    //     printf("\n");
    //   }
    MPI_Gather(&count_wantmove_label_one, 1, MPI_INT, 
               wantmove_ones_counts,
               1, MPI_INT, 0, MPI_COMM_WORLD);
    // if (my_rank == 0)
    //   {
    //     printf("Labeled one: \n");
    //     for (int i=0; i<psize; i++)
    //       {
    //         printf("%d ", wantmove_ones_counts[i]);
    //       }
    //     printf("\n");
    //   }
    
    // We could also do MPI_Reduce, but now that 
    // we've gatehered everyone, it's better to sum locally
    if (my_rank == 0)
    {
        for (int i=0; i<psize; i++)
        {
            total_wantmove += wantmove_counts[i];
            total_labeled_wantmove += wantmove_ones_counts[i];
        }
        
        // printf("total wantmove: %d \n", total_wantmove);
        // printf("total labeled wantmove: %d \n", total_labeled_wantmove);
        
        if (total_wantmove >0)
        {
            p_labeled = double(total_labeled_wantmove)/double(total_wantmove);
        }
        else
        {
            //printf("Everyone is good! \n");            
        }
    }
    
    // Now I want to split the tenants between cores
    // This is like putting total_labeled_wantmove balls
    // into psize bins, but each bins is limited by wantmove_counts
    // So we imagine that all balls are in line and we 
    // randomly select bin borders
    // If the selection is not valid, throw it away and try again
    
    if (my_rank == 0)
    {
//         for (int i=0; i<psize; i++)
//         {
//             printf("%d of %d \n", wantmove_ones_counts[i], wantmove_counts[i]);
//         }
        printf("\n");
        while(n_tries < max_tries)
        {
            n_tries++;
            remainder = total_labeled_wantmove;
            for(int i=0; i<psize-1; i++)
            {
                std::uniform_int_distribution<int> distribution(0, std::min(remainder, wantmove_counts[i]));
                n_houses = distribution(generator_2);
                remainder -= n_houses;
                new_counts[i] = n_houses;
            }
            new_counts[psize - 1] = remainder;
            
//             for(int i=0; i<psize; i++)
//             {
//                 printf("%d ", new_counts[i]);
//             }
//             printf("\n");
            
            if (remainder <= wantmove_counts[psize - 1])
            {
//                 printf("Success \n");
//                 printf("");
                break;
            } else {
//                 if (n_tries % 500 == 0)
//                 {
//                 printf("Invalid configuration, retrying...\n");
//                 }
            }
        }
        if (n_tries == max_tries)
        {
            printf("Number of tries exceeded \n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(new_counts, psize, MPI_INT, 0, MPI_COMM_WORLD);
        
    // repaint all who want to move
    // first new_counts[i] are labeled, the rest are unlabeled
    
//     printf("Rank %d, Wantmove %d, New labeled %d \n", my_rank, count_wantmove, new_counts[my_rank]);
    
    for (int i=0; i<count_wantmove; i++)
    {
        if (i < new_counts[my_rank])
        {
            houses[wantmove[i]].status = 1;
        } else {
            houses[wantmove[i]].status = 0;
        }
    }
    
//     printf("Rank %d, labeled wantmove %d -> %d \n",  my_rank, 
//            count_wantmove_label_one, new_counts[my_rank]);
    
    LocalShuffle(count_wantmove);
    UpdateBorders();

    
    
    
    // Old shuffle
    
/*    MPI_Bcast(&p_labeled, 1, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
    
    // Now everyone knows p_labeled
    // Unnormalized random recoloring of points
    // Should be efficient, but doesn't conserve the qty of colored points
    std::default_random_engine generator;
    std::uniform_real_distribution<double> real_distribution(0.0,1.0);
    // printf("rank: %d, p_labeled = %f \n", my_rank, p_labeled);
    for (int i=0; i<count_wantmove; i++)
    { 
        randval = real_distribution(generator);
        if (randval < p_labeled)
        {
            houses[wantmove[i]].status = 1;
        }
        else
        {
            houses[wantmove[i]].status = 0;
        }
    } */
    
    
    
    // // ineffective shuffle
    // for (int i=0; i<total_labeled_wantmove; i++)
    // {
    //   new_colors[i] = 1;
    // }
    // for (int i=total_labeled_wantmove; i<total_wantmove; i++)
    // {
    //   new_colors[i] = 0;
    // }
    
    // std::default_random_engine generator;
    // std::uniform_int_distribution<int> distribution(0, count - 1);
    
    // int tmp;
    // for (int k = 0; k < total_wantmove; ++k)
    // {
    //     j = distribution(generator);
    //     tmp = new_colors[j];
    //     new_colors[j] = new_colors[k];
    //     new_colors[k] = tmp;
    // }
    
    
    
    
}

void City::Iterate(const int iterations)
{
    // int count;
    
    // for (int k = 0; k < iterations; ++k)
    //   {
    //     count = EvaluateMove();
    //     Shuffle(count);
    
    //     FileDump(k);
    //   }
    
    // return;
}

void City::FileDump(const int iteration)
{
    char buffer[32]; // The filename buffer.
    snprintf(buffer, sizeof(char) * 32, "dump_%d.bin", iteration);
    
    FILE * file = fopen(buffer, "wb");
    int count = size_x * size_y;
    
    fwrite(&count, sizeof(int), 1, file);
    
    for (int k = 0; k < count; ++k)
    {
        fwrite(houses + k, sizeof(int), 1, file);
    }
    
    fclose(file);
    
    return;
}

void City::Print()
{
    //printf("I'm a part of the city with rank %d out of %d \n", my_rank, psize - 1);
//     printf("My map is: \n");
    for (int i = 0; i<size_y; i++)
    {
        for (int j=0; j<size_x; j++)
        {
            printf("%d ", houses[i * size_x + j].status);
        }
        printf("\n");
    }
    // printf("My left border is: \n");
    // for (int i=0; i<size_y; i++)
    //   {
    //     printf("%d ", my_left_border[i].status);
    //   }
//     printf("\nMy right border is: \n");
//     for (int i=0; i<size_y; i++)
//       {
//         printf("%d ", my_right_border[i].status);
//       }
//     printf("\n");
//     if (my_rank > 0)
//       {
//         printf("My left neighbour's border is: \n");
//         for (int i=0; i<size_y; i++)
//         {
//     	printf("%d ", left_neighbour_border[i].status);
//         }
//         printf("\n");
//       }
//     
//     if (my_rank < psize - 1)
//     {
//         printf("My right neighbour's border is: \n");
//         for (int i=0; i<size_y; i++)
//         {
//             printf("%d ", right_neighbour_border[i].status);
//         }
//         printf("\n");	     
//     }
    
    // printf("My wantmove list is: \n");
    // for (int i=0; i<count_wantmove; i++)
    //   {
    //     printf("%d ", wantmove[i]);
    //   }
    // printf(", %d in total \n", count_wantmove);
    // printf("Among them %d are labeled one\n", count_wantmove_label_one);
//     printf("=============\n");
}

void City::SendBordersL2R()
{
    MPI_Request local_request;
    //  MPI_Status status;
    
    /// left to right
    if (my_rank < psize - 1)
    {
        for (int i=0; i<size_y; i++)
        {
            MPI_Isend(&my_right_border[i].status, 1, MPI_INT,
                      my_rank + 1, my_rank, MPI_COMM_WORLD, &local_request);
            //printf("Sent %d to the right \n", my_right_border[i].status);
        }
    }
    if (my_rank > 0)
    {
        for (int i=0; i<size_y; i++)
        {
            //	  count = 1 + i + 2 * size_y * my_rank;
            MPI_Irecv(&left_neighbour_border[i].status, 1, MPI_INT,
                      my_rank - 1, my_rank - 1, MPI_COMM_WORLD, &local_request);
            //printf("Received %d from the left \n", left_neighbour_border[i].status);
            
        }
    }
    // MPI_Wait(&request, &status);
    // MPI_Barrier(MPI_COMM_WORLD);
    
}

void City::SendBordersR2L()
{
    MPI_Request request;
    /// right to left
    if (my_rank > 0)
    {
        for (int i=0; i<size_y; i++)
        {
            MPI_Isend(&my_left_border[i].status, 1, MPI_INT,
                      my_rank - 1, my_rank * 2, MPI_COMM_WORLD, &request);
            //printf("Sent %d to the left \n", my_left_border[i].status);
        }
    }
    
    if (my_rank < psize - 1)
    {
        for (int i=0; i<size_y; i++)
        {
            //	  count = 1 + i + 2 * size_y * my_rank;
            MPI_Irecv(&right_neighbour_border[i].status, 1, MPI_INT,
                      my_rank + 1, (my_rank + 1) * 2, MPI_COMM_WORLD, &request);
            //printf("Received %d from the right \n", tmp_datapoint);
            //right_neighbour_border[i].status = tmp_datapoint;
        }
    }
    //  MPI_Wait(&request, &status);
    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    
    MPI_Status status;
    MPI_Request request;
    
    int psize;
    int prank;
    MPI_Datatype ColumnType;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    
    const int size_x = 20;
    const int size_y = 20;
    const double coeff = .51;
    const int iter = 10;
    
    MPI_Type_vector(size_y, 1, size_x, MPI_INT, &ColumnType);
    MPI_Type_commit(&ColumnType);
    
    //  randgen_t generator;
    
    
    
    City city(size_x, size_y, coeff, prank);
    city.SendBordersL2R();
    MPI_Barrier(MPI_COMM_WORLD);
    city.SendBordersR2L();
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (prank == 0)
    {
        printf("Round %d, city block %d\n", 0, prank);
        city.Print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (prank == 1)
    {
        printf("Round %d, city block %d\n", 0, prank);
        city.Print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int n=1; n<iter; n++)
    {
        city.SendBordersL2R();
        MPI_Barrier(MPI_COMM_WORLD);
        city.SendBordersR2L();
        MPI_Barrier(MPI_COMM_WORLD);
        
        city.EvaluateMove();
        MPI_Barrier(MPI_COMM_WORLD);
        city.Shuffle();
        MPI_Barrier(MPI_COMM_WORLD);
        

//         for (int i=0; i<psize; i++)
//           {
//             if (prank==i)
//               {
//                 printf("Round %d, city block %d\n", n, prank);
//                 city.Print();
//               }
//             MPI_Barrier(MPI_COMM_WORLD);
//           }
        
        if (prank == 0 && n % 2 == 0)
        {
            printf("Round %d, city block %d\n", n, prank);
            city.Print();
        }
        MPI_Barrier(MPI_COMM_WORLD);
//         if (prank == 1 && n % 2 == 0)
//         {
//             printf("Round %d, city block %d\n", n, prank);
//             city.Print();
//         }
//         MPI_Barrier(MPI_COMM_WORLD);
    }
    
    
    
    MPI_Finalize();		
    
    return 0;
}
