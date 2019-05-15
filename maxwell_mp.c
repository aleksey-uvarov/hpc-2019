#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double boundary(double t) {
  double fun = 0;
  if (t < M_PI * 20)
    {
      fun = sin(t);
    }
  return fun;
}

void swapArrays(double *sourceArr, double *destArr, int size) {
  
}

int main() {
  double L = 2 * M_PI * 40;
  double T = 2 * M_PI;
  double dt = 2 * M_PI / 100;
  double dx = dt;
  int N_x = (int) floor(L / dx);
  int N_t = (int) floor(T / dt);
  //int N_x = 10;
  //int N_t = 10;

  int i = 0;
  int j;

  double F_one[N_x];
  for (i = 0; i < N_x; i++) {
    F_one[i] = 0.0;
  }
  i = 0;
  double F_two[N_x];

  double *current = F_one;
  double *next = F_two;
  double *swap = current;

  // new time step
  FILE *f = fopen("out.txt", "w");

#pragma omp parallel
  {
    while (i < N_t)
      {
#pragma omp for
	for (j = 1; j < N_x; ++j)
	  {
	    next[j] = current[j - 1];
	  }
#pragma omp single
        {
        next[0] = boundary(i * dt);

        for (j = 0; j < N_x; j++) 
            { 
            fprintf(f, "%f ", next[j]); 
            } 
        fprintf(f, "\n"); 

        swap = current;
        current = next;
        next = swap;	  
        i++;
        }
      }    
  }

  fclose(f);
  return 0;
}
