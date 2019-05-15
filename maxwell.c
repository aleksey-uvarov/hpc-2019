#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


// usage:
// gcc maxwell.c -o maxwell -lm


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
  double dt = 2 * M_PI / 1000;
  double dx = dt;
  int N_x = (int) floor(L / dx);
  int N_t = (int) floor(T / dt);

  int i;
  int j;

  double F_one[N_x];
  for (i = 0; i < N_x; i++) {
    F_one[i] = 0.0;
  }
  double F_two[N_x];

  double *current = F_one;
  double *next = F_two;
  double *swap = current;

  FILE *f = fopen("out.txt", "w");

  // new time step
  for (j = 1; j < N_t; j++) {
    
    for (i = N_x - 1; i > 0; i--) {
      next[i] = current[i - 1];
    }
    next[0] = boundary(j * dt);
    

     for (i = 0; i < N_x; i++) { 
      fprintf(f, "%f ", next[i]);
     }
    fprintf(f, "\n"); 
    
    swap = current;
    current = next;
    next = swap;
  }
  fclose(f);
  
  return 0;
}
