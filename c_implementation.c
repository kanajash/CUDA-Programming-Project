#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void dumpArr(size_t row, size_t col, float arr[row][col])
{
  printf("Array output: \n");
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      printf("%.2f ", arr[i][j]);
    }
    printf("\n");
  }
}

void C_hadamard(size_t ARR_SIZE, float z[ARR_SIZE][ARR_SIZE], float x[ARR_SIZE][ARR_SIZE], float y[ARR_SIZE][ARR_SIZE])
{
  for (size_t i = 0; i < ARR_SIZE; ++i) {
    for (size_t j = 0; j < ARR_SIZE; ++j) {
      z[i][j] = x[i][j] * y[i][j];
    }
  }
}

int main()
{
  const size_t ARR_SIZE = 1024;
  size_t NUM_EXEC = 10;

  // https://stackoverflow.com/questions/3911400/how-to-pass-2d-array-matrix-in-a-function-in-c
  // int (*array)[cols] = malloc(rows * cols * sizeof(array[0][0]));
  float (*x)[ARR_SIZE] = malloc(ARR_SIZE * ARR_SIZE * sizeof(x[0][0]));
  float (*y)[ARR_SIZE] = malloc(ARR_SIZE * ARR_SIZE * sizeof(y[0][0]));
  float (*z)[ARR_SIZE] = malloc(ARR_SIZE * ARR_SIZE * sizeof(z[0][0]));

  for (size_t i = 0; i < ARR_SIZE; ++i) {
    for (size_t j = 0; j < ARR_SIZE; ++j) {
      x[i][j] = 1.0f;
      y[i][j] = 2.0f;
    }
  }

  clock_t start, end;
  double elapse, time_taken;
  elapse = 0.0f;
  // fill in cache
  C_hadamard(ARR_SIZE, z, x, y);


  for (size_t i = 0; i < NUM_EXEC; ++i) {
      start = clock();
      C_hadamard(ARR_SIZE, z, x, y);
      end = clock();
      time_taken = (end-start)*1E6/CLOCKS_PER_SEC;
      elapse +=  time_taken;
  }


  size_t err_count = 0;

  for (size_t i = 0; i < ARR_SIZE; ++i) {
    for (size_t j = 0; j < ARR_SIZE; ++j) {
      if( (x[i][j] * y[i][j]) != z[i][j] ) {
        err_count++;
      }
    }
  }

  free(x);
  free(y);
  free(z);
  // dumpArr(ARR_SIZE, ARR_SIZE, z);
  printf("Function in C took an average time of %lf in microseconds with array size: %lu\n", elapse/NUM_EXEC, ARR_SIZE * ARR_SIZE);
  printf("Total error count: %lu", err_count);

  return 0;
}
