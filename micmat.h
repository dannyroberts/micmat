#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>
#include <omp.h>
#include <mkl.h>
#include <math.h>
#include <offload.h>

void tester(void);

float *allocate_host(int N);

void fill_randn(int skip_num, int N, float *A, float mu, float sigma);

void fill_uniform(int skip_num, int N, float *A);

void fill_bernoulli(int skip_num, int N, float *A, float p);

void fill_zeros(int N, float *A, int offloaded);

void fill_ones(int N, float *A, int offloaded);

float *ones_mic(int N);

void expo(int N, float *A);

void clip(int N, float *A, float LOWER, float UPPER);

void clip_low(int N, float *A, float LOWER);

void flooro(int N, float *A);

void sign(int N, float *A);

float *equal(int N, float *A, float *B);

float *elementwise_or(int N, float *A, float *B);

float *leq(int N, float *A, float B);

float *geq(int N, float *A, float B);

float *greater(int N, float *A, float B);

float *labels_to_vectors(int N, int K, float *A);

void lg(int N, float *A);

void abso(int N, float *A);

void sqrto(int N, float *A);

float normo(int N, float *A);

void powo(int N, float *A, float b);

void deepcopy(int N, float *A, float *B);

float *unalign_host(int N, float *A);

float output_float(float *A);

void copy(int N, float *A, float *B, int offloaded);

void replace_host(int N, float *A, float *B);

void replace_mic(int N, float *A, float *B);

void replace_partial_host(int N_A, int SHIFT_A, float *A, int N_B, int SHIFT_B, float *B);

void replace_partial_mic(int N_A, int SHIFT_A, float *A, int N_B, int SHIFT_B, float *B);

float *get_partial(int N, int SHIFT, float *A);

void T(int ROWS, int COLS, float *A);

void scale(int N, float *A, float c);

void mult(int ROWS_A, int COLS_A, float *A, int ROWS_X, int COLS_X, float *X);

void invert(int N, float *A);

void divide(int ROWS_A, int COLS_A, float *A, int ROWS_X, int COLS_X, float *X);

void print_slice(int ROWS, int COLS, float *A, int offloaded);

void print_slice_mic(int ROWS, int COLS, float *A);

float *slice_inds(int N, int *indices, float *A, int indices_offloaded, int offloaded);

float *slice_cols(int N, int *indices, int ROWS, int COLS, float *A, int indices_offloaded, int offloaded);

float *slice_rows(int N, int *indices, int ROWS, int COLS, float *A, int indices_offloaded, int offloaded);

void offload_mic(int N, float *A);

void pull_mic(int N, float *A);

void push_mic(int N, float *A);

float *cast_float(int N, int *A, int offloaded);

void free_host(int N, float *A);

void free_mic(int N, float *A);

void free_mic_int(int N, int *A);

void update(int ROWS_A, int COLS_A, float *A, int ROWS_X, int COLS_X, float *X, float ALPHA);

void update_const(int N, float *A, float c);

float *dot(int ROWS_A, int COLS_A, int T_A, float *A, int COLS_B, int T_B, float *B);

void dot_replace(int ROWS_A, int COLS_A, int T_A, float *A, int ROWS_B, int COLS_B, int T_B, float *B, float BETA, float *C);

float *dot_vec(int N, float *A, float *B);

float *sum_axis(int ROWS_A, int COLS_A, float *A, int AXIS);

int *max_axis(int ROWS_A, int COLS_A, float *A, int AXIS);

void index_global_to_local(int ROWS_A, int COLS_A, int *A, int AXIS);

float sumo(int N, float *A);

void convolve(int N, int C, int H, int W, float *INPUTS, int K, int Y, int X, float *FILTERS, float *OUTPUTS, int tight);

void check_mic_status(void);

void ping_each_core(void);

int main(void);