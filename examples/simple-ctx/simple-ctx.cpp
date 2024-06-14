#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;
};

#define TIME_CONVERSION_khz 2394230*1000

inline static uint64_t rdtsc(void) {
  uint64_t tsc;
  asm volatile("rdtsc;"
               "shl $32,%%rdx;"
	       "or %%rdx,%%rax"
               : "=a"(tsc)
               :
               : "%rcx", "%rdx");
  return tsc;
}

void load_model(simple_model & model, float * a, float * b, int rows_A, int cols_A, int rows_B, int cols_B);
struct ggml_cgraph * build_graph(const simple_model& model);
struct ggml_tensor * compute(const simple_model & model);

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float * a, float * b, int rows_A, int cols_A, int rows_B, int cols_B) {
    size_t ctx_size = 0;
    {
        ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a
        ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // tensor b
        ctx_size += 2 * ggml_tensor_overhead(), // tensors
        ctx_size += ggml_graph_overhead(); // compute graph
        ctx_size += 1024; // some overhead
    }

    struct ggml_init_params params {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);

    memcpy(model.a->data, a, ggml_nbytes(model.a));
    memcpy(model.b->data, b, ggml_nbytes(model.b));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model) {
  struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);
    // result = a*b^T
    struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);
    ggml_build_forward_expand(gf, result);
    return gf;
}

// compute with backend
struct ggml_tensor * compute(const simple_model & model) {
  struct ggml_cgraph * gf = build_graph(model);
    int n_threads = 16; // number of threads to perform some operations with multi-threading
    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);
    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

int main(void) {
  //ggml_time_init();
  for (int t = 0; t < 1; t++) {
    uint64_t tsc_start = rdtsc();
    
    for (int i = 0; i < 1; i++) {
      // initialize data of matrices to perform matrix multiplication
      const int rows_A = 4, cols_A = 2;
      
      float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
      };
      
      const int rows_B = 3, cols_B = 2;
      float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
      };
      
      simple_model model;
      load_model(model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);

      // perform computation in cpu
      volatile struct ggml_tensor * result = compute(model);             
      // free memory
      ggml_free(model.ctx);
    }
    uint64_t tsc_stop = rdtsc();
    uint64_t tsc_diff = tsc_stop - tsc_start;
    float tdiff = (tsc_diff/(float)TIME_CONVERSION_khz)/1000000.0;
    printf("TSC: %.3lf seconds\n", tdiff);
  }
  return 0;
}
