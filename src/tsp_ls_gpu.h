#ifndef TSP_LS_GPU_H
#define TSP_LS_GPU_H

#include <cstdint>

#define NN_SIZE  32

/**
 * GPU version of various local search algorithms for the TSP problem
 */

/**
  GPU version of the 2-opt heuristic with additional improvements in the form of
  dont_look_ bits and changes restricted to nearest neighbours of each node.
  Based on the ACOTSP source code by T. Stuzle
**/
__global__ 
void opt2(const float * __restrict__ dist_matrix,
                     uint32_t dimension,
                     uint32_t *routes,
                     float *routes_len,
                     uint32_t route_size,
                     const uint32_t * __restrict__ nn_lists,
                     int *cust_pos_);


__global__ 
void opt3(const float * __restrict__ dist_matrix,
          uint32_t dimension,
          uint32_t *routes,
          float *routes_len,
          uint32_t route_size,
          const uint32_t * __restrict__ nn_lists,
          int *route_node_indices);

#endif
