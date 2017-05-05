#ifndef GPU_ACS_H
#define GPU_ACS_H

#include "cuda_utils.h"

struct GPU_ACSParams : public Managed {
    uint32_t ants_count_ = 10;
    float beta_ = 3.0;
    float q0_ = 0.9;
    float rho_ = 0.2; 
    float phi_ = 0.01; // local pheromone update
    float initial_pheromone_ = 0.0;

    GPU_ACSParams &operator=(const ACSParams &other) {
        ants_count_ = other.ants_count_;
        beta_ = (float)other.beta_;
        q0_ = (float)other.q0_;
        rho_ = (float)other.rho_;
        phi_ = (float)other.phi_;
        initial_pheromone_ = (float)other.initial_pheromone_;
        return *this;
    }
};


class ACSGPU {
};

#endif
