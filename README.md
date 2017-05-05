# GPUBasedACS

This is an implementation of the GPU-based Ant Colony System for solving the traveling salesman problem (TSP) as described in 
_Skinderowicz, Rafał. "The GPU-based parallel ant colony system." Journal of Parallel and Distributed Computing 98 (2016): 48-60._


# Preliminaries

CUDA 6.5 or newer should be installed. You also need a CUDA-enabled GPU.


# Building

Makefile should be adjusted so that the executable for the target GPU
architecture is created.

For example:

    -gencode arch=compute_50,code=sm_50


This software is intended to compile and run on Linux. It was tested with GCC v5.4 and CUDA 8.0.

To compile & build run:

    make

It could take a while. If everything goes OK, "gpuacs" executable should be
created in the current directory.


# Running

The program takes a number of command line parameters, most of them should be
visible after running:
    ./gpuacs --help


An example execution of the ACS-GPU-Alt (the faster version of the ACS-GPU) could be started with:

    ./gpuants --test tsp/rat783.tsp  --outdir results/ --alg acs_gpu_alt
    --iter 10000

where
- --alg is the name of the algorithm to run.
- --iter is the number of the ACS iterations to execute,
- --test is the path to the TSP data instance,
- --outdir is the path to a directory in which a file with results should be
  created. Results are saved in JSON format (*.js)

Valid values for the --alg argument:
- acs - a serial version of the "standard" ACS
- acs_spm  - a serial version of the ACS but with the Selective Pheromone Memory (SPM)
  instead of a full matrix
- acs_gpu - a GPU version of the ACS
- acs_gpu_alt - an optimized GPU version of the ACS
- acs_spm_gpu - a GPU version of the ACS with the Selective Pheromone Memory
  (SPM)

# License

The source code is licensed under the [MIT
License](http://opensource.org/licenses/MIT):

Copyright 2017 Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
