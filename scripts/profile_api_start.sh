#!/bin/bash

# This command does not profile until the Cuda API to Start Profiling is executed
nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi $@
