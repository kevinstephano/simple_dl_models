#!/bin/bash

nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu $@
