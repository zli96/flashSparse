#!/bin/bash
# Insatll FlashSparse
rm -rf build &&
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
python setup.py install
