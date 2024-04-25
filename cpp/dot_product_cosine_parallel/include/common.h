#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>

void checkCudaError(cudaError_t error);

void loadDataFromFile(const std::string& filename, std::vector<float>& array);
