#pragma once

// Windows-specific includes.
#define VC_EXTRALEAN
#define NO_MIN_MAX
#include <windows.h>

// STL.
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <memory>
#include <assert.h>
#include <string>
#include <random>

// CUDA.
#include <cuda.h>
#include <curand.h>

// Const.
const float IN_GROUP_CONNECTION_MEAN = 0.01f;
const float IN_GROUP_CONNECTION_STD = 0.1f;
const float INTER_GROUP_CONNECTION_MEAN = 0.01f;
const float INTER_GROUP_CONNECTION_STD = 1.0f;
const float THRESHOLD_MEAN = 0.5f;
const float THRESHOLD_STD = 0.1f;


