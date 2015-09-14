#include "main.hpp"
#include "kernels.cuh"

// Device kernels.

__global__ void kernelAbs(
    float* arr,
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    arr[i] = abs(arr[i]);
}

__global__ void kernelEliminateConnections(
    float* inGroupConnectionWeghts,
    float* turnedOnWires,
    int groupSize,
    int groupNumber,
    float inGroupWireProbability)
{
    int groupIdx = blockIdx.x;
    int connectionIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if ((groupIdx >= groupNumber) || (connectionIdx >= groupSize * groupSize))
    {
        return;
    }
    int x = connectionIdx % groupSize;
    int y = connectionIdx / groupSize;
    int k = groupIdx * groupSize * groupSize + connectionIdx;
    if (x == y)
    {
        inGroupConnectionWeghts[k] = 0.0f;
        return;
    }
    if (turnedOnWires[k] >= inGroupWireProbability)
    {
        inGroupConnectionWeghts[k] = 0.0f;
    }
}

__global__ void kernelProcessInGroupConnections(
    float* neuronState,
    float* bufferState,
    float* inGroupConnectionWeghts,
    int groupSize,
    int groupNumber)
{
    __shared__ float groupNeuronState[512];

    int currentNeuron = threadIdx.x;
    int currentGroup = blockIdx.x;
    if ((currentNeuron >= groupSize) || (currentGroup >= groupNumber))
    {
        return;
    }

    groupNeuronState[currentNeuron] = neuronState[currentGroup * groupSize + currentNeuron];

    __syncthreads();

    int connectionShift = groupSize * (currentGroup * groupSize + currentNeuron);
    float ac = 0.0f;
    for (int i = 0; i < groupSize; i++)
    {
        ac += inGroupConnectionWeghts[connectionShift + i] * groupNeuronState[i];
    }
    bufferState[currentGroup * groupSize + currentNeuron] += ac;
}

__global__ void kernelProcessInterGroupConnections(
    float* neuronState,
    float* bufferState,
    int* interGroupConnectionFrom,
    int* interGroupConnectionTo,
    float* interGroupConnectionWeights,
    int numberOfInterGroupConnections)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numberOfInterGroupConnections)
    {
        return;
    }
    int fromIdx = interGroupConnectionFrom[i];
    int toIdx = interGroupConnectionTo[i];
    if (neuronState[fromIdx] > 0.5f)
    {
        float weight = interGroupConnectionWeights[i];
        atomicAdd(&bufferState[toIdx], weight);
    }
}

__global__ void kernelApplyThreshold(
    float* bufferState,
    float* thresholds,
    int groupSize,
    int groupNumber)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groupSize * groupNumber)
    {
        return;
    }
    if (bufferState[i] >= thresholds[i])
    {
        bufferState[i] = 1.0f;
    }
    else
    {
        bufferState[i] = 0.0f;
    }
}

// Helper functions.

int RoundUpDiv(int x, int y)
{
    return (x + y - 1) / y;
}

void Absify(
    void* arr,
    int size)
{
    int BLOCK = 256;
    dim3 threads(BLOCK);
    dim3 blocks(RoundUpDiv(size, BLOCK));

    kernelAbs<<<blocks, threads>>>((float*)arr, size);
}

void EliminateConnections(
    void* inGroupConnectionWeghts,
    void* turnedOnWires,
    int groupSize, 
    int groupNumber,
    float inGroupWireProbability)
{
    int BLOCK = 256;
    dim3 threads(1, BLOCK);
    dim3 blocks(groupNumber, RoundUpDiv(groupSize * groupSize, BLOCK));

    kernelEliminateConnections<<<blocks, threads>>>(
        (float*)inGroupConnectionWeghts, 
        (float*)turnedOnWires,
        groupSize, 
        groupNumber, 
        inGroupWireProbability);
}

// Just to double-check that everything is correct.
void EmulateProcessInGroupConnections(
    void* neuronState,
    void* bufferState,
    void* inGroupConnectionWeghts,
    int groupSize,
    int groupNumber)
{
    std::vector<float> neuronStateVect(groupSize * groupNumber, 0.0f);
    std::vector<float> bufferStateVect(groupSize * groupNumber, 0.0f);
    std::vector<float> weightsVect(groupSize * groupSize * groupNumber, 0.0f);
    
    cudaMemcpy(&neuronStateVect[0], neuronState, groupSize * groupNumber * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bufferStateVect[0], bufferState, groupSize * groupNumber * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&weightsVect[0], inGroupConnectionWeghts, groupSize * groupSize * groupNumber * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < groupNumber; i++)
    {
        for (int j = 0; j < groupSize; j++)
        {
            int weightShift = groupSize * (groupSize * i + j);
            float ac = 0.0f;
            for (int k = 0; k < groupSize; k++)
            {
                ac += weightsVect[weightShift + k] * neuronStateVect[i * groupSize + k];
            }
            bufferStateVect[i * groupSize + j] += ac;
        }
    }

    cudaMemcpy(bufferState, &bufferStateVect[0], groupSize * groupNumber * sizeof(float), cudaMemcpyHostToDevice);
}

void ProcessInGroupConnections(
    void* neuronState,
    void* bufferState,
    void* inGroupConnectionWeghts,
    int groupSize,
    int groupNumber)
{
    dim3 threads(groupSize);
    dim3 blocks(groupNumber);

    kernelProcessInGroupConnections<<<blocks, threads>>>(
        (float*)neuronState,
        (float*)bufferState,
        (float*)inGroupConnectionWeghts,
        groupSize,
        groupNumber);
}

void ProcessInterGroupConnections(
    void* neuronState,
    void* bufferState,
    void* interGroupConnectionFrom,
    void* interGroupConnectionTo,
    void* interGroupConnectionWeights,
    int numberOfInterGroupConnections)
{
    int BLOCK = 256;
    dim3 threads(BLOCK);
    dim3 blocks(RoundUpDiv(numberOfInterGroupConnections, BLOCK));

    kernelProcessInterGroupConnections<<<blocks, threads>>>(
        (float*)neuronState,
        (float*)bufferState,
        (int*)interGroupConnectionFrom,
        (int*)interGroupConnectionTo,
        (float*)interGroupConnectionWeights,
        numberOfInterGroupConnections);
}

void ApplyThreshold(
    void* bufferState,
    void* thresholds,
    int groupSize,
    int groupNumber)
{
    int BLOCK = 256;
    dim3 threads(BLOCK);
    dim3 blocks(RoundUpDiv(groupSize * groupNumber, BLOCK));

    kernelApplyThreshold<<<blocks, threads>>>(
        (float*)bufferState,
        (float*)thresholds,
        groupSize,
        groupNumber);
}