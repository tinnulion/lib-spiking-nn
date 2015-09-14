#pragma once

#include "main.hpp"
#include "kernels.cuh"

#define CUDA_CHECKED_CALL(cuda_call) \
{ \
    cudaError_t status = cuda_call; \
    cudaEnsureSuccess(status, __FILE__, __LINE__); \
}

void cudaEnsureSuccess(cudaError_t status, const char *const filename, int line_number)
{
    if (status != cudaSuccess)
    {
        std::cout << "Error : " << cudaGetErrorString(status) << " in "
            << filename << " : Line " << line_number << std::endl;
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

#define CURAND_CHECKED_CALL(cuda_call) \
{ \
    curandStatus_t status = cuda_call; \
    curandEnsureSuccess(status, __FILE__, __LINE__); \
}

void curandEnsureSuccess(curandStatus_t status, const char *const filename, int line_number)
{
    if (status != CURAND_STATUS_SUCCESS)
    {
        std::cout << "CURAND error: " << status << " in "
            << filename << " : Line " << line_number << std::endl;
        throw std::runtime_error(std::string("CURAND error: ") + std::to_string(status));
    }
}

class SpikingNN
{
public:

    SpikingNN(
        int groupSize, 
        int groupNumber,
        int numberOfInterGroupConnections,
        float inGroupWireProbability) :
        mGroupSize(groupSize),
        mGroupNumber(groupNumber),
        mNumberOfInterGroupConnections(numberOfInterGroupConnections)
    {
        AllocateDeviceMemory();
        InitWeightsAndThresholds(inGroupWireProbability);
    }

    virtual ~SpikingNN()
    {
        CUDA_CHECKED_CALL(cudaFree(dNeuronState));
        CUDA_CHECKED_CALL(cudaFree(dBufferState));
        CUDA_CHECKED_CALL(cudaFree(dThresholds));
        CUDA_CHECKED_CALL(cudaFree(dInGroupConnectionWeghts));
        CUDA_CHECKED_CALL(cudaFree(dInterGroupConnectionFrom));
        CUDA_CHECKED_CALL(cudaFree(dInterGroupConnectionTo));
        CUDA_CHECKED_CALL(cudaFree(dInterGroupConnectionWeights));
    }

    void Iterate(
        float* inputGroupState,
        float* newNeuronState)
    {
        // Set activations for input group.
        CUDA_CHECKED_CALL(cudaMemcpy(
            dNeuronState,
            inputGroupState,
            mGroupSize * sizeof(float),
            cudaMemcpyHostToDevice));

        // Clear buffer.
        CUDA_CHECKED_CALL(cudaMemset(
            dBufferState,
            0,
            mGroupSize * mGroupNumber * sizeof(float)));

        // Process inter-group connections.
        ProcessInterGroupConnections(
            dNeuronState,
            dBufferState,
            dInterGroupConnectionFrom,
            dInterGroupConnectionTo,
            dInterGroupConnectionWeights,
            mNumberOfInterGroupConnections);

        // Process in-group connections.
        ProcessInGroupConnections(
            dNeuronState,
            dBufferState,
            dInGroupConnectionWeghts,
            mGroupSize, 
            mGroupNumber);

        // Apply thresholds.
        ApplyThreshold(
            dBufferState, 
            dThresholds, 
            mGroupSize, 
            mGroupNumber);

        // Copy buffer to current state.
        CUDA_CHECKED_CALL(cudaMemcpy(
            dNeuronState,
            dBufferState,
            mGroupSize * mGroupNumber * sizeof(float),
            cudaMemcpyDeviceToDevice));

        // Copy data from device.
        CUDA_CHECKED_CALL(cudaMemcpy(
            newNeuronState,
            dNeuronState,
            mGroupSize * mGroupNumber * sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    SpikingNN(const SpikingNN&) = delete;
    SpikingNN& operator= (const SpikingNN&) = delete;
    SpikingNN(const SpikingNN&&) = delete;
    SpikingNN& operator= (const SpikingNN&&) = delete;

private:
    
    // Parameters
    int mGroupSize;
    int mGroupNumber;
    int mNumberOfInterGroupConnections;

    // CUDA matrices and vectors.
    void* dNeuronState;
    void* dBufferState;
    void* dThresholds;
    void* dInGroupConnectionWeghts;
    void* dInterGroupConnectionFrom;
    void* dInterGroupConnectionTo;
    void* dInterGroupConnectionWeights;

    void AllocateDeviceMemory()
    {
        CUDA_CHECKED_CALL(cudaMalloc(
            &dNeuronState,
            mGroupSize * mGroupNumber * sizeof(float)));
        CUDA_CHECKED_CALL(cudaMemset(
            dNeuronState,
            0,
            mGroupSize * mGroupNumber * sizeof(float)));
        CUDA_CHECKED_CALL(cudaMalloc(
            &dBufferState,
            mGroupSize * mGroupNumber * sizeof(float)));
        CUDA_CHECKED_CALL(cudaMalloc(
            &dThresholds,
            mGroupSize * mGroupNumber * sizeof(float)));
        CUDA_CHECKED_CALL(cudaMalloc(
            &dInGroupConnectionWeghts,
            mGroupSize * mGroupSize * mGroupNumber * sizeof(float)));
        CUDA_CHECKED_CALL(cudaMalloc(
            &dInterGroupConnectionFrom,
            mNumberOfInterGroupConnections * sizeof(int)));
        CUDA_CHECKED_CALL(cudaMalloc(
            &dInterGroupConnectionTo,
            mNumberOfInterGroupConnections * sizeof(int)));
        CUDA_CHECKED_CALL(cudaMalloc(
            &dInterGroupConnectionWeights,
            mNumberOfInterGroupConnections * sizeof(float)));
    }

    void InitInGroupConnections(float inGroupWireProbability)
    {
        // Init weights by curand.
        curandGenerator_t device_generator;
        CURAND_CHECKED_CALL(curandCreateGenerator(&device_generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECKED_CALL(curandSetPseudoRandomGeneratorSeed(device_generator, 10ULL));
        CURAND_CHECKED_CALL(curandGenerateNormal(
            device_generator,
            reinterpret_cast<float*>(dInGroupConnectionWeghts),
            mGroupSize * mGroupSize * mGroupNumber,
            IN_GROUP_CONNECTION_MEAN,
            IN_GROUP_CONNECTION_STD));
        
        void* turnedOnWires;
        CUDA_CHECKED_CALL(cudaMalloc(&turnedOnWires, sizeof(float) * mGroupSize * mGroupSize * mGroupNumber));
        CURAND_CHECKED_CALL(curandGenerateUniform(
            device_generator,
            reinterpret_cast<float*>(turnedOnWires),
            mGroupSize * mGroupSize * mGroupNumber));
        CURAND_CHECKED_CALL(curandDestroyGenerator(device_generator));

        // Eliminate self-connections.
        EliminateConnections(dInGroupConnectionWeghts, turnedOnWires, mGroupSize, mGroupNumber, inGroupWireProbability);

        CUDA_CHECKED_CALL(cudaFree(turnedOnWires));

       //Absify(dInGroupConnectionWeghts, mGroupSize * mGroupSize * mGroupNumber);
    }

    void InitInterGroupConnections()
    {
        // Initialize indices on host.
        std::mt19937 hostGenerator(0);
        std::uniform_int_distribution<> uniformDistributionFill(0, mGroupNumber - 1);
        std::uniform_int_distribution<> uniformDistributionShrinked(1, mGroupNumber - 1);
        std::uniform_int_distribution<> uniformDistributionShift(0, mGroupSize - 1);
        std::vector<int> fromIndices(mNumberOfInterGroupConnections, 0);
        std::vector<int> toIndices(mNumberOfInterGroupConnections, 0);
        for (int i = 0; i < mNumberOfInterGroupConnections; i++)
        {
            int fromGroup = uniformDistributionFill(hostGenerator);
            int toGroup = (uniformDistributionShrinked(hostGenerator) + fromGroup) % mGroupNumber;

            int fromShift = uniformDistributionShift(hostGenerator);
            int toShift = uniformDistributionShift(hostGenerator);

            fromIndices[i] = fromGroup * mGroupSize + fromShift;
            toIndices[i] = toGroup * mGroupSize + toShift;
        }
        CUDA_CHECKED_CALL(cudaMemcpy(
            dInterGroupConnectionFrom, 
            &fromIndices[0], 
            sizeof(int) * mNumberOfInterGroupConnections, 
            cudaMemcpyHostToDevice));
        CUDA_CHECKED_CALL(cudaMemcpy(
            dInterGroupConnectionTo,
            &toIndices[0],
            sizeof(int) * mNumberOfInterGroupConnections,
            cudaMemcpyHostToDevice));

        // Initialize weights by curand.
        curandGenerator_t device_generator;
        CURAND_CHECKED_CALL(curandCreateGenerator(&device_generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECKED_CALL(curandSetPseudoRandomGeneratorSeed(device_generator, 20ULL));
        CURAND_CHECKED_CALL(curandGenerateNormal(
            device_generator,
            reinterpret_cast<float*>(dInterGroupConnectionWeights),
            mNumberOfInterGroupConnections,
            INTER_GROUP_CONNECTION_MEAN,
            INTER_GROUP_CONNECTION_STD));
        CURAND_CHECKED_CALL(curandDestroyGenerator(device_generator));

        //Absify(dInterGroupConnectionWeights, mNumberOfInterGroupConnections);
    }

    void InitThresholds()
    {
        curandGenerator_t device_generator;
        CURAND_CHECKED_CALL(curandCreateGenerator(&device_generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECKED_CALL(curandSetPseudoRandomGeneratorSeed(device_generator, 30ULL));
        CURAND_CHECKED_CALL(curandGenerateNormal(
            device_generator,
            reinterpret_cast<float*>(dThresholds),
            mGroupSize * mGroupNumber,
            THRESHOLD_MEAN,
            THRESHOLD_STD));
        CURAND_CHECKED_CALL(curandDestroyGenerator(device_generator));

        Absify(dThresholds, mGroupSize * mGroupNumber);
    }

    void InitDummyInGroupConnections()
    {
        std::vector<float> weights(mGroupSize * mGroupSize * mGroupNumber, 1.0f);
        for (size_t i = 0; i < weights.size(); i++)
        {
            int j = i % (mGroupSize * mGroupSize);
            int x = j % mGroupSize;
            int y = j / mGroupSize;
            if (x == y)
            {
                weights[i] = 0.0f;
            }
        }
        CUDA_CHECKED_CALL(cudaMemcpy(
            dInGroupConnectionWeghts,
            &weights[0],
            sizeof(float) * mGroupSize * mGroupSize * mGroupNumber,
            cudaMemcpyHostToDevice));
    }

    void InitDummyInterGroupConnections()
    {
        std::vector<int> fromIndices(mNumberOfInterGroupConnections, mGroupSize * mGroupNumber - 1);
        std::vector<int> toIndices(mNumberOfInterGroupConnections, mGroupSize * mGroupNumber - 1);
        std::vector<float> weights(mNumberOfInterGroupConnections, 1.0f);
        for (int i = 0; i < mGroupNumber - 1; i++)
        {
            fromIndices[i] = mGroupSize * i;
            toIndices[i] = mGroupSize * (i + 1);
        }
        CUDA_CHECKED_CALL(cudaMemcpy(
            dInterGroupConnectionFrom,
            &fromIndices[0],
            sizeof(int) * mNumberOfInterGroupConnections,
            cudaMemcpyHostToDevice));
        CUDA_CHECKED_CALL(cudaMemcpy(
            dInterGroupConnectionTo,
            &toIndices[0],
            sizeof(int) * mNumberOfInterGroupConnections,
            cudaMemcpyHostToDevice));
        CUDA_CHECKED_CALL(cudaMemcpy(
            dInterGroupConnectionWeights,
            &weights[0],
            sizeof(float) * mNumberOfInterGroupConnections,
            cudaMemcpyHostToDevice));
    }

    void InitDummyThresholds()
    {
        std::vector<float> thresholds(mGroupSize * mGroupNumber, 0.5f);
        CUDA_CHECKED_CALL(cudaMemcpy(
            dThresholds,
            &thresholds[0],
            sizeof(float) * mGroupSize * mGroupNumber,
            cudaMemcpyHostToDevice));
    }

    void InitWeightsAndThresholds(float inGroupWireProbability)
    {
        InitInGroupConnections(inGroupWireProbability);
        InitInterGroupConnections();
        InitThresholds();

        //InitDummyInGroupConnections();
        //InitDummyInterGroupConnections();
        //InitDummyThresholds();
    }
};

