#include "main.hpp"
#include "SpikingNN.hpp"

BOOL WINAPI DllMain(HINSTANCE module_handle, DWORD reason_for_call, LPVOID reserved)
{
    return TRUE;
}

extern "C" __declspec(dllexport) void __cdecl Create(
    int groupSize,
    int groupNumber,
    int numberOfInterGroupConnections,
    float inGroupWireProbability,
    int device_id)
{
    cudaSetDevice(device_id);
    SpikingNN* handle = new SpikingNN(
        groupSize, 
        groupNumber, 
        numberOfInterGroupConnections, 
        inGroupWireProbability);
}

extern "C" __declspec(dllexport) void __cdecl Iterate(
    SpikingNN* handle,
    float* inputGroup,
    float* newNeuronState)
{
    handle->Iterate(inputGroup, newNeuronState);
}

extern "C" __declspec(dllexport) void __cdecl Destroy(
    SpikingNN* handle)
{
    delete handle;
}