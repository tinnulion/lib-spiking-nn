#pragma once

void Absify(
    void* arr,
    int size);

void EliminateConnections(
    void* inGroupConnectionWeghts,
    void* turnedOnWires,
    int groupSize,
    int groupNumber,
    float inGroupWireProbability);

void EmulateProcessInGroupConnections(
    void* neuronState,
    void* bufferState,
    void* inGroupConnectionWeghts,
    int groupSize,
    int groupNumber);

void ProcessInGroupConnections(
    void* neuronState,
    void* bufferState,
    void* inGroupConnectionWeghts,
    int groupSize,
    int groupNumber);

void ProcessInterGroupConnections(
    void* neuronState,
    void* bufferState,
    void* interGroupConnectionFrom,
    void* interGroupConnectionTo,
    void* interGroupConnectionWeights,
    int numberOfInterGroupConnections);

void ApplyThreshold(
    void* bufferState,
    void* thresholds,
    int groupSize,
    int groupNumber);
