#include <stdio.h>
#include <cstdio>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <cuda.h>
#include <texture_types.h>
#include <iostream>

#define M_PI 3.14159265358979323846
#define COEF 50
#define VERTCOUNT COEF * COEF * 2
#define RADIUS 10.0f
#define FGSIZE 20
#define FGSHIFT FGSIZE / 2
#define THREADSPERBLOCK 1024


struct Vertex {
    float x, y, z;
};

__constant__ Vertex vert[VERTCOUNT];

texture<float, 3, cudaReadModeElementType> df_tex;

cudaArray* df_Array = 0;

float func(float x, float y, float z) {
    return (0.5 * sqrtf(15.0 / M_PI)) * (0.5 * sqrtf(15.0 / M_PI)) *
        z * z * y * y * sqrtf(1.0f - z * z / RADIUS / RADIUS) / RADIUS /
        RADIUS / RADIUS / RADIUS;
}

float check(Vertex* v) {

    float sum = 0.0f;

    for (int i = 0; i < VERTCOUNT; ++i)
        sum += func(v[i].x, v[i].y, v[i].z);
    return sum;
}

void FindValsInGridNodes(float* arr_f, int x_size, int y_size, int z_size) {

    for (int z = 0; z < z_size; z++) {
        for (int y = 0; y < y_size; y++) {
            for (int x = 0; x < x_size; x++) {
                arr_f[z_size * (z * y_size + y) + x] = func(x - FGSHIFT, y - FGSHIFT, z - FGSHIFT);
            }
        }
    }
}

void GetCoordOnSphere() {

    Vertex* temp_vert = (Vertex*)malloc(sizeof(Vertex) * VERTCOUNT);
    int i = 0;

    for (int iphi = 0; iphi < 2 * COEF; ++iphi) {
        for (int ipsi = 0; ipsi < COEF; ++ipsi, ++i) {
            float phi = iphi * M_PI / COEF;
            float psi = ipsi * M_PI / COEF;
            temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
            temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
            temp_vert[i].z = RADIUS * cosf(psi);
        }
    }
    std::cout << "Check: " << check(temp_vert) * M_PI * M_PI / COEF / COEF << "\n";
    cudaMemcpyToSymbol(vert, temp_vert, sizeof(Vertex) * VERTCOUNT, 0,
        cudaMemcpyHostToDevice);
    free(temp_vert);
}

void TuneTexture(cudaChannelFormatDesc channelDesc) {
    df_tex.normalized = false;
    df_tex.filterMode = cudaFilterModeLinear;
    df_tex.addressMode[0] = cudaAddressModeClamp;
    df_tex.addressMode[1] = cudaAddressModeClamp;
    df_tex.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(df_tex, df_Array, channelDesc);
}

void FillTextureMemory(float* df_h) {

    const cudaExtent volumeSize = make_cudaExtent(FGSIZE, FGSIZE,
        FGSIZE);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&df_Array, &channelDesc, volumeSize);
    cudaMemcpy3DParms cpyParams = { 0 };
    cpyParams.srcPtr = make_cudaPitchedPtr((void*)df_h, volumeSize.width
        * sizeof(float), volumeSize.width, volumeSize.height);
    cpyParams.dstArray = df_Array;
    cpyParams.extent = volumeSize;
    cpyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyParams);

    TuneTexture(channelDesc);
}

void FreeTexture() {
    cudaUnbindTexture(df_tex);
    cudaFreeArray(df_Array);
}

__device__ float interpol(float x, float y, float z, float* arr) {

    float denominator, res = 0.0f;
    float z_r[2] = { z - 1, z + 1 };
    float y_r[2] = { y - 1, y + 1 };
    float x_r[2] = { x - 1, x + 1 };

    denominator = (z_r[1] - z_r[0]) * (y_r[1] - y_r[0]) * (x_r[1] -
        x_r[0]);
    res += (arr[FGSIZE * ((int)z_r[0] * FGSIZE + (int)y_r[0]) +
        (int)x_r[0]] * (z_r[1] - z) * (y_r[1] - y) * (x_r[1] - z)) /
        denominator;
    res += (arr[FGSIZE * ((int)z_r[0] * FGSIZE + (int)y_r[0]) +
        (int)x_r[1]] * (z_r[1] - z) * (y_r[1] - y) * (x - x_r[0])) /
        denominator;
    res += (arr[FGSIZE * ((int)z_r[0] * FGSIZE + (int)y_r[1]) +
        (int)x_r[0]] * (z_r[1] - z) * (y - y_r[0]) * (x_r[1] - x)) /
        denominator;
    res += (arr[FGSIZE * ((int)z_r[0] * FGSIZE + (int)y_r[1]) +
        (int)x_r[1]] * (z_r[1] - z) * (y - y_r[0]) * (x - x_r[0])) /
        denominator;
    res += (arr[FGSIZE * ((int)z_r[1] * FGSIZE + (int)y_r[0]) +
        (int)x_r[0]] * (z - z_r[0]) * (y_r[1] - y) * (x_r[1] - x)) /
        denominator;
    res += (arr[FGSIZE * ((int)z_r[1] * FGSIZE + (int)y_r[0]) +
        (int)x_r[1]] * (z - z_r[0]) * (y_r[1] - y) * (x - x_r[0])) /
        denominator;
    res += (arr[FGSIZE * ((int)z_r[1] * FGSIZE + (int)y_r[1]) +
        (int)x_r[0]] * (z - z_r[0]) * (y - y_r[0]) * (x_r[1] - x)) /
        denominator;
    res += (arr[FGSIZE * ((int)z_r[1] * FGSIZE + (int)y_r[1]) +
        (int)x_r[1]] * (z - z_r[0]) * (y - y_r[0]) * (x - x_r[0])) /
        denominator;
    return res;
}

__global__ void FindIntegralWithTextureMem(float* a) {

    __shared__ float cache[THREADSPERBLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float x = vert[tid].x + FGSHIFT ;
    float y = vert[tid].y + FGSHIFT ;
    float z = vert[tid].z + FGSHIFT ;

    cache[cacheIndex] = tex3D(df_tex, x, y, z);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheIndex < s)
            cache[cacheIndex] += cache[cacheIndex + s];
        __syncthreads();
    }
    if (cacheIndex == 0)
        a[blockIdx.x] = cache[0];
}

__global__ void FindIntegral(float* a, float* val) {

    __shared__ float cache[THREADSPERBLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float x = vert[tid].x + FGSHIFT;
    float y = vert[tid].y + FGSHIFT ;
    float z = vert[tid].z + FGSHIFT ;

    cache[cacheIndex] = interpol(x, y, z, val);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheIndex < s)
            cache[cacheIndex] += cache[cacheIndex + s];
        __syncthreads();
    }
    if (cacheIndex == 0)
        a[blockIdx.x] = cache[0];
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float tmr;
    int blocksCount = std::min(32, (VERTCOUNT + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
    float* arr = (float*)malloc(sizeof(float) * FGSIZE * FGSIZE *
        FGSIZE);
    float* values, * sum_dev, * sum = (float*)malloc(sizeof(float) * blocksCount), answ;
    cudaMalloc((void**)&sum_dev, sizeof(float) * blocksCount);
    GetCoordOnSphere();

    FindValsInGridNodes(arr, FGSIZE, FGSIZE, FGSIZE);
    cudaMalloc((void**)&values, sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
    cudaMemcpy(values, arr, sizeof(float) * FGSIZE * FGSIZE * FGSIZE, cudaMemcpyHostToDevice);
    FillTextureMemory(arr);

    cudaEventRecord(start, 0);

    FindIntegralWithTextureMem << <blocksCount, THREADSPERBLOCK >> > (sum_dev);
    cudaDeviceSynchronize();
    cudaMemcpy(sum, sum_dev, sizeof(float) * blocksCount, cudaMemcpyDeviceToHost);
    answ = 0.0f;
    for (int i = 0; i < blocksCount; i++)
        answ += sum[i];
    answ *= M_PI * M_PI / COEF / COEF;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmr, start, stop);
    std::cout << "Sum with Texture memory:  " << answ << "\tTime: " << tmr << "\n";

    cudaEventRecord(start, 0);

    FindIntegral << <blocksCount, THREADSPERBLOCK >> > (sum_dev, values);
    cudaDeviceSynchronize();    
    cudaMemcpy(sum, sum_dev, sizeof(float) * blocksCount, cudaMemcpyDeviceToHost);
    answ = 0.0f;
    for (int i = 0; i < blocksCount; i++)
        answ += sum[i];
    answ *= M_PI * M_PI / COEF / COEF;

    cudaEventRecord(stop, 0);   
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmr, start, stop);
    std::cout << "Sum with Linear interpol: " << answ << "\tTime: " << tmr << "\n";

    cudaFree(sum_dev);
    free(sum);
    FreeTexture();
    free(arr);
    return 0;
}

