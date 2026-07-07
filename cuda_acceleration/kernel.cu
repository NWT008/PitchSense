#include <cuda_runtime.h>
#include <stdio.h>

__global__ void motionKernel(
    unsigned char* prevFrame,
    unsigned char* currFrame,
    unsigned char* motionMask,
    int width,
    int height,
    int threshold
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixelIndex = row * width + col;
        int colorIndex = pixelIndex * 3;

        int prevB = (int)prevFrame[colorIndex + 0];
        int prevG = (int)prevFrame[colorIndex + 1];
        int prevR = (int)prevFrame[colorIndex + 2];

        int currB = (int)currFrame[colorIndex + 0];
        int currG = (int)currFrame[colorIndex + 1];
        int currR = (int)currFrame[colorIndex + 2];

        int diffB = abs(currB - prevB);
        int diffG = abs(currG - prevG);
        int diffR = abs(currR - prevR);

        int motionScore = diffB + diffG + diffR;

        if (motionScore > threshold)
            motionMask[pixelIndex] = 255;
        else
            motionMask[pixelIndex] = 0;
    }
}

void launchMotionKernel(
    unsigned char* d_prevFrame,
    unsigned char* d_currFrame,
    unsigned char* d_motionMask,
    int width,
    int height,
    int threshold
) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    motionKernel<<<gridSize, blockSize>>>(
        d_prevFrame,
        d_currFrame,
        d_motionMask,
        width,
        height,
        threshold
    );

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
