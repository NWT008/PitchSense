#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)               \
                      << " at line " << __LINE__ << std::endl;                  \
            return 1;                                                            \
        }                                                                        \
    } while (0)

// Function implemented in kernel.cu
void launchMotionKernel(
    unsigned char* d_prevFrame,
    unsigned char* d_currFrame,
    unsigned char* d_motionMask,
    int width,
    int height,
    int threshold
);

void cpuMotionMask(
    const unsigned char* prevFrame,
    const unsigned char* currFrame,
    unsigned char* motionMask,
    int width,
    int height,
    int threshold
) {
    int totalPixels = width * height;

    for (int i = 0; i < totalPixels; i++) {
        int colorIndex = i * 3;

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
            motionMask[i] = 255;
        else
            motionMask[i] = 0;
    }
}

void generateFrame(
    std::vector<unsigned char>& frame,
    int width,
    int height,
    int frameNumber
) {
    int totalPixels = width * height;

    // Simple synthetic background plus moving bright "ball" region.
    for (int i = 0; i < totalPixels; i++) {
        int colorIndex = i * 3;

        int x = i % width;
        int y = i / width;

        frame[colorIndex + 0] = (unsigned char)((x + y) % 180);       // B
        frame[colorIndex + 1] = (unsigned char)((2 * x + y) % 180);   // G
        frame[colorIndex + 2] = (unsigned char)((x + 2 * y) % 180);   // R
    }

    // Moving ball-like object.
    int ballX = 50 + frameNumber * 5;
    int ballY = height / 2 + (int)(40.0 * sin(frameNumber * 0.08));
    int radius = 8;

    if (ballX >= width) {
        ballX = width - 1;
    }

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int x = ballX + dx;
            int y = ballY + dy;

            if (x >= 0 && x < width && y >= 0 && y < height) {
                if (dx * dx + dy * dy <= radius * radius) {
                    int pixelIndex = y * width + x;
                    int colorIndex = pixelIndex * 3;

                    frame[colorIndex + 0] = 240;
                    frame[colorIndex + 1] = 240;
                    frame[colorIndex + 2] = 240;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int width = 1920;
    int height = 1080;
    int numFrames = 300;
    int threshold = 45;

    if (argc >= 5) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        numFrames = atoi(argv[3]);
        threshold = atoi(argv[4]);
    }

    std::cout << "GPITCHU CUDA Motion-Mask Benchmark" << std::endl;
    std::cout << "Frames:     " << numFrames << std::endl;

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA device found." << std::endl;
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device: " << prop.name << std::endl;

    int totalPixels = width * height;
    size_t frameBytes = (size_t)totalPixels * 3 * sizeof(unsigned char);
    size_t maskBytes = (size_t)totalPixels * sizeof(unsigned char);

    std::vector<unsigned char> prevFrame(frameBytes);
    std::vector<unsigned char> currFrame(frameBytes);
    std::vector<unsigned char> cpuMask(maskBytes);
    std::vector<unsigned char> gpuMask(maskBytes);

    generateFrame(prevFrame, width, height, 0);

    unsigned char* d_prevFrame = nullptr;
    unsigned char* d_currFrame = nullptr;
    unsigned char* d_motionMask = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_prevFrame, frameBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_currFrame, frameBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_motionMask, maskBytes));

    CUDA_CHECK(cudaMemcpy(d_prevFrame, prevFrame.data(), frameBytes, cudaMemcpyHostToDevice));

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    double totalCpuMs = 0.0;
    double totalGpuPipelineMs = 0.0;
    float totalGpuKernelMs = 0.0f;

    int mismatches = 0;

    for (int frame = 1; frame < numFrames; frame++) {
        generateFrame(currFrame, width, height, frame);

        // CPU benchmark
        auto cpuStart = std::chrono::high_resolution_clock::now();

        cpuMotionMask(
            prevFrame.data(),
            currFrame.data(),
            cpuMask.data(),
            width,
            height,
            threshold
        );

        auto cpuEnd = std::chrono::high_resolution_clock::now();

        totalCpuMs += std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        // GPU benchmark including memory copies
        auto gpuStart = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(d_currFrame, currFrame.data(), frameBytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(startEvent));

        launchMotionKernel(
            d_prevFrame,
            d_currFrame,
            d_motionMask,
            width,
            height,
            threshold
        );

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float kernelMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernelMs, startEvent, stopEvent));
        totalGpuKernelMs += kernelMs;

        CUDA_CHECK(cudaMemcpy(gpuMask.data(), d_motionMask, maskBytes, cudaMemcpyDeviceToHost));

        auto gpuEnd = std::chrono::high_resolution_clock::now();

        totalGpuPipelineMs += std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();

        // Correctness check
        for (int i = 0; i < totalPixels; i++) {
            if (cpuMask[i] != gpuMask[i]) {
                mismatches++;
                if (mismatches > 10)
                    break;
            }
        }

        // Current becomes previous
        prevFrame.swap(currFrame);
        std::swap(d_prevFrame, d_currFrame);

        if (frame % 50 == 0) {
            std::cout << "Processed frame pair " << frame << std::endl;
        }
    }

    int framePairs = numFrames - 1;

    double avgCpuMs = totalCpuMs / framePairs;
    double avgGpuKernelMs = totalGpuKernelMs / framePairs;
    double avgGpuPipelineMs = totalGpuPipelineMs / framePairs;

    std::cout << "\nRESULTS" << std::endl;
    std::cout << "Frame pairs processed:      " << framePairs << std::endl;
    std::cout << "Average CPU time:           " << avgCpuMs << " ms/frame-pair" << std::endl;
    std::cout << "Average GPU kernel time:    " << avgGpuKernelMs << " ms/frame-pair" << std::endl;
    std::cout << "Average GPU pipeline time:  " << avgGpuPipelineMs << " ms/frame-pair" << std::endl;
    std::cout << "CPU/GPU kernel speedup:     " << avgCpuMs / avgGpuKernelMs << "x" << std::endl;
    std::cout << "CPU/GPU pipeline speedup:   " << avgCpuMs / avgGpuPipelineMs << "x" << std::endl;

    if (mismatches == 0) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED, mismatches found" << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    CUDA_CHECK(cudaFree(d_prevFrame));
    CUDA_CHECK(cudaFree(d_currFrame));
    CUDA_CHECK(cudaFree(d_motionMask));

    return 0;
}
