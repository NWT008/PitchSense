#ifndef KERNEL_H
#define KERNEL_H

void launchMotionKernel(
    unsigned char* d_prevFrame,
    unsigned char* d_currFrame,
    unsigned char* d_motionMask,
    int width,
    int height,
    int threshold
);

#endif
