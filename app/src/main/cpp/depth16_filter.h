#ifndef DEPTH16_FILTER_H
#define DEPTH16_FILTER_H

#include <stdint.h>

float calculateZLensFromDepth(const uint16_t* depthData, int width, int height);

#endif // DEPTH16_FILTER_H
