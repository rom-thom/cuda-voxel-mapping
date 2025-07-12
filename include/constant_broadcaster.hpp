#ifndef CONSTANT_BROADCASTER_HPP
#define CONSTANT_BROADCASTER_HPP

#include <cstdint>

void broadcast_intrinsics(const float* intrinsics);
void broadcast_image_size(int width, int height);
void broadcast_grid_constants(uint grid_size_x, uint grid_size_y, uint grid_size_z, float resolution);
void broadcast_depth_range(float min_depth, float max_depth);
void broadcast_log_odds(float occ, float free, float min, float max, float occ_thresh, float free_thresh);

#endif // CONSTANT_BROADCASTER_HPP