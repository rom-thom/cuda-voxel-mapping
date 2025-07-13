#include "constant_broadcaster.hpp"

extern "C" void set_insertion_intrinsics_d(const float* intrinsics);
extern "C" void set_insertion_image_size_d(uint width, uint height);
extern "C" void set_insertion_grid_constants_d(uint grid_size_x, uint grid_size_y, uint grid_size_z, float resolution);
extern "C" void set_insertion_depth_range_d(float min_depth, float max_depth);
extern "C" void set_insertion_log_odds_properties_d(float log_odds_occupied, float log_odds_free, float log_odds_min, float log_odds_max, float occupancy_threshold, float free_threshold);

extern "C" void set_extraction_intrinsics_d(const float* intrinsics);
extern "C" void set_extraction_image_size_d(uint width, uint height);
extern "C" void set_extraction_grid_constants_d(uint grid_size_x, uint grid_size_y, uint grid_size_z, float resolution);
extern "C" void set_extraction_depth_range_d(float min_depth, float max_depth);
extern "C" void set_extraction_log_odds_properties_d(float log_odds_occupied, float log_odds_free, float log_odds_min, float log_odds_max, float occupancy_threshold, float free_threshold);


void broadcast_intrinsics(const float* intrinsics) {
    set_insertion_intrinsics_d(intrinsics);
}

void broadcast_image_size(int width, int height) {
    set_insertion_image_size_d(width, height);
}

void broadcast_grid_constants(uint grid_size_x, uint grid_size_y, uint grid_size_z, float resolution) {
    set_insertion_grid_constants_d(grid_size_x, grid_size_y, grid_size_z, resolution);
    set_extraction_grid_constants_d(grid_size_x, grid_size_y, grid_size_z, resolution);
}

void broadcast_depth_range(float min_depth, float max_depth) {
    set_insertion_depth_range_d(min_depth, max_depth);
}

void broadcast_log_odds(float occ, float free, float min, float max, float occ_thresh, float free_thresh) {
    set_insertion_log_odds_properties_d(occ, free, min, max, occ_thresh, free_thresh);
    set_extraction_log_odds_properties_d(occ, free, min, max, occ_thresh, free_thresh);
}