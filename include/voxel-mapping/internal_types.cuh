#ifndef INTERNAL_TYPES_CUH
#define INTERNAL_TYPES_CUH

#include "voxel-mapping/types.hpp"
#include <cstdint>
#include <cuco/static_map.cuh>

using ChunkKey = uint64_t;
using ChunkPtr = VoxelType*;

using VoxelKey = uint64_t;
using VoxelFlag = uint32_t;

enum class UpdateType : uint8_t {
    Free = 1u,
    Occupied = 2u
};

struct VoxelUpdate {
    ChunkKey key;
    uint32_t intra_chunk_idx;
    UpdateType update_type;
};

using ChunkMap = cuco::static_map<
    ChunkKey,
    ChunkPtr,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<ChunkKey>,
    cuco::linear_probing<32, cuco::default_hash_function<ChunkKey>>
>;

using ChunkMapRef = decltype(std::declval<ChunkMap&>().ref(cuco::op::find, cuco::op::insert_and_find));

using ConstChunkMapRef = decltype(std::declval<ChunkMap&>().ref(cuco::op::find));

using VoxelUpdateSet = cuco::static_map<
    VoxelKey,
    VoxelFlag,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<VoxelKey>,
    cuco::linear_probing<1, cuco::default_hash_function<VoxelKey>>
>;

using UpdateSetRef = decltype(std::declval<VoxelUpdateSet&>()
                                     .ref(cuco::op::find, cuco::op::insert_and_find));

struct UpdateContext  {
    UpdateSetRef voxel_update_set_ref;
    VoxelUpdate* update_list;
    uint32_t* update_counter;
    size_t update_list_capacity;
};

#endif // INTERNAL_TYPES_CUH