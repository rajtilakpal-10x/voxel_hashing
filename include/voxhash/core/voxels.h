#pragma once
#include <memory>

namespace voxhash
{

    struct TsdfVoxel
    {
        float tsdf{0.0f};
        float weight{0.0f};
    };

    struct SemanticVoxel
    {
        uint16_t label{0};
        double weight{0.0};
    };

} // namespace voxhash