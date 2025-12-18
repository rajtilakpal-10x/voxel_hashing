#pragma once

#include <string>

namespace voxhash
{

    enum class MemoryType
    {
        kHost,
        kDevice,
        kUnified
    };

    inline std::string to_string(MemoryType type)
    {
        switch (type)
        {
        case MemoryType::kHost:
            return "Host";
        case MemoryType::kDevice:
            return "Device";
        case MemoryType::kUnified:
            return "Unified";
        default:
            return "Unknown Device";
        }
    }

}