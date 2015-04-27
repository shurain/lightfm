#pragma once

#include <cstdint>
#include <cstdlib>

namespace lightfm {
    uint32_t uniform_hash(const void * key, int len, uint32_t seed);
}
