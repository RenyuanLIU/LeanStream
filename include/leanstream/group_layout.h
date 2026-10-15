#pragma once
#include "types.h"
namespace leanstream::runtime {
struct GroupLayout { uint32_t logical_group, physical_group; uint64_t file_offset; uint32_t bytes; };
}
