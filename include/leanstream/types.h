#pragma once
#include <cstddef>
#include <cstdint>
namespace leanstream::runtime {
using GroupId = int32_t;
using SlotId = int32_t;
template<class T> struct Span {
    const T* data=nullptr; size_t size=0;
    const T& operator[](size_t i) const { return data[i]; }
};
enum class IOState : uint8_t { Empty, Inflight, Ready };
enum class SourceKind : uint32_t { Cache, Staging };
enum class ConsumerMode : uint8_t { BatchCompat, Progressive, Auto };
struct GroupState { SlotId cache_slot=-1, staging_slot=-1; IOState io_state=IOState::Empty; };
// Binary-compatible with the production five-int descriptor.
struct ExecDesc {
    uint32_t logical_group, physical_group, source_slot, dest_slot;
    SourceKind source_kind;
};
static_assert(sizeof(ExecDesc)==20);
}
