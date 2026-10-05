#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// The destination must be contiguous; source gaps are skipped exactly as in
// ggml_backend_tensor_set_2d's row loop. No tensor value is decoded or changed.
constexpr size_t RPC_WEIGHT_PACK_LIMIT = 8 * 1024 * 1024;
template<class Sink>
bool rpc_pack_weight_rows(bool enabled, bool weights, const void * data,
                         size_t offset, size_t width, size_t rows,
                         size_t dst_stride, size_t src_stride,
                         Sink sink, size_t & calls) {
    calls = 0;
    if (!enabled || !weights || rows <= 1 || width == 0 ||
        width > RPC_WEIGHT_PACK_LIMIT || dst_stride != width) return false;
    const auto limit = std::numeric_limits<size_t>::max();
    if (rows > limit / width || offset > limit - rows * width ||
        (src_stride && rows - 1 > (limit - width) / src_stride))
        throw std::overflow_error("RPC weight row extent overflow");
    const size_t per_batch = std::min(rows, RPC_WEIGHT_PACK_LIMIT / width);
    std::vector<unsigned char> packed(per_batch * width);
    const auto * source = static_cast<const unsigned char *>(data);
    for (size_t first = 0; first < rows; ) {
        const size_t count = std::min(per_batch, rows - first);
        for (size_t i = 0; i < count; ++i)
            std::memcpy(packed.data() + i * width, source + (first + i) * src_stride, width);
        sink(offset + first * width, packed.data(), count * width);
        ++calls;
        first += count;
    }
    return true;
}
