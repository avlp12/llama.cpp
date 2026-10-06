#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include <memory>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void require(bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); std::exit(1); } }
static std::vector<uint8_t> read(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "fixture file");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
}
// Exercises the engine's SET_ROWS and GET_ROWS dispatch, not direct codec calls.
// Fixtures come from an independent IEEE scalar oracle, reusable with CUDA.
int main(int argc, char ** argv) {
    require(argc == 2, "expected independently generated fixture directory");
    const std::string root = argv[1];
    auto input = read(root+"/input.f32"), packed = read(root+"/expected.packed"), expected = read(root+"/expected.f32");
    require(input.size() == 16*512*4 && packed.size() == 16*656 && expected.size() == input.size(), "fixture sizes");
    require(ggml_row_size(GGML_TYPE_B0_FP8_MLA, 512) == 656, "physical B0 row size");
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    require(ctx != nullptr, "context");
    auto * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_B0_FP8_MLA, 512, 20, 2);
    auto * values = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 8, 2);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 8, 2);
    auto * selection = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 2);
    auto * written = ggml_set_rows(ctx, storage, values, ids);
    auto * result = ggml_get_rows(ctx, written, selection);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * backend = ggml_backend_cpu_init();
    require(backend != nullptr, "CPU backend");
    ggml_backend_cpu_set_n_threads(backend, 2);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "buffer");
    ggml_backend_buffer_clear(buffer, 0);
    int64_t rows[16]; int32_t selects[16];
    for (int s = 0; s < 2; ++s) for (int i = 0; i < 8; ++i) {
        rows[s*8+i] = 18-2*i;
    }
    for (int s = 0; s < 2; ++s) for (int i = 0; i < 8; ++i) {
        selects[s*8+i] = int32_t(rows[s*8+7-i]);
    }
    ggml_backend_tensor_set(values, input.data(), 0, input.size());
    ggml_backend_tensor_set(ids, rows, 0, sizeof(rows));
    ggml_backend_tensor_set(selection, selects, 0, sizeof(selects));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "actual engine graph");
    std::vector<uint8_t> got(40*656), readback(expected.size());
    ggml_backend_tensor_get(storage, got.data(), 0, got.size());
    ggml_backend_tensor_get(result, readback.data(), 0, readback.size());
    for (int s = 0; s < 2; ++s) for (int i = 0; i < 8; ++i) {
        const size_t cache = (s*20+rows[s*8+i])*656;
        require(std::memcmp(got.data()+cache, packed.data()+(s*8+i)*656, 656) == 0, "packed cache vs independent B0 bytes");
        require(std::memcmp(readback.data()+(s*8+i)*512*4, expected.data()+(s*8+7-i)*512*4, 512*4) == 0, "selected FP32 values");
    }
    require(ggml_validate_row_data(GGML_TYPE_B0_FP8_MLA, got.data(), got.size()), "state row validation including cleared rows");
    // A zero scale only denotes a wholly cleared row. A corrupt populated row is refused.
    auto bad = packed;
    std::memset(bad.data()+656+512, 0, 4);
    bad[656+0] = 1;
    require(!ggml_validate_row_data(GGML_TYPE_B0_FP8_MLA, bad.data(), bad.size()), "zero-scale corruption refusal");
    bad = packed; bad[0] = 127;
    require(!ggml_validate_row_data(GGML_TYPE_B0_FP8_MLA, bad.data(), bad.size()), "NaN code refusal");
    bad = packed; bad[528] = 1;
    require(!ggml_validate_row_data(GGML_TYPE_B0_FP8_MLA, bad.data(), bad.size()), "nonzero RoPE refusal");
    // Clear and reuse the same storage with new stream row positions.
    for (int s = 0; s < 2; ++s) for (int i = 0; i < 8; ++i) { rows[s*8+i] = i; selects[s*8+i] = i; }
    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_set(values, input.data(), 0, input.size());
    ggml_backend_tensor_set(ids, rows, 0, sizeof(rows));
    ggml_backend_tensor_set(selection, selects, 0, sizeof(selects));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "storage reuse graph");
    ggml_backend_tensor_get(result, readback.data(), 0, readback.size());
    require(readback == expected, "reuse readback");
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_GLM5_NEXT, llama_model_default_params()));
        require(bool(model), "model metadata fixture");
        auto & h = model->hparams;
        h = {};
        h.n_layer_all = 1; h.n_embd = 64;
        h.n_head_arr[0] = 16; h.n_head_kv_arr[0] = 1;
        h.n_lora_kv = 512;
        h.n_embd_head_k_full = h.n_embd_head_v_full = 512;
        h.n_embd_head_k_mla_impl = h.n_embd_head_v_mla_impl = 512;
        llama_kv_cache cache(*model, h, GGML_TYPE_B0_FP8_MLA, GGML_TYPE_F16,
                false, false, false, 20, 2, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr);
        auto * c = ggml_init({1024*1024, nullptr, true});
        require(cache.build_input_k_rot(c) == nullptr, "B0 main MLA has no extra Hadamard rotation");
        auto * v = ggml_new_tensor_3d(c, GGML_TYPE_F32, 512, 1, 16);
        auto * ix = ggml_new_tensor_1d(c, GGML_TYPE_I64, 16);
        llama_kv_cache::slot_info info{}; info.s0 = 0; info.s1 = 1; info.resize(2);
        auto * write = cache.cpy_k(c, v, ix, 0, info);
        auto * dense = cache.get_k(c, 0, 20, info);
        auto right_info = info; right_info.s0 = right_info.s1 = 1;
        auto * right = cache.get_k(c, 0, 20, right_info);
        auto * g = ggml_new_graph(c);
        ggml_build_forward_expand(g, write); ggml_build_forward_expand(g, dense); ggml_build_forward_expand(g, right);
        auto * buf = ggml_backend_alloc_ctx_tensors(c, backend);
        require(buf != nullptr, "actual cache graph buffer");
        cache.clear(true);
        for (int stream = 0; stream < 2; ++stream) for (int i = 0; i < 8; ++i) rows[stream*8+i] = stream*20+i;
        ggml_backend_tensor_set(v, input.data(), 0, input.size());
        ggml_backend_tensor_set(ix, rows, 0, sizeof(rows));
        require(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS, "actual cpy_k/get_k graph");
        std::vector<uint8_t> dense_values(40*512*4), right_values(20*512*4);
        ggml_backend_tensor_get(dense, dense_values.data(), 0, dense_values.size());
        ggml_backend_tensor_get(right, right_values.data(), 0, right_values.size());
        for (int stream = 0; stream < 2; ++stream) {
            require(std::memcmp(dense_values.data()+stream*20*512*4, expected.data()+stream*8*512*4, 8*512*4) == 0, "real dense cache stream values");
            for (size_t j = (stream*20+8)*512*4; j < size_t(stream+1)*20*512*4; ++j) require(dense_values[j] == 0, "cleared unwritten rows");
        }
        require(std::memcmp(right_values.data(), dense_values.data()+20*512*4, right_values.size()) == 0, "dense stream offset");
        cache.clear(true);
        ggml_backend_tensor_get(cache.get_k_storage(0), got.data(), 0, got.size());
        for (auto byte : got) require(byte == 0, "actual cache clear");
        ggml_backend_buffer_free(buf); ggml_free(c);
    }
    ggml_backend_free(backend);
    std::puts("CPU_B0_FP8_KV_GRAPH_PASS rows16 streams2 packed10496bytes readback8192values corruption3 clear_reuse actual_cache_cpy_dense_stream_offset; no model/GPU");
}
