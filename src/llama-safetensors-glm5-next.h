#pragma once
#include "llama-safetensors-importer.h"
#include "llama-safetensors-quant.h"
#include "llama-safetensors-tensor.h"
#include <map>
#include <memory>

class llama_safetensors_glm5_next_importer final : public llama_safetensors_importer {
public:
    llama_safetensors_glm5_next_importer(const std::filesystem::path &, llama_safetensors_json,
                                       llama_safetensors_io_mode = llama_safetensors_io_mode::MMAP);
    static bool probe(const llama_safetensors_json &);
    gguf_context * build_metadata() const override;
    bool describe(const std::string &, ggml_type &, std::array<int64_t, GGML_MAX_DIMS> &) const override;
    size_t tensor_capacity_hint() const override;
    void bind(const std::string &) const override;
    bool load(const std::string &, ggml_tensor *, bool) const override;
    std::vector<uint8_t> materialize(const std::string &, ggml_type, size_t) const override;
    void validate_complete() const override;
private:
    struct binding {
        std::vector<llama_safetensors_tensor_binding> parts;
        enum transform { NONE, EXP_A, MLA_K, MLA_V } op = NONE;
        bool stack = false;
        bool force_f32 = false;
    };
    binding resolve(const std::string &) const;
    std::filesystem::path root_;
    llama_safetensors_json config_, text_, generation_;
    llama_safetensors_tokenizer_json tokenizer_;
    std::optional<std::string> chat_template_;
    llama_safetensors_registry registry_;
    std::unique_ptr<llama_safetensors_quant_adapters> quant_;
    uint32_t layers_, mtp_, experts_;
};
