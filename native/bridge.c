/* Parameter structs and llama-owned allocations never cross the Rust ABI. */
#include "llama.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct ac_model {
    struct llama_model * model;
    struct llama_context * ctx;
    struct llama_sampler * sampler;
    const struct llama_vocab * vocab;
};

static int ac_error(char * error, size_t capacity, const char * message) {
    if (capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
    return -1;
}

/* Called exactly once by Rust. Backends remain loaded for the process lifetime. */
int ac_init(const char * directory, char * error, size_t capacity) {
    const char * version = llama_version();
    if (!version || strcmp(version, "0.7.0-dev") != 0) {
        return ac_error(error, capacity,
            "incompatible llama.cpp version; require 0.7.0-dev b11669, commit 620d2225c");
    }
    ggml_backend_load_all_from_path(directory);
    ggml_backend_load_all();
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        return ac_error(error, capacity,
            "no CPU backend loaded; LLAMA_CPP_DIR must include compatible libggml-cpu-*.so plugins");
    }
    llama_backend_init();
    return 0;
}

void ac_reset(struct ac_model * handle) {
    if (handle->sampler) {
        llama_sampler_free(handle->sampler);
        handle->sampler = NULL;
    }
    /* Clearing only sequence metadata is insufficient for hybrid recurrent state. */
    llama_memory_t memory = llama_get_memory(handle->ctx);
    if (memory) {
        llama_memory_clear(memory, true);
    }
}

void ac_free(struct ac_model * handle) {
    if (!handle) {
        return;
    }
    if (handle->sampler) {
        llama_sampler_free(handle->sampler);
    }
    if (handle->ctx) {
        llama_free(handle->ctx);
    }
    if (handle->model) {
        llama_model_free(handle->model);
    }
    free(handle);
}

struct ac_model * ac_load(const char * path, uint32_t context_size,
        uint32_t batch_size, int32_t threads, int32_t gpu_layers,
        char * error, size_t capacity) {
    struct ac_model * handle = calloc(1, sizeof(*handle));
    if (!handle) {
        ac_error(error, capacity, "cannot allocate native model handle");
        return NULL;
    }
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    /* An empty device list also prevents opportunistic GPU allocations on CPU runs. */
    ggml_backend_dev_t cpu_only[] = { NULL };
    if (gpu_layers == 0) {
        mp.devices = cpu_only;
    }
    handle->model = llama_model_load_from_file(path, mp);
    if (!handle->model) {
        ac_error(error, capacity, "llama_model_load_from_file failed; check GGUF, architecture support, and llama.cpp diagnostics");
        ac_free(handle);
        return NULL;
    }
    handle->vocab = llama_model_get_vocab(handle->model);
    if (!handle->vocab || llama_model_has_encoder(handle->model)) {
        ac_error(error, capacity, "native inference requires a decoder-only model with a vocabulary");
        ac_free(handle);
        return NULL;
    }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = context_size;
    cp.n_batch = batch_size;
    cp.n_ubatch = batch_size;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    if (gpu_layers == 0) {
        cp.offload_kqv = false;
        cp.op_offload = false;
    }
    handle->ctx = llama_init_from_model(handle->model, cp);
    if (!handle->ctx || !llama_get_memory(handle->ctx)) {
        ac_error(error, capacity, "llama_init_from_model failed or model has no autoregressive memory; see llama.cpp diagnostics");
        ac_free(handle);
        return NULL;
    }
    return handle;
}

uint32_t ac_context_size(const struct ac_model * handle) {
    return llama_n_ctx(handle->ctx);
}

uint32_t ac_batch_size(const struct ac_model * handle) {
    return llama_n_batch(handle->ctx);
}

int32_t ac_metadata(const struct ac_model * handle, const char * key,
        char * output, size_t capacity) {
    return llama_model_meta_val_str(handle->model, key, output, capacity);
}

/* Borrowed until ac_free; Rust copies this before returning from its method. */
const char * ac_template_source(const struct ac_model * handle) {
    return llama_model_chat_template(handle->model, NULL);
}

int32_t ac_template(const struct ac_model * handle,
        const char * const * roles, const char * const * contents, size_t count,
        char * output, int32_t capacity) {
    const char * tmpl = llama_model_chat_template(handle->model, NULL);
    if (!tmpl || count == 0 || count > SIZE_MAX / sizeof(struct llama_chat_message)) {
        return -1;
    }
    struct llama_chat_message * messages = calloc(count, sizeof(*messages));
    if (!messages) {
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        messages[i].role = roles[i];
        messages[i].content = contents[i];
    }
    int32_t result = llama_chat_apply_template(tmpl, messages, count, true, output, capacity);
    free(messages);
    return result;
}

int32_t ac_tokenize(const struct ac_model * handle, const char * text,
        int32_t length, int32_t * tokens, int32_t capacity) {
    return llama_tokenize(handle->vocab, text, length, tokens, capacity, true, true);
}

int32_t ac_piece(const struct ac_model * handle, int32_t token, char * output, int32_t capacity) {
    return llama_token_to_piece(handle->vocab, token, output, capacity, 0, true);
}

static int ac_add_sampler(struct ac_model * handle, struct llama_sampler * sampler) {
    if (!sampler) {
        return -1;
    }
    llama_sampler_chain_add(handle->sampler, sampler);
    return 0;
}

int ac_begin(struct ac_model * handle, float temperature, float top_p,
        uint32_t seed, char * error, size_t capacity) {
    ac_reset(handle);
    handle->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!handle->sampler) {
        return ac_error(error, capacity, "cannot allocate sampler chain");
    }
    int status;
    if (temperature == 0.0f) {
        status = ac_add_sampler(handle, llama_sampler_init_greedy());
    } else {
        status = ac_add_sampler(handle, llama_sampler_init_temp(temperature));
        if (status == 0) {
            status = ac_add_sampler(handle, llama_sampler_init_top_p(top_p, 1));
        }
        if (status == 0) {
            status = ac_add_sampler(handle, llama_sampler_init_dist(seed));
        }
    }
    if (status != 0) {
        ac_reset(handle);
        return ac_error(error, capacity, "cannot allocate generation sampler");
    }
    return 0;
}

int ac_decode(struct ac_model * handle, const int32_t * tokens, int32_t count,
        char * error, size_t capacity) {
    if (count <= 0 || (uint32_t) count > llama_n_batch(handle->ctx)) {
        return ac_error(error, capacity, "decode batch is empty or exceeds llama_n_batch");
    }
    /* llama_batch_get_one borrows tokens for this synchronous decode only. */
    struct llama_batch batch = llama_batch_get_one((llama_token *) tokens, count);
    int status = llama_decode(handle->ctx, batch);
    if (status != 0) {
        if (capacity > 0) {
            snprintf(error, capacity, "llama_decode failed with status %d (batch=%d); context capacity, allocation, or backend error", status, count);
        }
        return -1;
    }
    return 0;
}

int32_t ac_sample(struct ac_model * handle) {
    return llama_sampler_sample(handle->sampler, handle->ctx, -1);
}

int ac_is_eog(const struct ac_model * handle, int32_t token) {
    return llama_vocab_is_eog(handle->vocab, token) ? 1 : 0;
}
