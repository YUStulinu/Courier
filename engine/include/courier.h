/* Courier's engine: the local models behind a C API, for the Rust orchestrator and the Python wake
 * word training (ctypes).
 *
 *   speech detection   Silero VAD (onnxruntime), or an energy detector to compare against
 *   wake word          Google's speech embedding and a small classifier trained in wakeword/ (onnxruntime)
 *   recognition        whisper.cpp
 *   language model     llama.cpp, streaming, with a JSON grammar switched on inside tool calls
 *   voice              Piper (onnxruntime), phonemes from the CMU dictionary
 *
 * Every function that can fail returns NULL or a negative number; courier_last_error() says why (per
 * thread). Strings returned by the engine are freed with courier_free_string, audio with courier_free_pcm.
 * Audio in is 16 kHz mono float in [-1, 1]. */
#ifndef COURIER_H
#define COURIER_H

#include <stdint.h>

#ifdef _WIN32
#ifdef COURIER_BUILDING
#define COURIER_API __declspec(dllexport)
#else
#define COURIER_API __declspec(dllimport)
#endif
#else
#define COURIER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

COURIER_API const char* courier_last_error(void);
COURIER_API void courier_free_string(char* s);
COURIER_API void courier_free_pcm(int16_t* pcm);
/* Whether the language and speech models were built for an NVIDIA GPU (CUDA). */
COURIER_API int courier_has_gpu(void);

/* --- speech detection --------------------------------------------------------------------------
 * Frames of 512 samples (32 ms). `model` is Silero's ONNX file, or NULL for the energy detector.
 * process() buffers what does not fill a frame and writes one speech probability (0-1) per finished
 * frame into `probs`, at most `max` of them; it returns how many it wrote. */
typedef struct courier_vad courier_vad;
COURIER_API courier_vad* courier_vad_new(const char* model);
COURIER_API int courier_vad_process(courier_vad* v, const float* samples, int n, float* probs, int max);
COURIER_API void courier_vad_reset(courier_vad* v);
COURIER_API void courier_vad_free(courier_vad* v);

/* --- wake word ----------------------------------------------------------------------------------
 * Speech features exactly as the detector computes them (for training): a 96-number speech
 * embedding every 80 ms (Google's speech_embedding over 32-band log-mel frames). `folder` holds
 * speech_mel.onnx and speech_embedding.onnx. process() streams like the detector does and writes up to
 * `max` embeddings of 96 floats; it returns how many. */
#define COURIER_EMBEDDING_DIM 96
typedef struct courier_features courier_features;
COURIER_API courier_features* courier_features_new(const char* folder);
COURIER_API int courier_features_process(courier_features* f, const float* samples, int n, float* out, int max);
COURIER_API void courier_features_reset(courier_features* f);
COURIER_API void courier_features_free(courier_features* f);

/* The detector scores the last 16 embeddings (about 2 s of audio) every 80 ms. It fires when `hits` scores in a row reach
 * `threshold`, then stays quiet for `refractory_ms`. process() writes the scores it computed into
 * `scores` (at most `max`, may be NULL) and returns how many detections happened in this audio (0 or
 * more); courier_wake_last_ms() says where the last one was, in ms since the stream started. */
typedef struct courier_wake courier_wake;
COURIER_API courier_wake* courier_wake_new(const char* model, float threshold, int hits, int refractory_ms);
COURIER_API int courier_wake_process(courier_wake* w, const float* samples, int n, float* scores, int max, int* n_scores);
COURIER_API double courier_wake_last_ms(const courier_wake* w);
COURIER_API void courier_wake_set_threshold(courier_wake* w, float threshold);
COURIER_API void courier_wake_reset(courier_wake* w);
COURIER_API void courier_wake_free(courier_wake* w);

/* --- recognition ---------------------------------------------------------------------------------
 * `gpu`: use the GPU if the engine was built with one. Returns JSON: {"text": ..., "ms": ...}.
 * `fast`: shorten Whisper's encoder window to the audio's length (faster, a little less accurate). */
typedef struct courier_asr courier_asr;
COURIER_API courier_asr* courier_asr_new(const char* model, int threads, int gpu);
COURIER_API char* courier_asr_transcribe(courier_asr* a, const float* samples, int n, int fast);
COURIER_API void courier_asr_free(courier_asr* a);

/* --- language model -----------------------------------------------------------------------------
 * `prompt` is the whole chat-formatted prompt (the orchestrator writes it). Generation is greedy. The
 * context keeps what it computed: a prompt that starts like the last one is computed only from where
 * they differ.
 * When the output contains `trigger` (e.g. "<tool_call>"), what follows is constrained to the JSON
 * schema `tool_schema` and generation stops as soon as that JSON object is closed. Either may be NULL.
 * `on_piece` gets the text as it is generated, whole UTF-8 characters at a time; returning nonzero
 * stops generation. Returns JSON: {"text", "stop": "end"|"tool"|"length"|"cancel", "prompt_tokens",
 * "reused_tokens", "generated", "ms_prompt", "ms_first", "ms"}. */
typedef struct courier_llm courier_llm;
typedef int (*courier_piece_cb)(const char* piece, void* user);
COURIER_API courier_llm* courier_llm_new(const char* model, int context, int gpu_layers);
COURIER_API char* courier_llm_generate(courier_llm* l, const char* prompt, const char* trigger, const char* tool_schema,
                                       int max_tokens, courier_piece_cb on_piece, void* user);
COURIER_API int courier_llm_count_tokens(courier_llm* l, const char* text);
COURIER_API void courier_llm_free(courier_llm* l);

/* --- voice ----------------------------------------------------------------------------------------
 * A Piper voice (.onnx and its .onnx.json) and the CMU dictionary. speak() writes 16-bit PCM at the
 * voice's rate into *pcm (free with courier_free_pcm) and returns the number of samples. `speaker` is
 * for multi-speaker voices (0 otherwise); `speed` > 1 talks faster; `variation` scales the voice's
 * own noise settings (1 = as trained). */
typedef struct courier_tts courier_tts;
COURIER_API courier_tts* courier_tts_new(const char* voice, const char* config, const char* cmudict, int threads);
COURIER_API int courier_tts_rate(const courier_tts* t);
COURIER_API int courier_tts_speakers(const courier_tts* t);
COURIER_API int courier_tts_speak(courier_tts* t, const char* text, int speaker, float speed, float variation, int16_t** pcm);
/* The text as the voice reads it ("at 6:30 PM" -> "at six thirty p m") and its phonemes, as JSON
 * {"text": ..., "ipa": ...}. */
COURIER_API char* courier_tts_explain(courier_tts* t, const char* text);
COURIER_API void courier_tts_free(courier_tts* t);

#ifdef __cplusplus
}
#endif

#endif
