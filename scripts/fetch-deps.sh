#!/usr/bin/env bash
# Downloads what Courier uses but does not keep in git, checks each file's SHA-256 and unpacks it into
# third_party/ (ignored by git):
#   - sources: whisper.cpp, llama.cpp (they share one ggml), miniaudio;
#   - onnxruntime for this OS (Silero VAD, the wake word model and the Piper voice run on it);
#   - models: Whisper small.en and base.en, Silero VAD, the Piper LJSpeech voice, CMUdict.
#
#   scripts/fetch-deps.sh            the above
#   scripts/fetch-deps.sh --ci       without Whisper small.en; with Qwen2.5-0.5B for the language tests
#   scripts/fetch-deps.sh --llm      also Qwen2.5-3B-Instruct Q4_K_M (the assistant's model) into models/
#   scripts/fetch-deps.sh --llm7     also Qwen2.5-7B-Instruct Q4_K_M (the comparison) into models/
#   scripts/fetch-deps.sh --train    also the voices for wake word training and its evaluation
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tp="$root/third_party"
dl="$tp/downloads"
mkdir -p "$dl" "$tp/models" "$root/models"
ci=0 llm=0 llm7=0 train=0
for a in "$@"; do
  case "$a" in
    --ci) ci=1 ;;
    --llm) llm=1 ;;
    --llm7) llm7=1 ;;
    --train) train=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

fetch() { # url sha256 file
  local url="$1" sha="$2" file="$dl/$3"
  if [ ! -f "$file" ] || ! echo "$sha  $file" | sha256sum -c - >/dev/null 2>&1; then
    echo "downloading $url"
    curl -fsSL --retry 3 -o "$file.part" "$url" || { echo "download failed: $url" >&2; exit 1; }
    echo "$sha  $file.part" | sha256sum -c - >/dev/null || { echo "checksum mismatch: $url" >&2; rm -f "$file.part"; exit 1; }
    mv "$file.part" "$file"
  fi
}

unpack() { # file dir-inside-archive target
  local file="$dl/$1" target="$tp/$3"
  [ -d "$target" ] && return
  local tmp; tmp="$(mktemp -d)"
  case "$file" in
    *.zip) unzip -q "$file" -d "$tmp" ;;
    *) tar -xzf "$file" -C "$tmp" ;;
  esac
  mv "$tmp/$2" "$target"
  rm -rf "$tmp"
}

model() { # file in downloads -> third_party/models
  cp -u "$dl/$1" "$tp/models/$1"
}

fetch https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v1.9.5.tar.gz \
  ff1a9053feb509ff9d7729703355541ae9690073a6b1c40eb692c962e0dc1720 whisper.cpp-1.9.5.tar.gz
unpack whisper.cpp-1.9.5.tar.gz whisper.cpp-1.9.5 whisper.cpp

fetch https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11429.tar.gz \
  0487b2e58d0f87c50def234ebda5dd8c1ab6c7afbc440ebef0080203110db6d7 llama.cpp-b11429.tar.gz
unpack llama.cpp-b11429.tar.gz llama.cpp-b11429 llama.cpp

fetch https://raw.githubusercontent.com/mackron/miniaudio/0.11.25/miniaudio.h \
  ac7af4de748b7e26b777f37e01cee313a308a7296a3eb080e2906b320cc55c89 miniaudio-0.11.25.h
mkdir -p "$tp/miniaudio" && cp -u "$dl/miniaudio-0.11.25.h" "$tp/miniaudio/miniaudio.h"

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    fetch https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-win-x64-1.31.1.zip \
      5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c onnxruntime-win-x64-1.31.1.zip
    unpack onnxruntime-win-x64-1.31.1.zip onnxruntime-win-x64-1.31.1 onnxruntime ;;
  Linux)
    fetch https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz \
      c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9 onnxruntime-linux-x64-1.31.1.tgz
    unpack onnxruntime-linux-x64-1.31.1.tgz onnxruntime-linux-x64-1.31.1 onnxruntime ;;
esac

# Speech detection: Silero VAD v6 (MIT).
fetch https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx \
  1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3 silero_vad-6.2.3.onnx
model silero_vad-6.2.3.onnx

# The voice reads phonemes made from the CMU pronouncing dictionary (no espeak).
fetch https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict \
  81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22 cmudict.dict
model cmudict.dict

piper=https://huggingface.co/rhasspy/piper-voices/resolve/main/en
fetch "$piper/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx" \
  6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22 en_US-ljspeech-medium.onnx
fetch "$piper/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json" \
  141d612cc0a95ed7efc1ca936b845c2364967f2e9217c5dbfcf69fc4d6c65860 en_US-ljspeech-medium.onnx.json
model en_US-ljspeech-medium.onnx
model en_US-ljspeech-medium.onnx.json

whisper=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
fetch "$whisper/ggml-base.en-q5_1.bin" 4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f ggml-base.en-q5_1.bin
model ggml-base.en-q5_1.bin
if [ $ci = 0 ]; then
  fetch "$whisper/ggml-small.en-q5_1.bin" bfdff4894dcb76bbf647d56263ea2a96645423f1669176f4844a1bf8e478ad30 ggml-small.en-q5_1.bin
  model ggml-small.en-q5_1.bin
fi

# Language models (GGUF). The CI one is small enough to download on every cold cache.
if [ $ci = 1 ]; then
  fetch https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db qwen2.5-0.5b-instruct-q4_k_m.gguf
  model qwen2.5-0.5b-instruct-q4_k_m.gguf
fi
if [ $llm = 1 ]; then
  fetch https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
    9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94 Qwen2.5-3B-Instruct-Q4_K_M.gguf
  mv -f "$dl/Qwen2.5-3B-Instruct-Q4_K_M.gguf" "$root/models/"
fi
if [ $llm7 = 1 ]; then
  fetch https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423 Qwen2.5-7B-Instruct-Q4_K_M.gguf
  mv -f "$dl/Qwen2.5-7B-Instruct-Q4_K_M.gguf" "$root/models/"
fi

# Wake word training: LibriTTS-R (904 speakers) makes the training voices; the evaluation uses voices the
# model never heard: VCTK (109 British speakers), L2-ARCTIC (24 speakers with Arabic, Mandarin, Hindi,
# Korean, Spanish and Vietnamese accents), CMU ARCTIC (18) and six single-speaker voices.
if [ $train = 1 ]; then
  voice() { # path-under-en name sha256-of-onnx sha256-of-json
    fetch "$piper/$1/$2.onnx" "$3" "$2.onnx"
    fetch "$piper/$1/$2.onnx.json" "$4" "$2.onnx.json"
    model "$2.onnx"
    model "$2.onnx.json"
  }
  voice en_US/libritts_r/medium en_US-libritts_r-medium 10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195 \
    b471dc60d2d8335e819c393d196d6fbf792817f40051257b269878505bc9afb3
  voice en_GB/vctk/medium en_GB-vctk-medium 4e9fc85ab9009385319fc6bae7f55577f8a2d7ee77fd9159a5500eb6531f41e6 \
    7f85e6391ed0f7f46e4abd19345929a16be931a0c9945086f96692dce2087fa8
  voice en_US/l2arctic/medium en_US-l2arctic-medium d89f6f124bf1e7735b2179d2141b8001c3e19169d5e743ed6e35624f4c76f044 \
    a97e2ba653e9efcdc1bdcec64a398c8beb19ae5e8dfdbfe4ad6841983e56c07c
  voice en_US/arctic/medium en_US-arctic-medium 483303e294947a3ec2f910ea96093d876e1640f5772e9d89e511d6c82c667286 \
    db2ca1a55db01cdd3ce28ae63037ac525133e9e00ca557430dec572643235efe
  voice en_US/joe/medium en_US-joe-medium 58afce0321b8d9c46d7cdf9c16500cc55a793b4220212dba6b70fb788b3baf06 \
    3d6d5410b3795cb1950595247ef8f06190719e6fdbfa3a2356d8ec368e1aad33
  voice en_US/kristin/medium en_US-kristin-medium 5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4 \
    5681426d4aead22195de70531eeeeddb46493cfaffc5764b2ea3db73428b651c
  voice en_US/bryce/medium en_US-bryce-medium dc9caa6c313199ffb5ac698b6e542fa6cba388aeaf2731e25262e33b9810aef1 \
    7ceb1bc4af6d4e41b6d1edbb86c67e91e01eaa71f66db4cd0ae92ac704d415be
  voice en_US/norman/medium en_US-norman-medium b9739443232a80a59c7d18810dd856899bf16a7964725f5ab81ea49b1351cb71 \
    6c2db7f558a4a8deb9fe822583c1c5105f6c4e834dd0f9de8ad17a888ee9fe1d
  voice en_US/john/medium en_US-john-medium 789c6c875726e627ddee93d51d8727859abe9c091c3d141591f4b83c2072e988 \
    af60f177b6b550f3d7a302720c0fb89e7f94a82b5dca464775ef63b1c69ba09a
  voice en_GB/cori/medium en_GB-cori-medium 1899f98e5fb8310154f3c2973f4b8a929ba7245e722b3d3a85680b833d95f10d \
    e262c16d7f192f69d4edd6b4ef8a5915379e67495fcc402f1ab15eeb33da3d36
fi
echo "dependencies are in $tp"
