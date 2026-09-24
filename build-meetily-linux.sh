#!/usr/bin/env bash
# Build Meetily v0.4.1 from source on Pop!_OS 24.04 with NVIDIA CUDA and install it per-user.
#
#   ./build-meetily-linux.sh
#
# Steps:
#   1. apt-installs the missing build deps (sudo; only if something is missing)
#   2. builds the llama-helper sidecar and the Tauri app with CUDA for sm_86 (RTX A1000)
#   3. unpacks the resulting .deb into ~/.local/opt/meetily (no files under /usr)
#   4. adds a `meetily` launcher in ~/.local/bin and an app-menu entry
#   5. adds an ALSA loopback device to ~/.asoundrc so "System Audio" capture works
#
# Env overrides: CUDA_ARCH (default 86), INSTALL_DIR (default ~/.local/opt/meetily),
#                SKIP_ASOUNDRC=1 to leave ~/.asoundrc alone.
#
# Why not upstream's build-gpu.sh:
#   - it and scripts/tauri-auto.js force CMAKE_CUDA_ARCHITECTURES=75 (this GPU is 8.6)
#   - `tauri build` exits non-zero at the end without an updater signing key (needs --no-sign)
#   - Ubuntu's CUDA puts cudart_static in /usr/lib/x86_64-linux-gnu, which llama-cpp-sys-2's
#     CUDA lookup never searches (upstream PR #687)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Absolute, no trailing slash: "$INSTALL_DIR.new" must be a sibling, and the launcher needs an
# absolute path. -s so a symlinked INSTALL_DIR is replaced, not its target.
INSTALL_DIR="$(realpath -ms -- "${INSTALL_DIR:-$HOME/.local/opt/meetily}")"
CUDA_ARCH="${CUDA_ARCH:-86}"
TRIPLE="x86_64-unknown-linux-gnu"
SECONDS=0

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

is_installed() { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'install ok installed'; }

# ---------------------------------------------------------------------------------------------
step "1/6 System packages"
# cmake: whisper.cpp + llama.cpp builds. libclang-dev: bindgen (v18 is known-good; newer breaks
# whisper bindings, upstream #428). libasound2-dev: cpal/alsa-sys. nvidia-cuda-toolkit: CUDA 12.0;
# nvcc 12.0 rejects gcc 13 and uses its own gcc-12 wrappers, hence g++-12. libcublas12,
# libcublaslt12, libcudart12: the meetily binary loads these at run time, so they are installed
# explicitly to survive a later toolkit removal.
# Do NOT install nvidia-driver-550 (upstream docs) or libnccl-dev (breaks the llama.cpp link).
APT_PKGS=(cmake libclang-dev libasound2-dev nvidia-cuda-toolkit g++-12 libcublas12 libcublaslt12 libcudart12)
missing=()
for p in "${APT_PKGS[@]}"; do is_installed "$p" || missing+=("$p"); done
if ((${#missing[@]})); then
  echo "Installing: ${missing[*]} (~1.5 GB download, ~5 GB on disk for the CUDA toolkit)"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${missing[@]}"
else
  echo "All build packages already installed."
fi

# ---------------------------------------------------------------------------------------------
step "2/6 Build environment"
# Homebrew (and asdf/pyenv/bun) sit ahead of /usr/bin on this machine. CMake searches
# <dir>/../lib for every PATH entry, so Homebrew's openssl/zlib/zstd could leak into llama.cpp's
# build. Use a minimal PATH: cargo, the nvm node dir, and the system dirs.
node_path="$(command -v node)" || die "node not found (nvm)"
node_bin="$(dirname "$node_path")"
export PATH="$HOME/.cargo/bin:$node_bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PKG_CONFIG=/usr/bin/pkg-config
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig:/usr/lib/pkgconfig
export LIBCLANG_PATH=/usr/lib/llvm-18/lib
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" CMAKE_CUDA_STANDARD=17 CMAKE_POSITION_INDEPENDENT_CODE=ON
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS TAURI_GPU_FEATURE CUDA_PATH CUDAHOSTCXX NVCC_CCBIN CC CXX

command -v nvcc  >/dev/null || die "nvcc not found after installing nvidia-cuda-toolkit"
command -v cmake >/dev/null || die "cmake not found"
command -v cargo >/dev/null || die "cargo not found (rustup)"
for f in "$LIBCLANG_PATH/libclang.so" /usr/lib/x86_64-linux-gnu/libcudart_static.a \
         /usr/lib/x86_64-linux-gnu/libculibos.a /usr/lib/x86_64-linux-gnu/libcublas.so.12; do
  [[ -e "$f" ]] || die "missing $f"
done
pkg-config --exists alsa || die "pkg-config cannot find alsa (libasound2-dev)"
if ! command -v pnpm >/dev/null; then
  echo "Installing pnpm 9.15.9 (the version upstream CI pins) into the nvm prefix"
  npm install -g pnpm@9.15.9
fi
echo "nvcc:  $(nvcc --version | tail -1)"
echo "cmake: $(cmake --version | head -1)"
echo "CUDA arch: sm_$CUDA_ARCH"

# The sys crates don't rebuild when CMAKE_* env changes, so clear them if the arch changed.
cd "$REPO_DIR"
arch_stamp="target/.meetily-cuda-arch"
if [[ -f "$arch_stamp" && "$(cat "$arch_stamp")" != "$CUDA_ARCH" ]]; then
  echo "CUDA arch changed since last build; cleaning whisper/llama native builds"
  cargo clean --release -p whisper-rs-sys -p llama-cpp-sys-2
fi

# ---------------------------------------------------------------------------------------------
step "3/6 Frontend dependencies"
(cd frontend && pnpm install --frozen-lockfile)

# ---------------------------------------------------------------------------------------------
step "4/6 llama-helper sidecar (CUDA)"
# llama-cpp-sys-2 links cudart_static/cublas_static/culibos, which rustc must find itself, and
# Ubuntu keeps them in /usr/lib/x86_64-linux-gnu. This -L is scoped to llama-helper on purpose:
# applied to the main app it would swap in the system libzstd/libsqlite3/libbz2 .a files for the
# bundled ones. target-cpu=native turns on GGML_NATIVE (AVX2/VNNI) for layers left on the CPU.
helper_rustflags="-L native=/usr/lib/x86_64-linux-gnu -C target-cpu=native"
if ! RUSTFLAGS="$helper_rustflags" cargo build --release --locked -p llama-helper --features cuda; then
  # llama.cpp compiles cpp-httplib with OpenSSL but llama-cpp-sys-2 never links it.
  warn "llama-helper build failed; retrying with -lssl -lcrypto"
  RUSTFLAGS="$helper_rustflags -C link-arg=-lssl -C link-arg=-lcrypto" \
    cargo build --release --locked -p llama-helper --features cuda
fi
mkdir -p frontend/src-tauri/binaries target
cp target/release/llama-helper "frontend/src-tauri/binaries/llama-helper-$TRIPLE"
echo "$CUDA_ARCH" > "$arch_stamp"

# ---------------------------------------------------------------------------------------------
step "5/6 Meetily app (CUDA, .deb bundle)"
# build.rs downloads a static ffmpeg into src-tauri/binaries on first build; next build fetches
# Google Fonts; ort downloads ONNX Runtime. Network access is required.
(cd frontend && pnpm tauri build --no-sign --bundles deb --features cuda)

deb="$(ls -t target/release/bundle/deb/*.deb 2>/dev/null | head -1 || true)"
[[ -n "$deb" ]] || die "no .deb produced under target/release/bundle/deb"
echo "Built: $deb"
for cache in target/release/build/{whisper-rs-sys,llama-cpp-sys-2}-*/out/build/CMakeCache.txt; do
  [[ -f "$cache" ]] || continue
  grep -q "^CMAKE_CUDA_ARCHITECTURES:.*=$CUDA_ARCH\$" "$cache" \
    || warn "$cache does not target sm_$CUDA_ARCH: $(grep '^CMAKE_CUDA_ARCHITECTURES' "$cache")"
done

# ---------------------------------------------------------------------------------------------
step "6/6 Install to $INSTALL_DIR"
if [[ -e "$INSTALL_DIR" && ! -x "$INSTALL_DIR/usr/bin/meetily" ]]; then
  die "$INSTALL_DIR exists but is not a Meetily install; refusing to overwrite"
fi
rm -rf "$INSTALL_DIR.new"
mkdir -p "$INSTALL_DIR.new"
dpkg-deb -x "$deb" "$INSTALL_DIR.new"
rm -rf "$INSTALL_DIR"
mv "$INSTALL_DIR.new" "$INSTALL_DIR"
# Tauri resolves resources (templates/) at <exe>/../lib/meetily; the sidecars sit next to the exe.
[[ -d "$INSTALL_DIR/usr/lib/meetily" ]] || warn "resource dir $INSTALL_DIR/usr/lib/meetily missing"
for b in meetily ffmpeg llama-helper; do
  [[ -x "$INSTALL_DIR/usr/bin/$b" ]] || die "missing $INSTALL_DIR/usr/bin/$b"
done

if ldd "$INSTALL_DIR/usr/bin/meetily" | grep -q 'not found'; then
  ldd "$INSTALL_DIR/usr/bin/meetily" | grep 'not found' >&2
  die "meetily has unresolved shared libraries"
fi
ldd "$INSTALL_DIR/usr/bin/meetily" | grep -q libcublas \
  && echo "CUDA: meetily links cuBLAS (whisper GPU backend compiled in)" \
  || warn "meetily does not link cuBLAS; the CUDA feature may not have been applied"

mem_gb=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 1048576 ))
(( mem_gb > 255 )) && mem_gb=255   # the app parses MEMORY_GB as u8
mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
cat > "$HOME/.local/bin/meetily" <<EOF
#!/bin/sh
# WebKitGTK's DMABUF renderer can give a blank window on NVIDIA + Wayland (meetily #435).
export WEBKIT_DISABLE_DMABUF_RENDERER=1
# Meetily's hardware tiering (audio/hardware_detector.rs) only detects CUDA via CUDA_PATH,
# CUDA_HOME or /usr/local/cuda, and assumes 8 GB RAM unless MEMORY_GB is set. Without these it
# picks the Low tier for Whisper (no flash attention, beam size 1) despite the CUDA build.
export CUDA_HOME="\${CUDA_HOME:-/usr}"
export MEMORY_GB="\${MEMORY_GB:-$mem_gb}"
exec "$INSTALL_DIR/usr/bin/meetily" "\$@"
EOF
chmod +x "$HOME/.local/bin/meetily"

icon="$(find "$INSTALL_DIR/usr/share/icons" -name '*.png' 2>/dev/null | sort -V | tail -1 || true)"
cat > "$HOME/.local/share/applications/meetily.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Meetily
Comment=Local AI meeting notes
Exec=$HOME/.local/bin/meetily
Icon=${icon:-meetily}
Categories=AudioVideo;Office;
Terminal=false
EOF
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true

# ---------------------------------------------------------------------------------------------
# Linux system-audio capture in v0.4.1 only accepts ALSA devices whose name contains "monitor",
# then looks the name up with a " (System Audio)" suffix (upstream #701). These two PCMs route the
# PipeWire monitor of the current default output through the ALSA pulse plugin under both names.
if [[ "${SKIP_ASOUNDRC:-0}" != 1 ]] && ! grep -qs 'meetily_monitor' "$HOME/.asoundrc"; then
  step "Adding system-audio loopback device to ~/.asoundrc"
  cat >> "$HOME/.asoundrc" <<'EOF'

# Meetily v0.4.1 system-audio capture: the UI lists "<name> (System Audio)" and then looks up
# that exact string, so both names are needed.
pcm.meetily_monitor {
    type pulse
    device "@DEFAULT_MONITOR@"
    hint { show on  description "Speaker loopback (default sink monitor)" }
}
pcm."meetily_monitor (System Audio)" {
    type pulse
    device "@DEFAULT_MONITOR@"
    hint { show on  description "Speaker loopback alias for Meetily lookup" }
}
EOF
fi

printf '\n\033[1;32mDone in %dm%ds.\033[0m\n' $((SECONDS / 60)) $((SECONDS % 60))
cat <<EOF

Launch:   meetily            (or "Meetily" in the app menu; run from a terminal to see logs)
In the app, Settings -> Recordings:
  Microphone:    Default  (or "pipewire")
  System Audio:  meetily_monitor (System Audio)
First run downloads the Parakeet (~670 MB) and summary (~2.7 GB) models.
Data: ~/.local/share/com.meetily.ai   Recordings: ~/Documents/meetily-recordings
EOF
