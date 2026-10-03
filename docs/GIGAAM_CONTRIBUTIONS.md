# Handy-GigaAM: contribution and verification

Handy is developed upstream by [CJ Pais and contributors](https://github.com/cjpais/Handy). This repository is an unofficial derivative under the original MIT license. GigaAM is now supported upstream; this repository records the integration experiment and additional Windows build work.

## Published changes

[Commit 922cfce](https://github.com/Servideus/Handy-GigaAM/commit/922cfceff18f9f42026afbb0c47cfdfb671aa731) adds Windows Whisper Vulkan configuration, Ninja for nested CMake builds and a short Cargo target directory to avoid Windows path limits. It also includes GigaAM model/UI adjustments. These changes are not a claim to authorship of Handy or GigaAM.

## Local work not published

The working copy has additional changes to model selection and transcription: a multilingual FP16 GigaAM configuration and a separate ONNX punctuation stage, with changes to dependencies and EN/RU strings. The new `punctuation.rs` was untracked at the start of this review. These files have been preserved, not included in this documentation commit, and not presented as a verified release.

## Verification on 2026-10-03

- `npm run build`: TypeScript and Vite build passed.
- `npm run lint`: passed.
- `cargo fmt --manifest-path src-tauri/Cargo.toml -- --check`: passed.
- `cargo clippy --manifest-path src-tauri/Cargo.toml --locked`: stopped while compiling dependencies because `link.exe` was unavailable.
- `npm run format:check`: reported formatting differences in 181 existing files. No repository-wide formatting was applied.
- Full Tauri debug build with a temporary npm frontend command: reached Rust compilation, then failed because `link.exe` was unavailable. Visual Studio locator did not report an installed toolchain. A frontend build does not verify desktop recording, GigaAM inference or insertion.

Windows builds require MSVC C++ Build Tools, WebView2 and the Vulkan toolchain for this configuration. Use a Visual Studio x64 Developer terminal with `cl.exe`, `link.exe`, Ninja and Vulkan tools available, then follow [BUILD.md](../BUILD.md). The existing `.cargo/config.toml` selects `C:/ht`; local verification used `CARGO_TARGET_DIR=D:/ht` without changing that committed configuration.

The current full build and the local punctuation/model changes remain unverified. Existing release binaries are retained; none was replaced by an unverified new build.
