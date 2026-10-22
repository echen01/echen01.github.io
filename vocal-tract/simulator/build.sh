#!/bin/sh
set -eu
cd "$(dirname "$0")"
cargo test --offline
cargo build --offline --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/difftlm_browser.wasm difftlm.wasm
node tests/test.mjs
