use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_lib_path = PathBuf::from(&cuda_path).join("lib64");

    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=nvrtc");

    println!("cargo:rerun-if-changed=build.rs");
}
