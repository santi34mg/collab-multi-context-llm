use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("could not execute {command:?}: {error}"));
    assert!(status.success(), "{command:?} failed with {status}");
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_NATIVE");
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }
    for name in ["LLAMA_CPP_DIR", "CC", "AR"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=native/bridge.c");
    println!("cargo:rerun-if-changed=vendor/llama.cpp/include");
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("linux"),
        "the native bridge currently supports Linux shared-library installations"
    );
    let root = PathBuf::from(env::var_os("LLAMA_CPP_DIR").expect(
        "native requires LLAMA_CPP_DIR: the explicit directory containing the compatible llama.cpp b11669 shared libraries (commit 620d2225c)",
    ));
    let lib = [root.clone(), root.join("lib"), root.join("lib64")]
        .into_iter()
        .find(|path| path.join("libllama.so").is_file())
        .unwrap_or_else(|| panic!("no libllama.so in {}, lib, or lib64", root.display()))
        .canonicalize()
        .expect("cannot resolve LLAMA_CPP_DIR");
    let lib_text = lib.to_str().expect("LLAMA_CPP_DIR must be UTF-8");
    assert!(
        !lib_text.contains(['\n', '\r', ',']),
        "LLAMA_CPP_DIR cannot contain newlines or commas"
    );
    for name in ["llama", "ggml", "ggml-base"] {
        let path = lib.join(format!("lib{name}.so"));
        assert!(path.is_file(), "missing {}", path.display());
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR missing"));
    let object = out.join("llama_bridge.o");
    let archive = out.join("libagent_llama_bridge.a");
    let cc = env::var_os("CC").unwrap_or_else(|| "cc".into());
    let ar = env::var_os("AR").unwrap_or_else(|| "ar".into());
    run(Command::new(cc)
        .args(["-std=c11", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror"])
        .arg("-I")
        .arg(Path::new("vendor/llama.cpp/include"))
        .arg("-c")
        .arg("native/bridge.c")
        .arg("-o")
        .arg(&object));
    run(Command::new(ar).arg("crs").arg(&archive).arg(&object));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=agent_llama_bridge");
    println!("cargo:rustc-link-search=native={lib_text}");
    for name in ["llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_text}");
    println!("cargo:rustc-env=AGENT_LLAMA_CPP_LIB_DIR={lib_text}");
}
