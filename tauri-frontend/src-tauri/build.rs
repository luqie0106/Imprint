use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let native_root = manifest_dir.join("../../native-geo");
    let header = native_root.join("include/imprint_geo.h");
    let cpp_header = native_root.join("include/imprint_geo.hpp");
    let source = native_root.join("src/geo.cpp");
    let preview_header = native_root.join("include/native_preview.h");
    let preview_source = native_root.join("src/native_preview.cpp");
    let renderer_include = manifest_dir.join("../../native-renderer/include");
    let renderer_header = renderer_include.join("imprint_renderer.h");
    if !header.is_file()
        || !cpp_header.is_file()
        || !source.is_file()
        || !preview_header.is_file()
        || !preview_source.is_file()
        || !renderer_header.is_file()
    {
        panic!("native-geo core or native renderer preview adapter is missing; expected native-geo sources and native-renderer/include/imprint_renderer.h");
    }
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", cpp_header.display());
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", preview_header.display());
    println!("cargo:rerun-if-changed={}", preview_source.display());
    println!("cargo:rerun-if-changed={}", renderer_header.display());

    let mut preview = cc::Build::new();
    preview
        .cpp(true)
        .std("c++17")
        .include(native_root.join("include"))
        .include(&renderer_include)
        .file(&preview_source);
    preview.compile("imprint_geo_native_preview");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dl");
    }

    let mut native = cc::Build::new();
    native
        .cpp(true)
        .std("c++17")
        .include(native_root.join("include"))
        .file(&source);

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let image_io_header = native_root.join("include/image_io.h");
        let image_io_source = native_root.join("src/image_io.mm");
        if !image_io_header.is_file() || !image_io_source.is_file() {
            panic!("macOS ImageIO source bridge is missing from native-geo");
        }
        println!("cargo:rerun-if-changed={}", image_io_header.display());
        println!("cargo:rerun-if-changed={}", image_io_source.display());
        native.file(image_io_source).flag("-fobjc-arc");
        println!("cargo:rustc-link-lib=framework=ImageIO");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=CoreImage");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    native.compile("imprint_geo");
    tauri_build::build()
}
