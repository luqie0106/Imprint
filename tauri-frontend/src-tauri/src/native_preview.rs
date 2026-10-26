//! Adapter to the existing renderer's stable full-resolution C ABI.
//! This stage accepts only byte-identical neutral results for finished photos.
use image::{ImageBuffer, RgbImage};
use std::collections::HashSet;
use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};

extern "C" {
    fn imprint_geo_native_preview(
        library_path: *const c_char,
        rgb: *const u8,
        width: u32,
        height: u32,
        out_rgb: *mut u8,
        backend: *mut c_char,
        backend_cap: u32,
        error: *mut c_char,
        errcap: u32,
    ) -> i32;
}

fn candidates() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("IMPRINT_NATIVE_RENDERER_LIB") {
        return vec![PathBuf::from(path)];
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut roots = vec![root.clone()];
    if let Some(path) = std::env::var_os("IMPRINT_PROJECT_ROOT") {
        roots.push(path.into());
    }
    // A managed worktree deliberately omits ignored build artifacts. Locate the
    // common checkout's pre-existing native library without copying or rebuilding it.
    if let Ok(pointer) = std::fs::read_to_string(root.join(".git")) {
        if let Some(value) = pointer.trim().strip_prefix("gitdir:") {
            let value = PathBuf::from(value.trim());
            let git_dir = if value.is_absolute() {
                value
            } else {
                root.join(value)
            };
            if let Ok(common) = std::fs::read_to_string(git_dir.join("commondir")) {
                if let Ok(common) = git_dir.join(common.trim()).canonicalize() {
                    if let Some(checkout) = common.parent() {
                        roots.push(checkout.into());
                    }
                }
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            roots.push(parent.into());
            roots.push(parent.join("../Resources"));
        }
    }
    #[cfg(target_os = "macos")]
    let names = ["libimprint_renderer.dylib", "imprint_renderer.dylib"];
    #[cfg(target_os = "windows")]
    let names = ["imprint_renderer.dll", "libimprint_renderer.dll"];
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let names = ["libimprint_renderer.so", "imprint_renderer.so"];
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for root in roots {
        for suffix in [
            "",
            "native-renderer/build",
            "native-renderer/build/Release",
            "dist-python/imprint_api/_internal",
            "_internal",
        ] {
            for name in names {
                let path = root.join(suffix).join(name);
                if path.is_file() {
                    let canonical = path.canonicalize().unwrap_or(path);
                    if seen.insert(canonical.clone()) {
                        result.push(canonical);
                    }
                }
            }
        }
    }
    result
}

fn run(rgb: &RgbImage, library: &Path) -> Result<(RgbImage, String), String> {
    let library = CString::new(library.to_string_lossy().as_bytes())
        .map_err(|_| "Invalid renderer library path")?;
    let mut output = vec![0; rgb.as_raw().len()];
    let mut backend = [0 as c_char; 128];
    let mut error = [0 as c_char; 1024];
    // Adapter ABI uses 1=success; it does not write output on failure.
    let status = unsafe {
        imprint_geo_native_preview(
            library.as_ptr(),
            rgb.as_raw().as_ptr(),
            rgb.width(),
            rgb.height(),
            output.as_mut_ptr(),
            backend.as_mut_ptr(),
            backend.len() as u32,
            error.as_mut_ptr(),
            error.len() as u32,
        )
    };
    if status != 1 {
        let error = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(error.into_owned());
    }
    let backend = unsafe { CStr::from_ptr(backend.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let image = ImageBuffer::from_raw(rgb.width(), rgb.height(), output)
        .ok_or("Invalid renderer output dimensions")?;
    Ok((image, backend))
}

pub fn prepare(rgb: RgbImage) -> (RgbImage, String) {
    let mut last_error = None;
    for candidate in candidates() {
        match run(&rgb, &candidate) {
            Ok((image, backend)) => return (image, format!("底图 C++/{backend} · 地图 C++ CPU")),
            Err(error) => last_error = Some(error),
        }
    }
    let reason = last_error
        .as_deref()
        .unwrap_or("未找到既有 native renderer 库");
    (
        rgb,
        format!("C++ CPU 原尺寸路径（旧 GPU 接口不可用：{reason}）"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_renderer_leaves_finished_source_unchanged() {
        let rgb = RgbImage::from_raw(2, 1, vec![17, 93, 201, 255, 0, 128]).unwrap();
        let before = rgb.clone();
        assert!(run(&rgb, Path::new("/imprint-test-missing/native-renderer")).is_err());
        assert_eq!(rgb, before);
    }
    #[test]
    #[ignore = "requires IMPRINT_GEO_NATIVE_LIB and native GPU access"]
    fn existing_renderer_full_resolution_is_byte_identical() {
        let path = std::env::var("IMPRINT_GEO_NATIVE_LIB").expect("set IMPRINT_GEO_NATIVE_LIB");
        let rgb = RgbImage::from_fn(1801, 905, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let (rendered, backend) = run(&rgb, Path::new(&path)).unwrap();
        assert_eq!(rendered.dimensions(), (1801, 905));
        assert_eq!(rendered, rgb);
        assert!(!backend.is_empty());
        eprintln!("existing renderer backend={backend}, byte identity verified at 1801x905");
    }
}
