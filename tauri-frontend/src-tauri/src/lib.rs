use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{Manager, State};

#[derive(Default)]
pub struct AppState {
    pub api_port: Arc<Mutex<Option<u16>>>,
    pub python_child: Arc<Mutex<Option<Child>>>,
}

#[tauri::command]
fn get_api_port(state: State<AppState>) -> Result<u16, String> {
    let port_opt = state.api_port.lock().map_err(|e| e.to_string())?;
    port_opt.ok_or_else(|| "FastAPI sidecar is still starting up...".to_string())
}

/// 查找最适合的 Python 解释器或 sidecar 可执行文件
fn find_python_executable() -> Option<PathBuf> {
    // 1. 优先使用指定的 conda py311 虚拟环境
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();

    let conda_candidates = [
        PathBuf::from(&home).join(".conda/envs/py311/bin/python"),
        PathBuf::from(&home).join(".conda/envs/py311/python.exe"),
        PathBuf::from("/opt/homebrew/anaconda3/envs/py311/bin/python"),
        PathBuf::from("C:\\ProgramData\\anaconda3\\envs\\py311\\python.exe"),
        PathBuf::from("C:\\miniconda3\\envs\\py311\\python.exe"),
        PathBuf::from("C:\\anaconda3\\envs\\py311\\python.exe"),
    ];

    for p in &conda_candidates {
        if p.exists() {
            return Some(p.clone());
        }
    }

    // 2. 尝试系统 PATH 中的 python3 / python
    #[cfg(target_os = "windows")]
    let which_cmd = "where";
    #[cfg(not(target_os = "windows"))]
    let which_cmd = "which";

    for cmd in &["python3", "python"] {
        if let Ok(output) = Command::new(which_cmd).arg(cmd).output() {
            if output.status.success() {
                let stdout_str = String::from_utf8_lossy(&output.stdout);
                // Windows 下 where 可能会返回多行候选路径，取第一行
                if let Some(first_line) = stdout_str.lines().next() {
                    let path_str = first_line.trim().to_string();
                    if !path_str.is_empty() {
                        let p = PathBuf::from(path_str);
                        if p.exists() {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }

    None
}

/// 查找 src/app_api.py 脚本文件路径
fn find_app_api_script() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("src/app_api.py"),
        PathBuf::from("../src/app_api.py"),
        PathBuf::from("../../src/app_api.py"),
    ];

    for c in &candidates {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or_else(|_| c.clone()));
        }
    }
    None
}

/// 查找已打包的 sidecar 可执行文件或退回查找 .py 脚本
fn find_sidecar_or_script() -> (Option<PathBuf>, bool) {
    // 优先查找 PyInstaller 打包的 sidecar 可执行文件
    #[cfg(target_os = "windows")]
    let exe_name = "photo_sort_api.exe";
    #[cfg(not(target_os = "windows"))]
    let exe_name = "photo_sort_api";

    let sidecar_candidates = [
        PathBuf::from("dist-python/photo_sort_api").join(exe_name),
        PathBuf::from("../dist-python/photo_sort_api").join(exe_name),
        PathBuf::from("../../dist-python/photo_sort_api").join(exe_name),
    ];
    for c in &sidecar_candidates {
        if c.exists() {
            if let Ok(abs) = c.canonicalize() {
                return (Some(abs), true); // true = is sidecar, no python needed
            }
        }
    }

    // 退回到 .py 脚本（开发模式）
    (find_app_api_script(), false)
}

/// 启动 Python FastAPI 后端 sidecar 并捕获分配的端口
fn spawn_python_sidecar(
    api_port_arc: Arc<Mutex<Option<u16>>>,
    child_arc: Arc<Mutex<Option<Child>>>,
) {
    let (script_or_exe, is_sidecar) = find_sidecar_or_script();
    let path = match script_or_exe {
        Some(p) => p,
        None => {
            eprintln!("❌ 未找到 sidecar 或 app_api.py");
            return;
        }
    };

    let project_root = path
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    let mut cmd = if is_sidecar {
        println!(
            "🚀 正在启动已打包的 Python Sidecar 可执行文件:\n   Exe: {}\n   CWD: {}",
            path.display(),
            project_root.display()
        );
        Command::new(&path)
    } else {
        let python_exe = match find_python_executable() {
            Some(p) => p,
            None => {
                eprintln!("❌ 未找到 Python 解释器");
                return;
            }
        };
        println!(
            "🚀 正在使用 Python 启动 FastAPI Sidecar:\n   Python: {}\n   Script: {}\n   CWD: {}",
            python_exe.display(),
            path.display(),
            project_root.display()
        );
        let mut c = Command::new(&python_exe);
        c.arg(&path);
        c
    };

    cmd.current_dir(&project_root);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::inherit());

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    match cmd.spawn() {
        Ok(mut child) => {
            if let Some(stdout) = child.stdout.take() {
                let api_port_clone = Arc::clone(&api_port_arc);
                thread::spawn(move || {
                    let reader = BufReader::new(stdout);
                    for line_res in reader.lines() {
                        if let Ok(line) = line_res {
                            println!("[Python Sidecar] {}", line);
                            let trimmed = line.trim();
                            // 解析 {"port": 12345}
                            if trimmed.starts_with('{') && trimmed.contains("\"port\"") {
                                if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                                    if let Some(port_num) = val.get("port").and_then(|p| p.as_u64()) {
                                        let port = port_num as u16;
                                        println!("✅ 成功获取 FastAPI 监听端口: {}", port);
                                        if let Ok(mut port_lock) = api_port_clone.lock() {
                                            *port_lock = Some(port);
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }

            if let Ok(mut child_lock) = child_arc.lock() {
                *child_lock = Some(child);
            }
        }
        Err(e) => {
            eprintln!("❌ 启动 Python FastAPI sidecar 失败: {}", e);
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app_state = AppState::default();
    let api_port_arc = Arc::clone(&app_state.api_port);
    let python_child_arc = Arc::clone(&app_state.python_child);

    // 启动 sidecar 进程
    spawn_python_sidecar(Arc::clone(&api_port_arc), Arc::clone(&python_child_arc));

    let python_child_for_cleanup = Arc::clone(&python_child_arc);

    tauri::Builder::default()
        .manage(app_state)
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![get_api_port])
        .setup(|app| {
            // macOS 磨砂效果 / Windows Mica 效果
            if let Some(window) = app.get_webview_window("main") {
                #[cfg(target_os = "macos")]
                {
                    use window_vibrancy::{apply_vibrancy, NSVisualEffectMaterial};
                    let _ = apply_vibrancy(&window, NSVisualEffectMaterial::HudWindow, None, None);
                }

                #[cfg(target_os = "windows")]
                {
                    use window_vibrancy::apply_mica;
                    let _ = apply_mica(&window, Some(true));
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |_app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit = event {
                println!("🛑 正在关闭应用并清理 Python sidecar 进程...");
                if let Ok(mut child_lock) = python_child_for_cleanup.lock() {
                    if let Some(mut child) = child_lock.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                        println!("🧹 Python sidecar 进程已成功终止。");
                    }
                }
            }
        });
}
