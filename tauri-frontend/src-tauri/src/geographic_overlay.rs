#[path = "geo_osm.rs"]
mod geo_osm;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use exif::{Reader as ExifReader, Tag, Value};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::imageops;
use image::{
    ExtendedColorType, ImageBuffer, ImageEncoder, ImageFormat, ImageReader, RgbImage, RgbaImage,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::ffi::{c_char, CStr, CString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::State;
use uuid::Uuid;

const MAX_PIXELS: u64 = 100_000_000;
const MAX_SOURCE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_SESSION_RGB_BYTES: usize = 256 * 1024 * 1024;
const MAX_OPEN_SESSIONS: usize = 12;
const PROJECT_FORMAT: &str = "imprint.geographic-overlay";
const PROJECT_VERSION: u32 = 1;
const DEFAULT_OVERPASS_ENDPOINT: &str = "https://overpass-api.de/api/interpreter";

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NativeGeoConfig {
    latitude: f64,
    longitude: f64,
    opacity: f64,
    radius_km: f64,
    offset_x: f64,
    offset_y: f64,
    scale: f64,
    rotation: f64,
    line_width: f64,
    feather: f64,
    coverage: f64,
    manual_horizon: f64,
    color_rgb: u32,
    enabled: i32,
    has_location: i32,
    style: i32,
    auto_sky_fade: i32,
    labels: i32,
    north_arrow: i32,
    scale_bar: i32,
    photo_marker: i32,
    adaptive: i32,
    provider: i32,
}

#[repr(C)]
struct GeoVertex {
    latitude: f64,
    longitude: f64,
}

#[repr(C)]
struct GeoFeatureView {
    kind: u32,
    point_count: u32,
    points: *const GeoVertex,
    label: *const c_char,
}

extern "C" {
    fn imprint_geo_render(
        rgb: *const u8,
        width: u32,
        height: u32,
        config: *const NativeGeoConfig,
        overlay_rgba: *mut u8,
        error: *mut c_char,
        error_capacity: u32,
    ) -> i32;
    fn imprint_geo_render_features(
        rgb: *const u8,
        width: u32,
        height: u32,
        config: *const NativeGeoConfig,
        features: *const GeoFeatureView,
        feature_count: u32,
        overlay_rgba: *mut u8,
        error: *mut c_char,
        error_capacity: u32,
    ) -> i32;
    fn imprint_geo_composite(
        rgb: *const u8,
        overlay_rgba: *const u8,
        width: u32,
        height: u32,
        out_rgb: *mut u8,
    ) -> i32;
}

#[cfg(target_os = "macos")]
extern "C" {
    fn imprint_imageio_dimensions(
        path: *const c_char,
        width: *mut u32,
        height: *mut u32,
        error: *mut c_char,
        error_capacity: u32,
    ) -> i32;
    fn imprint_imageio_decode_rgba8_srgb(
        path: *const c_char,
        rgba: *mut u8,
        width: u32,
        height: u32,
        capacity: u64,
        error: *mut c_char,
        error_capacity: u32,
    ) -> i32;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GeoConfig {
    pub latitude: f64,
    pub longitude: f64,
    pub opacity: f64,
    #[serde(alias = "radiusKm")]
    pub radius_km: f64,
    #[serde(alias = "offsetX")]
    pub offset_x: f64,
    #[serde(alias = "offsetY")]
    pub offset_y: f64,
    pub scale: f64,
    pub rotation: f64,
    #[serde(alias = "lineWidth")]
    pub line_width: f64,
    pub feather: f64,
    pub coverage: f64,
    #[serde(alias = "manualHorizon")]
    pub manual_horizon: f64,
    #[serde(alias = "colorRgb")]
    pub color_rgb: u32,
    pub enabled: bool,
    #[serde(alias = "hasLocation")]
    pub has_location: bool,
    pub style: i32,
    #[serde(alias = "autoSkyFade")]
    pub auto_sky_fade: bool,
    pub labels: bool,
    #[serde(alias = "northArrow")]
    pub north_arrow: bool,
    #[serde(alias = "scaleBar")]
    pub scale_bar: bool,
    #[serde(alias = "photoMarker")]
    pub photo_marker: bool,
    pub adaptive: bool,
    pub provider: i32,
    #[serde(alias = "overpassEndpoint")]
    pub overpass_endpoint: String,
}

impl Default for GeoConfig {
    fn default() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
            opacity: 0.18,
            radius_km: 3.0,
            offset_x: 0.0,
            offset_y: 0.0,
            scale: 1.0,
            rotation: 0.0,
            line_width: 1.0,
            feather: 0.04,
            coverage: 0.8,
            manual_horizon: 0.0,
            color_rgb: 0,
            enabled: false,
            has_location: false,
            style: 0,
            auto_sky_fade: true,
            labels: true,
            north_arrow: true,
            scale_bar: true,
            photo_marker: true,
            adaptive: true,
            provider: 0,
            overpass_endpoint: DEFAULT_OVERPASS_ENDPOINT.into(),
        }
    }
}

impl GeoConfig {
    fn validate(&self) -> Result<(), String> {
        let finite = [
            self.latitude,
            self.longitude,
            self.opacity,
            self.radius_km,
            self.offset_x,
            self.offset_y,
            self.scale,
            self.rotation,
            self.line_width,
            self.feather,
            self.coverage,
            self.manual_horizon,
        ];
        if finite.iter().any(|value| !value.is_finite()) {
            return Err("地理叠加参数不能包含 NaN 或无穷值。".into());
        }
        if !(-90.0..=90.0).contains(&self.latitude) || !(-180.0..=180.0).contains(&self.longitude) {
            return Err("GPS 纬度或经度超出有效范围。".into());
        }
        if !(0.0..=1.0).contains(&self.opacity) {
            return Err("透明度必须在 0 到 1 之间。".into());
        }
        if !(0.05..=1000.0).contains(&self.radius_km) {
            return Err("地理范围必须在 0.05 到 1000 公里之间。".into());
        }
        if !(-1.0..=1.0).contains(&self.offset_x) || !(-1.0..=1.0).contains(&self.offset_y) {
            return Err("叠加偏移必须在 -1 到 1 之间。".into());
        }
        if !(0.1..=4.0).contains(&self.scale) {
            return Err("叠加缩放必须在 0.1 到 4 之间。".into());
        }
        if !(-36000.0..=36000.0).contains(&self.rotation) {
            return Err("叠加旋转角度超出有效范围。".into());
        }
        if !(0.25..=40.0).contains(&self.line_width) {
            return Err("线宽必须在 0.25 到 40 之间。".into());
        }
        if !(0.0..=0.3).contains(&self.feather) {
            return Err("羽化必须在 0 到 0.3 之间。".into());
        }
        if !(0.05..=1.0).contains(&self.coverage) {
            return Err("覆盖范围必须在 0.05 到 1 之间。".into());
        }
        if !(0.0..=1.0).contains(&self.manual_horizon) {
            return Err("地平线位置必须在 0 到 1 之间。".into());
        }
        if self.color_rgb > 0x00ff_ffff {
            return Err("叠加颜色必须是 24 位 RGB 值。".into());
        }
        if !(0..=3).contains(&self.style) {
            return Err("未知的地理叠加样式。".into());
        }
        if !(0..=1).contains(&self.provider) {
            return Err("未知的地理数据源。".into());
        }
        if self.overpass_endpoint.len() > 2048
            || self.overpass_endpoint.chars().any(char::is_control)
        {
            return Err("Overpass 地址过长或包含控制字符。".into());
        }
        if self.provider == 1 {
            if self.radius_km > 20.0 {
                return Err("公共 Overpass 查询范围最大为 20 公里；请手动缩小查询范围。".into());
            }
            geo_osm::validate_endpoint(&self.overpass_endpoint)?;
            if self.style == 3 {
                return Err("OSM 数据源不包含地形模型，不能使用地形样式。".into());
            }
        }
        if self.provider == 1 {
            if self.radius_km > 20.0 {
                return Err("公共 Overpass 查询范围最大为 20 公里；请手动缩小查询范围。".into());
            }
            geo_osm::validate_endpoint(&self.overpass_endpoint)?;
            if self.style == 3 {
                return Err("OSM 数据源不包含地形模型，不能使用地形样式。".into());
            }
        }
        if self.enabled && !self.has_location {
            return Err("请先提供照片 GPS 位置或手动设置位置。".into());
        }
        Ok(())
    }

    fn to_native(&self) -> NativeGeoConfig {
        NativeGeoConfig {
            latitude: self.latitude,
            longitude: self.longitude,
            opacity: self.opacity,
            radius_km: self.radius_km,
            offset_x: self.offset_x,
            offset_y: self.offset_y,
            scale: self.scale,
            rotation: self.rotation,
            line_width: self.line_width,
            feather: self.feather,
            coverage: self.coverage,
            manual_horizon: self.manual_horizon,
            color_rgb: self.color_rgb,
            enabled: i32::from(self.enabled),
            has_location: i32::from(self.has_location),
            style: self.style,
            auto_sky_fade: i32::from(self.auto_sky_fade),
            labels: i32::from(self.labels),
            north_arrow: i32::from(self.north_arrow),
            scale_bar: i32::from(self.scale_bar),
            photo_marker: i32::from(self.photo_marker),
            adaptive: i32::from(self.adaptive),
            provider: self.provider,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeoPhoto {
    pub id: String,
    pub source_name: String,
    pub width: u32,
    pub height: u32,
    pub preview: String,
    pub exif_location: Option<GeoLocation>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeoPreview {
    pub image: String,
    pub overlay: String,
    pub width: u32,
    pub height: u32,
    pub warning: String,
    pub attribution: String,
    pub feature_count: u32,
    pub cache_hit: bool,
    pub provider: String,
    pub render_backend: String,
}

#[derive(Clone, Default)]
pub struct GeoState {
    inner: Arc<GeoStateInner>,
}

struct GeoSession {
    source_path: PathBuf,
    source_name: String,
    width: u32,
    height: u32,
    preview_rgb: RgbImage,
    native_backend: String,
    render_mutex: Mutex<()>,
    exif_location: Option<GeoLocation>,
    project_path: Mutex<Option<PathBuf>>,
    preview_generation: AtomicU64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectFile {
    format: String,
    version: u32,
    provider: i32,
    mock: bool,
    source: ProjectSource,
    #[serde(default)]
    config: GeoConfig,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectSource {
    path: PathBuf,
    source_name: String,
    width: u32,
    height: u32,
    gps_source: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputKind {
    Jpeg,
    Png,
    Tiff,
    Dng,
}

impl InputKind {
    fn from_path(path: &Path) -> Result<Self, String> {
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .ok_or_else(|| "只支持 JPEG、PNG、TIFF 和 DNG 图像。".to_string())?
            .to_ascii_lowercase();
        match extension.as_str() {
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "png" => Ok(Self::Png),
            "tif" | "tiff" => Ok(Self::Tiff),
            "dng" => Ok(Self::Dng),
            _ => Err("只支持 JPEG、PNG、TIFF 和 DNG 图像。".into()),
        }
    }

    fn image_format(self) -> ImageFormat {
        match self {
            Self::Jpeg => ImageFormat::Jpeg,
            Self::Png => ImageFormat::Png,
            Self::Tiff | Self::Dng => ImageFormat::Tiff,
        }
    }
}

struct TiffSummary {
    width: Option<u32>,
    height: Option<u32>,
    active_width: Option<u32>,
    active_height: Option<u32>,
    default_crop_width: Option<u32>,
    default_crop_height: Option<u32>,
    has_dng_version: bool,
    has_icc_profile: bool,
}

struct DecodedSource {
    source_path: PathBuf,
    source_name: String,
    rgb: RgbImage,
    exif_location: Option<GeoLocation>,
}

#[tauri::command]
pub async fn geo_open(path: String, state: State<'_, GeoState>) -> Result<GeoPhoto, String> {
    let state = state.inner().clone_state();
    tauri::async_runtime::spawn_blocking(move || open_sync(&state, Path::new(&path)))
        .await
        .map_err(|error| format!("打开照片任务失败：{error}"))?
}

#[tauri::command]
pub async fn geo_preview(
    id: String,
    config: GeoConfig,
    state: State<'_, GeoState>,
) -> Result<GeoPreview, String> {
    let state = state.inner().clone_state();
    tauri::async_runtime::spawn_blocking(move || preview_sync(&state, &id, &config))
        .await
        .map_err(|error| format!("生成预览任务失败：{error}"))?
}

#[tauri::command]
pub async fn geo_export(
    id: String,
    config: GeoConfig,
    path: String,
    format: String,
    overlay_only: bool,
    state: State<'_, GeoState>,
) -> Result<String, String> {
    let state = state.inner().clone_state();
    tauri::async_runtime::spawn_blocking(move || {
        export_sync(
            &state,
            &id,
            &config,
            Path::new(&path),
            &format,
            overlay_only,
        )
    })
    .await
    .map_err(|error| format!("導出任务失败：{error}"))?
}

#[tauri::command]
pub async fn geo_save_project(
    id: String,
    config: GeoConfig,
    path: String,
    state: State<'_, GeoState>,
) -> Result<String, String> {
    let state = state.inner().clone_state();
    tauri::async_runtime::spawn_blocking(move || {
        save_project_sync(&state, &id, &config, Path::new(&path))
    })
    .await
    .map_err(|error| format!("保存项目任务失败：{error}"))?
}

#[tauri::command]
pub async fn geo_load_project(
    path: String,
    state: State<'_, GeoState>,
) -> Result<GeoLoadedProject, String> {
    let state = state.inner().clone_state();
    tauri::async_runtime::spawn_blocking(move || load_project_sync(&state, Path::new(&path)))
        .await
        .map_err(|error| format!("加载项目任务失败：{error}"))?
}

#[derive(Serialize)]
pub struct GeoLoadedProject {
    pub photo: GeoPhoto,
    pub config: GeoConfig,
}

impl GeoState {
    fn clone_state(&self) -> Arc<GeoStateInner> {
        Arc::clone(&self.inner)
    }
}

#[derive(Default)]
struct GeoStateInner {
    sessions: Mutex<HashMap<String, Arc<GeoSession>>>,
    order: Mutex<VecDeque<String>>,
    osm: geo_osm::OsmClient,
}

fn open_sync(state: &Arc<GeoStateInner>, path: &Path) -> Result<GeoPhoto, String> {
    let decoded = decode_source(path)?;
    let full_width = decoded.rgb.width();
    let full_height = decoded.rgb.height();
    let (preview_rgb, native_backend) = crate::native_preview::prepare(decoded.rgb);
    let preview = png_data_uri_rgb(&preview_rgb)?;
    let id = Uuid::new_v4().to_string();
    let photo = GeoPhoto {
        id: id.clone(),
        source_name: decoded.source_name.clone(),
        width: full_width,
        height: full_height,
        preview,
        exif_location: decoded.exif_location.clone(),
    };
    let session = Arc::new(GeoSession {
        source_path: decoded.source_path,
        source_name: decoded.source_name,
        width: full_width,
        height: full_height,
        preview_rgb,
        native_backend,
        render_mutex: Mutex::new(()),
        exif_location: decoded.exif_location,
        project_path: Mutex::new(None),
        preview_generation: AtomicU64::new(0),
    });
    insert_session(state, id, session)?;
    Ok(photo)
}

fn insert_session(
    state: &Arc<GeoStateInner>,
    id: String,
    session: Arc<GeoSession>,
) -> Result<(), String> {
    let mut sessions = state
        .sessions
        .lock()
        .map_err(|_| "照片会话锁不可用。".to_string())?;
    let mut order = state
        .order
        .lock()
        .map_err(|_| "照片会话顺序锁不可用。".to_string())?;
    sessions.insert(id.clone(), session);
    order.retain(|existing| existing != &id);
    order.push_back(id);
    while order.len() > MAX_OPEN_SESSIONS
        || (order.len() > 1
            && sessions
                .values()
                .map(|item| item.preview_rgb.as_raw().len())
                .sum::<usize>()
                > MAX_SESSION_RGB_BYTES)
    {
        if let Some(expired) = order.pop_front() {
            sessions.remove(&expired);
        }
    }
    Ok(())
}

fn get_session(state: &GeoStateInner, id: &str) -> Result<Arc<GeoSession>, String> {
    let sessions = state
        .sessions
        .lock()
        .map_err(|_| "照片会话锁不可用。".to_string())?;
    sessions
        .get(id)
        .cloned()
        .ok_or_else(|| "照片会话不存在或已过期，请重新打开照片。".into())
}

fn preview_sync(state: &GeoStateInner, id: &str, config: &GeoConfig) -> Result<GeoPreview, String> {
    config.validate()?;
    let session = get_session(state, id)?;
    let generation = session
        .preview_generation
        .fetch_add(1, Ordering::AcqRel)
        .wrapping_add(1);
    let _render_guard = session
        .render_mutex
        .lock()
        .map_err(|_| "预览渲染锁不可用。".to_string())?;
    if session.preview_generation.load(Ordering::Acquire) != generation {
        return Err("预览请求已由新参数替代。".into());
    }
    let fetched = if config.provider == 1 && config.enabled {
        Some(state.osm.fetch(
            &config.overpass_endpoint,
            config.latitude,
            config.longitude,
            config.radius_km,
        )?)
    } else {
        None
    };
    let features = fetched
        .as_ref()
        .map(|data| data.features.as_slice())
        .unwrap_or(&[]);
    if session.preview_generation.load(Ordering::Acquire) != generation {
        return Err("OSM 预览请求已取消，因为更新的预览已启动。".into());
    }
    let (overlay, composite) =
        render_overlay_and_composite(&session.preview_rgb, config, features)?;
    if session.preview_generation.load(Ordering::Acquire) != generation {
        return Err("OSM 预览请求已取消，因为更新的预览已启动。".into());
    }
    let overlay_is_empty = overlay.pixels().all(|pixel| pixel[3] == 0);
    let warning = if config.provider == 1 && !config.enabled {
        "OSM 数据源已选择；启用叠加后将查询当前坐标和半径。".to_string()
    } else if config.provider == 1 && overlay_is_empty {
        "OSM 要素已载入，但当前保护遮罩内没有可见线条。可调整位置；如需手动预览，请关闭自动天空渐隐并设置手动保护边界。".to_string()
    } else if config.style == 3 {
        "地形样式需要已验证的 DEM 数据源，目前不可用。".to_string()
    } else if !config.has_location && session.exif_location.is_none() {
        "照片没有 GPS 位置；请手动设置位置后再生成地理叠加。".to_string()
    } else if config.enabled && overlay_is_empty {
        "未检测到可靠天空，当前叠加层为空。可关闭自动天空渐隐并设置手动保护边界后重试。".to_string()
    } else if config.enabled {
        "天空保护采用图像启发式判断；请检查地图线条是否避开主体边缘。".to_string()
    } else {
        String::new()
    };
    Ok(GeoPreview {
        image: png_data_uri_rgb(&composite)?,
        overlay: png_data_uri_rgba(&overlay)?,
        width: session.preview_rgb.width(),
        height: session.preview_rgb.height(),
        warning,
        attribution: fetched
            .as_ref()
            .map(|data| format!("© OpenStreetMap contributors · {}", data.endpoint_host))
            .unwrap_or_else(|| {
                if config.provider == 1 {
                    "© OpenStreetMap contributors".to_string()
                } else {
                    "Mock 数据源；不包含真实地图数据".to_string()
                }
            }),
        feature_count: fetched
            .as_ref()
            .map(|data| data.features.len() as u32)
            .unwrap_or(0),
        cache_hit: fetched.as_ref().is_some_and(|data| data.cache_hit),
        provider: if config.provider == 1 { "osm" } else { "mock" }.into(),
        render_backend: session.native_backend.clone(),
    })
}

fn export_sync(
    state: &GeoStateInner,
    id: &str,
    config: &GeoConfig,
    output_path: &Path,
    format: &str,
    overlay_only: bool,
) -> Result<String, String> {
    config.validate()?;
    let session = get_session(state, id)?;
    let kind = match format.to_ascii_lowercase().as_str() {
        "png" => OutputFormat::Png,
        "jpeg" | "jpg" => OutputFormat::Jpeg,
        _ => return Err("输出格式只支持 PNG 或 JPEG。".into()),
    };
    if overlay_only && kind == OutputFormat::Jpeg {
        return Err("JPEG 不支持透明图层；透明叠加层请导出为 PNG。".into());
    }
    validate_output_extension(output_path, kind)?;
    let target = normalize_target_path(output_path)?;
    if target == session.source_path {
        return Err("输出路径不能覆盖原始照片。".into());
    }

    // Serialize full-size work with preview and discard superseded preview results.
    session.preview_generation.fetch_add(1, Ordering::AcqRel);
    let _render_guard = session
        .render_mutex
        .lock()
        .map_err(|_| "导出渲染锁不可用。".to_string())?;
    // Export intentionally reopens and decodes the original full-size source.
    let decoded = decode_source(&session.source_path)?;
    if decoded.rgb.width() != session.width || decoded.rgb.height() != session.height {
        return Err("原始照片尺寸与打开时不一致，已停止导出。".into());
    }
    let fetched = if config.provider == 1 && config.enabled {
        Some(state.osm.fetch(
            &config.overpass_endpoint,
            config.latitude,
            config.longitude,
            config.radius_km,
        )?)
    } else {
        None
    };
    let features = fetched
        .as_ref()
        .map(|data| data.features.as_slice())
        .unwrap_or(&[]);
    let (source_rgb, _) = crate::native_preview::prepare(decoded.rgb);
    let (overlay, composite) = render_overlay_and_composite(&source_rgb, config, features)?;
    let rgb = if overlay_only { None } else { Some(composite) };
    let rgba = if overlay_only { Some(overlay) } else { None };
    write_image_no_clobber(&target, kind, rgb.as_ref(), rgba.as_ref())?;
    Ok(target.to_string_lossy().into_owned())
}

fn render_overlay_and_composite(
    rgb: &RgbImage,
    config: &GeoConfig,
    features: &[geo_osm::GeoFeature],
) -> Result<(RgbaImage, RgbImage), String> {
    let width = rgb.width();
    let height = rgb.height();
    let pixel_count = checked_pixel_count(width, height)?;
    let mut overlay_bytes = vec![0u8; pixel_count * 4];
    let native_config = config.to_native();
    let mut error = [0 as c_char; 1024];
    let status = if config.provider == 1 && !config.enabled {
        overlay_bytes.fill(0);
        0
    } else if config.provider == 1 {
        render_osm_features(
            rgb,
            &native_config,
            features,
            &mut overlay_bytes,
            &mut error,
        )?
    } else {
        unsafe {
            imprint_geo_render(
                rgb.as_raw().as_ptr(),
                width,
                height,
                &native_config,
                overlay_bytes.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len() as u32,
            )
        }
    };
    if status != 0 {
        return Err(native_error(&error, "地理叠加渲染失败。"));
    }
    let overlay = ImageBuffer::from_raw(width, height, overlay_bytes)
        .ok_or_else(|| "地理叠加层尺寸无效。".to_string())?;
    let mut composite_bytes = vec![0u8; pixel_count * 3];
    let status = unsafe {
        imprint_geo_composite(
            rgb.as_raw().as_ptr(),
            overlay.as_raw().as_ptr(),
            width,
            height,
            composite_bytes.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err("合成地理叠加层失败。".into());
    }
    let composite = ImageBuffer::from_raw(width, height, composite_bytes)
        .ok_or_else(|| "合成结果尺寸无效。".to_string())?;
    Ok((overlay, composite))
}

fn render_osm_features(
    rgb: &RgbImage,
    native_config: &NativeGeoConfig,
    features: &[geo_osm::GeoFeature],
    overlay_bytes: &mut [u8],
    error: &mut [c_char],
) -> Result<i32, String> {
    let total_points = features
        .iter()
        .try_fold(0usize, |total, feature| {
            total.checked_add(feature.points.len())
        })
        .ok_or_else(|| "OSM 几何点数超出范围。".to_string())?;
    if features.len() > 10_000 || total_points > 500_000 {
        return Err("OSM 几何超过原生渲染上限。".into());
    }
    let mut points = Vec::with_capacity(total_points);
    let mut labels = Vec::with_capacity(features.len());
    for feature in features {
        if feature.points.is_empty()
            || feature.points.iter().any(|point| {
                !point.latitude.is_finite()
                    || !point.longitude.is_finite()
                    || !(-90.0..=90.0).contains(&point.latitude)
                    || !(-180.0..=180.0).contains(&point.longitude)
            })
        {
            return Err("OSM 几何包含空要素或无效经纬度。".into());
        }
        if feature.label.len() > 256 || !feature.label.is_ascii() || feature.label.contains('\0') {
            return Err("OSM 地名标签超过长度限制或包含无效字符。".into());
        }
        if feature.kind > 3 {
            return Err("OSM 要素类型无效。".into());
        }
        points.extend(feature.points.iter().map(|point| GeoVertex {
            latitude: point.latitude,
            longitude: point.longitude,
        }));
        labels.push(
            CString::new(feature.label.as_bytes())
                .map_err(|_| "OSM 地名标签包含无效字符。".to_string())?,
        );
    }
    let mut point_offset = 0usize;
    let feature_views = features
        .iter()
        .enumerate()
        .map(|(index, feature)| {
            let count = feature.points.len();
            let view = GeoFeatureView {
                kind: feature.kind,
                point_count: count as u32,
                points: points[point_offset..point_offset + count].as_ptr(),
                label: labels[index].as_ptr(),
            };
            point_offset += count;
            view
        })
        .collect::<Vec<_>>();
    let feature_ptr = if feature_views.is_empty() {
        std::ptr::null()
    } else {
        feature_views.as_ptr()
    };
    Ok(unsafe {
        imprint_geo_render_features(
            rgb.as_raw().as_ptr(),
            rgb.width(),
            rgb.height(),
            native_config,
            feature_ptr,
            feature_views.len() as u32,
            overlay_bytes.as_mut_ptr(),
            error.as_mut_ptr(),
            error.len() as u32,
        )
    })
}

fn decode_source(path: &Path) -> Result<DecodedSource, String> {
    let kind = InputKind::from_path(path)?;
    let canonical_path = path
        .canonicalize()
        .map_err(|_| "照片路径不存在或无法访问。".to_string())?;
    let metadata =
        fs::metadata(&canonical_path).map_err(|_| "无法读取照片文件信息。".to_string())?;
    if !metadata.is_file() {
        return Err("所选路径不是普通照片文件。".into());
    }
    if metadata.len() == 0 || metadata.len() > MAX_SOURCE_BYTES {
        return Err("照片文件为空或超过 8 GiB 解码上限。".into());
    }
    validate_actual_format(&canonical_path, kind)?;
    let exif = read_exif_metadata(&canonical_path);
    let tiff_summary = if matches!(kind, InputKind::Tiff | InputKind::Dng) {
        Some(
            read_tiff_summary(&canonical_path)
                .map_err(|_| "TIFF/DNG 目录结构无效。".to_string())?,
        )
    } else {
        None
    };
    if kind == InputKind::Dng
        && !tiff_summary
            .as_ref()
            .is_some_and(|summary| summary.has_dng_version)
    {
        return Err(".dng 文件缺少 DNGVersion 标记，不能按 DNG 解码。".into());
    }
    #[cfg(not(target_os = "macos"))]
    if kind == InputKind::Dng {
        return Err(
            "当前平台没有经过验证的全尺寸、色彩管理 DNG 解码器；请在 macOS 上打开，或先转换为 sRGB TIFF/JPEG。"
                .into(),
        );
    }
    let (width, height) = source_dimensions(&canonical_path, kind, tiff_summary.as_ref())?;
    checked_pixel_count(width, height)?;
    #[cfg(not(target_os = "macos"))]
    {
        if has_embedded_icc(&canonical_path, kind, tiff_summary.as_ref())? {
            return Err(
                "此平台不能将嵌入的 ICC 色彩配置转换为 sRGB；请先将照片转换为 sRGB。".into(),
            );
        }
        if exif.color_space_uncalibrated {
            return Err(
                "照片声明了未校准的色彩空间；当前平台只接受 sRGB 或无配置文件的输入。".into(),
            );
        }
    }
    let mut rgba = decode_full_rgba(&canonical_path, kind, width, height)?;
    if rgba.width() != width || rgba.height() != height {
        return Err("解码器返回的尺寸与全尺寸源图不一致。".into());
    }
    if kind == InputKind::Dng
        && !rgba
            .pixels()
            .any(|pixel| pixel[0] > 4 || pixel[1] > 4 || pixel[2] > 4)
    {
        return Err("DNG 解码结果全黑；为避免误导，未生成预览或导出。".into());
    }
    let orientation = exif.orientation;
    rgba = apply_orientation(rgba, orientation);
    let rgb = flatten_rgba_over_white(&rgba);
    let source_name = canonical_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("photo")
        .to_string();
    Ok(DecodedSource {
        source_path: canonical_path,
        source_name,
        rgb,
        exif_location: exif.location,
    })
}

#[derive(Default)]
struct ExifMetadata {
    orientation: u16,
    location: Option<GeoLocation>,
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    color_space_uncalibrated: bool,
}

fn read_exif_metadata(path: &Path) -> ExifMetadata {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(_) => {
            return ExifMetadata {
                orientation: 1,
                ..Default::default()
            }
        }
    };
    let exif = match ExifReader::new().read_from_container(&mut BufReader::new(file)) {
        Ok(exif) => exif,
        Err(_) => {
            return ExifMetadata {
                orientation: 1,
                ..Default::default()
            }
        }
    };
    let find = |tag: Tag| exif.fields().find(|field| field.tag == tag);
    let orientation = find(Tag::Orientation)
        .and_then(|field| first_uint(&field.value))
        .map(|value| value as u16)
        .filter(|value| (1..=8).contains(value))
        .unwrap_or(1);
    let latitude = find(Tag::GPSLatitude).and_then(|field| rational_triplet(&field.value));
    let longitude = find(Tag::GPSLongitude).and_then(|field| rational_triplet(&field.value));
    let latitude_ref = find(Tag::GPSLatitudeRef).and_then(|field| first_ascii(&field.value));
    let longitude_ref = find(Tag::GPSLongitudeRef).and_then(|field| first_ascii(&field.value));
    let location = match (latitude, longitude, latitude_ref, longitude_ref) {
        (Some(lat), Some(lon), Some(lat_ref), Some(lon_ref)) => {
            match (
                gps_decimal_degrees(lat, &lat_ref),
                gps_decimal_degrees(lon, &lon_ref),
            ) {
                (Some(latitude), Some(longitude))
                    if (-90.0..=90.0).contains(&latitude)
                        && (-180.0..=180.0).contains(&longitude) =>
                {
                    Some(GeoLocation {
                        latitude,
                        longitude,
                    })
                }
                _ => None,
            }
        }
        _ => None,
    };
    let color_space_uncalibrated = exif.fields().any(|field| {
        field.tag.to_string() == "ColorSpace"
            && first_uint(&field.value).is_some_and(|value| value == 0xffff)
    });
    ExifMetadata {
        orientation,
        location,
        color_space_uncalibrated,
    }
}

fn first_uint(value: &Value) -> Option<u32> {
    match value {
        Value::Short(values) => values.first().map(|value| u32::from(*value)),
        Value::Long(values) => values.first().copied(),
        Value::Byte(values) => values.first().map(|value| u32::from(*value)),
        _ => None,
    }
}

fn first_ascii(value: &Value) -> Option<String> {
    match value {
        Value::Ascii(values) => values.first().map(|bytes| {
            String::from_utf8_lossy(bytes)
                .trim_matches('\0')
                .trim()
                .to_string()
        }),
        _ => None,
    }
}

fn rational_triplet(value: &Value) -> Option<[f64; 3]> {
    match value {
        Value::Rational(values) if values.len() >= 3 => {
            let numbers: Vec<f64> = values
                .iter()
                .take(3)
                .map(|r| {
                    if r.denom == 0 {
                        f64::NAN
                    } else {
                        r.num as f64 / r.denom as f64
                    }
                })
                .collect();
            if numbers.iter().all(|number| number.is_finite()) {
                Some([numbers[0], numbers[1], numbers[2]])
            } else {
                None
            }
        }
        _ => None,
    }
}

fn gps_decimal_degrees(parts: [f64; 3], reference: &str) -> Option<f64> {
    if parts.iter().any(|part| !part.is_finite() || *part < 0.0) {
        return None;
    }
    let mut degrees = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0;
    if reference.eq_ignore_ascii_case("S") || reference.eq_ignore_ascii_case("W") {
        degrees = -degrees;
    } else if !reference.eq_ignore_ascii_case("N") && !reference.eq_ignore_ascii_case("E") {
        return None;
    }
    Some(degrees)
}

fn validate_actual_format(path: &Path, kind: InputKind) -> Result<(), String> {
    let reader = ImageReader::open(path)
        .map_err(|_| "无法打开所选图像文件。".to_string())?
        .with_guessed_format()
        .map_err(|_| "无法识别照片文件内容。".to_string())?;
    let actual = reader
        .format()
        .ok_or_else(|| "文件内容不是受支持的图像格式。".to_string())?;
    if actual != kind.image_format() {
        return Err("文件扩展名与实际图像内容不匹配。".into());
    }
    Ok(())
}

fn source_dimensions(
    path: &Path,
    kind: InputKind,
    tiff: Option<&TiffSummary>,
) -> Result<(u32, u32), String> {
    #[cfg(target_os = "macos")]
    {
        let (width, height) = imageio_dimensions(path)?;
        if kind == InputKind::Dng {
            let tiff = tiff.ok_or_else(|| "DNG TIFF 目录信息缺失。".to_string())?;
            if !dng_dimensions_match(tiff, width, height) {
                return Err(format!(
                    "macOS RAW 解码器返回的 {width}x{height} 与 DNG 全尺寸/有效区域尺寸不一致；未使用嵌入预览代替原图。"
                ));
            }
        }
        Ok((width, height))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let reader = ImageReader::open(path)
            .map_err(|_| "无法打开所选图像文件。".to_string())?
            .with_guessed_format()
            .map_err(|_| "无法识别照片文件内容。".to_string())?;
        let (width, height) = reader
            .into_dimensions()
            .map_err(|_| unsupported_decoder_message(kind))?;
        if kind == InputKind::Dng {
            let tiff = tiff.ok_or_else(|| "DNG TIFF 目录信息缺失。".to_string())?;
            if !dng_dimensions_match(tiff, width, height) {
                return Err(
                    "DNG 解码尺寸与全尺寸/有效区域 TIFF 尺寸不一致；未使用嵌入预览代替原图。"
                        .into(),
                );
            }
        }
        Ok((width, height))
    }
}

fn dng_dimensions_match(summary: &TiffSummary, width: u32, height: u32) -> bool {
    [
        (summary.width, summary.height),
        (summary.active_width, summary.active_height),
        (summary.default_crop_width, summary.default_crop_height),
    ]
    .into_iter()
    .any(|(candidate_width, candidate_height)| {
        candidate_width == Some(width) && candidate_height == Some(height)
    })
}

#[cfg(not(target_os = "macos"))]
fn unsupported_decoder_message(kind: InputKind) -> String {
    if kind == InputKind::Dng {
        "当前平台没有可用的全分辨率 DNG 解码器。请使用 macOS ImageIO/Core Image，或先转换为 sRGB TIFF/JPEG。".into()
    } else {
        "无法读取图像尺寸或文件内容已损坏。".into()
    }
}

fn decode_full_rgba(
    path: &Path,
    kind: InputKind,
    width: u32,
    height: u32,
) -> Result<RgbaImage, String> {
    let capacity = checked_pixel_count(width, height)?
        .checked_mul(4)
        .ok_or_else(|| "照片解码缓冲区尺寸溢出。".to_string())?;
    let capacity =
        usize::try_from(capacity).map_err(|_| "照片超过当前平台的内存寻址范围。".to_string())?;
    let mut pixels = vec![0u8; capacity];
    #[cfg(target_os = "macos")]
    {
        let _ = kind;
        imageio_decode_rgba(path, width, height, &mut pixels)?;
        return ImageBuffer::from_raw(width, height, pixels)
            .ok_or_else(|| "ImageIO 返回了无效的全尺寸图像缓冲区。".into());
    }
    #[cfg(not(target_os = "macos"))]
    {
        let decoded = ImageReader::open(path)
            .map_err(|_| "无法打开所选图像文件。".to_string())?
            .with_guessed_format()
            .map_err(|_| "无法识别照片文件内容。".to_string())?
            .decode()
            .map_err(|_| unsupported_decoder_message(kind))?;
        let rgba = decoded.to_rgba8();
        if rgba.width() != width || rgba.height() != height {
            return Err("图像解码器返回的不是源照片全尺寸。".into());
        }
        Ok(rgba)
    }
}

#[cfg(target_os = "macos")]
fn imageio_dimensions(path: &Path) -> Result<(u32, u32), String> {
    let path_string = path.to_string_lossy();
    let path_c =
        CString::new(path_string.as_bytes()).map_err(|_| "照片路径包含无效字符。".to_string())?;
    let mut width = 0;
    let mut height = 0;
    let mut error = [0 as c_char; 1024];
    let status = unsafe {
        imprint_imageio_dimensions(
            path_c.as_ptr(),
            &mut width,
            &mut height,
            error.as_mut_ptr(),
            error.len() as u32,
        )
    };
    if status != 0 || width == 0 || height == 0 {
        return Err(native_error(&error, "macOS ImageIO 无法读取全尺寸图像。"));
    }
    Ok((width, height))
}

#[cfg(target_os = "macos")]
fn imageio_decode_rgba(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &mut [u8],
) -> Result<(), String> {
    let path_string = path.to_string_lossy();
    let path_c =
        CString::new(path_string.as_bytes()).map_err(|_| "照片路径包含无效字符。".to_string())?;
    let mut error = [0 as c_char; 1024];
    let status = unsafe {
        imprint_imageio_decode_rgba8_srgb(
            path_c.as_ptr(),
            rgba.as_mut_ptr(),
            width,
            height,
            rgba.len() as u64,
            error.as_mut_ptr(),
            error.len() as u32,
        )
    };
    if status != 0 {
        return Err(native_error(&error, "macOS 没有生成全尺寸 sRGB 图像。"));
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn has_embedded_icc(
    path: &Path,
    kind: InputKind,
    tiff: Option<&TiffSummary>,
) -> Result<bool, String> {
    match kind {
        InputKind::Jpeg => jpeg_has_icc(path).map_err(|_| "JPEG ICC 元数据结构无效。".into()),
        InputKind::Png => png_has_icc(path).map_err(|_| "PNG ICC 元数据结构无效。".into()),
        InputKind::Tiff | InputKind::Dng => Ok(tiff.is_some_and(|summary| summary.has_icc_profile)),
    }
}

fn read_tiff_summary(path: &Path) -> io::Result<TiffSummary> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let little_endian = match &header[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid byte order",
            ))
        }
    };
    if read_u16(&header[2..4], little_endian) != 42 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported TIFF header",
        ));
    }
    let first_ifd = read_u32(&header[4..8], little_endian) as u64;
    let mut summary = TiffSummary {
        width: None,
        height: None,
        active_width: None,
        active_height: None,
        default_crop_width: None,
        default_crop_height: None,
        has_dng_version: false,
        has_icc_profile: false,
    };
    let mut pending = vec![first_ifd];
    let mut visited = std::collections::HashSet::new();
    let mut best_pixels = 0u64;
    while let Some(ifd_offset) = pending.pop() {
        if ifd_offset == 0 || !visited.insert(ifd_offset) || visited.len() > 32 {
            continue;
        }
        file.seek(SeekFrom::Start(ifd_offset))?;
        let mut count_bytes = [0u8; 2];
        file.read_exact(&mut count_bytes)?;
        let count = read_u16(&count_bytes, little_endian);
        if count > 8192 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "too many TIFF fields",
            ));
        }
        let mut candidate = TiffSummary {
            width: None,
            height: None,
            active_width: None,
            active_height: None,
            default_crop_width: None,
            default_crop_height: None,
            has_dng_version: false,
            has_icc_profile: false,
        };
        for _ in 0..count {
            let mut entry = [0u8; 12];
            file.read_exact(&mut entry)?;
            let tag = read_u16(&entry[0..2], little_endian);
            let field_type = read_u16(&entry[2..4], little_endian);
            let field_count = read_u32(&entry[4..8], little_endian);
            match tag {
                256 => {
                    candidate.width = tiff_inline_integer(&entry[8..12], field_type, little_endian)
                }
                257 => {
                    candidate.height = tiff_inline_integer(&entry[8..12], field_type, little_endian)
                }
                50_706 if field_type == 1 && field_count >= 4 => candidate.has_dng_version = true,
                34_675 => candidate.has_icc_profile = true,
                50_829 if field_type == 4 && field_count == 4 => {
                    if let Some(values) =
                        tiff_u32_values(&mut file, &entry, field_count, little_endian)
                    {
                        candidate.active_height = values[2].checked_sub(values[0]);
                        candidate.active_width = values[3].checked_sub(values[1]);
                    }
                }
                50_720 if field_count == 2 => {
                    if let Some((width, height)) =
                        tiff_crop_size(&mut file, &entry, field_type, little_endian)
                    {
                        candidate.default_crop_width = Some(width);
                        candidate.default_crop_height = Some(height);
                    }
                }
                330 if field_type == 4 && field_count > 0 && field_count <= 32 => {
                    if let Some(offsets) =
                        tiff_u32_values(&mut file, &entry, field_count, little_endian)
                    {
                        pending.extend(offsets.into_iter().map(u64::from));
                    }
                }
                _ => {}
            }
        }
        let mut next_ifd = [0u8; 4];
        file.read_exact(&mut next_ifd)?;
        let next_offset = read_u32(&next_ifd, little_endian);
        if next_offset != 0 {
            pending.push(u64::from(next_offset));
        }
        summary.has_dng_version |= candidate.has_dng_version;
        summary.has_icc_profile |= candidate.has_icc_profile;
        if let (Some(width), Some(height)) = (candidate.width, candidate.height) {
            let pixels = u64::from(width) * u64::from(height);
            if pixels > best_pixels {
                best_pixels = pixels;
                summary.width = Some(width);
                summary.height = Some(height);
                summary.active_width = candidate.active_width;
                summary.active_height = candidate.active_height;
                summary.default_crop_width = candidate.default_crop_width;
                summary.default_crop_height = candidate.default_crop_height;
            }
        }
    }
    Ok(summary)
}

fn tiff_u32_values(
    file: &mut File,
    entry: &[u8; 12],
    count: u32,
    little_endian: bool,
) -> Option<Vec<u32>> {
    let count = usize::try_from(count).ok()?;
    let current = file.stream_position().ok()?;
    let value = &entry[8..12];
    let bytes_needed = count.checked_mul(4)?;
    let mut bytes = vec![0u8; bytes_needed];
    if bytes_needed <= 4 {
        bytes[..bytes_needed].copy_from_slice(&value[..bytes_needed]);
    } else {
        let offset = u64::from(read_u32(value, little_endian));
        if file.seek(SeekFrom::Start(offset)).is_err() || file.read_exact(&mut bytes).is_err() {
            let _ = file.seek(SeekFrom::Start(current));
            return None;
        }
        let _ = file.seek(SeekFrom::Start(current));
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|bytes| read_u32(bytes, little_endian))
            .collect(),
    )
}

fn tiff_crop_size(
    file: &mut File,
    entry: &[u8; 12],
    field_type: u16,
    little_endian: bool,
) -> Option<(u32, u32)> {
    let count = read_u32(&entry[4..8], little_endian);
    match field_type {
        3 if count == 2 => Some((
            u32::from(read_u16(&entry[8..10], little_endian)),
            u32::from(read_u16(&entry[10..12], little_endian)),
        )),
        4 if count == 2 => {
            let values = tiff_u32_values(file, entry, 2, little_endian)?;
            Some((values[0], values[1]))
        }
        5 if count == 2 => {
            let current = file.stream_position().ok()?;
            let offset = u64::from(read_u32(&entry[8..12], little_endian));
            if file.seek(SeekFrom::Start(offset)).is_err() {
                return None;
            }
            let mut bytes = [0u8; 16];
            let read_result = file.read_exact(&mut bytes);
            let _ = file.seek(SeekFrom::Start(current));
            read_result.ok()?;
            let w_num = read_u32(&bytes[0..4], little_endian);
            let w_den = read_u32(&bytes[4..8], little_endian);
            let h_num = read_u32(&bytes[8..12], little_endian);
            let h_den = read_u32(&bytes[12..16], little_endian);
            if w_den == 0 || h_den == 0 {
                return None;
            }
            Some((
                (w_num as f64 / w_den as f64).round() as u32,
                (h_num as f64 / h_den as f64).round() as u32,
            ))
        }
        _ => None,
    }
}

fn tiff_inline_integer(bytes: &[u8], field_type: u16, little_endian: bool) -> Option<u32> {
    match field_type {
        3 => Some(u32::from(read_u16(&bytes[..2], little_endian))),
        4 => Some(read_u32(bytes, little_endian)),
        _ => None,
    }
}

fn read_u16(bytes: &[u8], little_endian: bool) -> u16 {
    if little_endian {
        u16::from_le_bytes([bytes[0], bytes[1]])
    } else {
        u16::from_be_bytes([bytes[0], bytes[1]])
    }
}

fn read_u32(bytes: &[u8], little_endian: bool) -> u32 {
    if little_endian {
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

#[cfg(not(target_os = "macos"))]
fn jpeg_has_icc(path: &Path) -> io::Result<bool> {
    let mut file = BufReader::new(File::open(path)?);
    let mut signature = [0u8; 2];
    file.read_exact(&mut signature)?;
    if signature != [0xff, 0xd8] {
        return Ok(false);
    }
    loop {
        let mut marker_start = [0u8; 1];
        if file.read_exact(&mut marker_start).is_err() {
            return Ok(false);
        }
        if marker_start[0] != 0xff {
            continue;
        }
        let mut marker = [0u8; 1];
        file.read_exact(&mut marker)?;
        while marker[0] == 0xff {
            file.read_exact(&mut marker)?;
        }
        if marker[0] == 0xda || marker[0] == 0xd9 {
            return Ok(false);
        }
        if matches!(marker[0], 0x01 | 0xd0..=0xd7) {
            continue;
        }
        let mut length_bytes = [0u8; 2];
        file.read_exact(&mut length_bytes)?;
        let length = u16::from_be_bytes(length_bytes) as usize;
        if length < 2 {
            return Ok(false);
        }
        let data_len = length - 2;
        let mut marker_data = vec![0u8; data_len.min(14)];
        file.read_exact(&mut marker_data)?;
        if marker[0] == 0xe2 && marker_data.starts_with(b"ICC_PROFILE\0") {
            return Ok(true);
        }
        if data_len > marker_data.len() {
            file.seek(SeekFrom::Current((data_len - marker_data.len()) as i64))?;
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn png_has_icc(path: &Path) -> io::Result<bool> {
    let mut file = BufReader::new(File::open(path)?);
    let mut signature = [0u8; 8];
    file.read_exact(&mut signature)?;
    if signature != *b"\x89PNG\r\n\x1a\n" {
        return Ok(false);
    }
    loop {
        let mut header = [0u8; 8];
        if file.read_exact(&mut header).is_err() {
            return Ok(false);
        }
        let length = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
        if &header[4..8] == b"iCCP" {
            return Ok(true);
        }
        if &header[4..8] == b"IEND" {
            return Ok(false);
        }
        file.seek(SeekFrom::Current((length + 4) as i64))?;
    }
}

fn checked_pixel_count(width: u32, height: u32) -> Result<usize, String> {
    if width == 0 || height == 0 {
        return Err("照片宽度和高度必须大于零。".into());
    }
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| "照片尺寸溢出。".to_string())?;
    if pixels > MAX_PIXELS {
        return Err(format!("照片分辨率超过 {MAX_PIXELS} 像素的解码上限。"));
    }
    usize::try_from(pixels).map_err(|_| "照片超过当前平台的内存寻址范围。".into())
}

fn apply_orientation(rgba: RgbaImage, orientation: u16) -> RgbaImage {
    match orientation {
        2 => imageops::flip_horizontal(&rgba),
        3 => imageops::rotate180(&rgba),
        4 => imageops::flip_vertical(&rgba),
        5 => imageops::rotate270(&imageops::flip_horizontal(&rgba)),
        6 => imageops::rotate90(&rgba),
        7 => imageops::rotate90(&imageops::flip_horizontal(&rgba)),
        8 => imageops::rotate270(&rgba),
        _ => rgba,
    }
}

fn flatten_rgba_over_white(rgba: &RgbaImage) -> RgbImage {
    let mut rgb = RgbImage::new(rgba.width(), rgba.height());
    for (source, target) in rgba.pixels().zip(rgb.pixels_mut()) {
        let alpha = source[3] as f64 / 255.0;
        for channel in 0..3 {
            let foreground = srgb_to_linear(source[channel] as f64 / 255.0);
            let linear = foreground * alpha + (1.0 - alpha);
            target[channel] = (linear_to_srgb(linear) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn png_data_uri_rgb(image: &RgbImage) -> Result<String, String> {
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgb8,
        )
        .map_err(|error| format!("编码 PNG 预览失败：{error}"))?;
    Ok(format!("data:image/png;base64,{}", BASE64.encode(bytes)))
}

fn png_data_uri_rgba(image: &RgbaImage) -> Result<String, String> {
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgba8,
        )
        .map_err(|error| format!("编码透明 PNG 预览失败：{error}"))?;
    Ok(format!("data:image/png;base64,{}", BASE64.encode(bytes)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Png,
    Jpeg,
}

fn validate_output_extension(path: &Path, format: OutputFormat) -> Result<(), String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match (format, extension.as_str()) {
        (OutputFormat::Png, "png") | (OutputFormat::Jpeg, "jpg" | "jpeg") => Ok(()),
        (OutputFormat::Png, _) => Err("PNG 输出路径必须使用 .png 扩展名。".into()),
        (OutputFormat::Jpeg, _) => Err("JPEG 输出路径必须使用 .jpg 或 .jpeg 扩展名。".into()),
    }
}

fn write_image_no_clobber(
    target: &Path,
    format: OutputFormat,
    rgb: Option<&RgbImage>,
    rgba: Option<&RgbaImage>,
) -> Result<(), String> {
    let (temporary, file) = create_temp_file(target)?;
    let write_result = (|| -> Result<(), String> {
        let mut writer = BufWriter::new(file);
        match (format, rgb, rgba) {
            (OutputFormat::Png, Some(image), None) => PngEncoder::new(&mut writer)
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    ExtendedColorType::Rgb8,
                )
                .map_err(|error| format!("PNG 编码失败：{error}"))?,
            (OutputFormat::Png, None, Some(image)) => PngEncoder::new(&mut writer)
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    ExtendedColorType::Rgba8,
                )
                .map_err(|error| format!("透明 PNG 编码失败：{error}"))?,
            (OutputFormat::Jpeg, Some(image), None) => {
                JpegEncoder::new_with_quality(&mut writer, 95)
                    .encode(
                        image.as_raw(),
                        image.width(),
                        image.height(),
                        ExtendedColorType::Rgb8,
                    )
                    .map_err(|error| format!("JPEG 编码失败：{error}"))?
            }
            _ => return Err("输出图像通道数与格式不匹配。".into()),
        }
        writer
            .flush()
            .map_err(|error| format!("写入输出图像失败：{error}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("同步输出图像失败：{error}"))?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = fs::hard_link(&temporary, target) {
        let _ = fs::remove_file(&temporary);
        return if error.kind() == io::ErrorKind::AlreadyExists {
            Err("目标文件已存在，已保留原文件。".into())
        } else {
            Err(format!(
                "无法安全提交输出文件（目标可能不支持原子无覆盖创建）：{error}"
            ))
        };
    }
    fs::remove_file(&temporary)
        .map_err(|error| format!("输出已提交，但无法清理临时文件：{error}"))?;
    Ok(())
}

fn create_temp_file(target: &Path) -> Result<(PathBuf, File), String> {
    let parent = target
        .parent()
        .ok_or_else(|| "输出路径缺少目录。".to_string())?;
    if !parent.is_dir() {
        return Err("输出目录不存在。".into());
    }
    let stem = target
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "输出文件名无效。".to_string())?;
    for _ in 0..12 {
        let temporary = parent.join(format!(".{stem}.{}.tmp", Uuid::new_v4()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("无法创建输出临时文件：{error}")),
        }
    }
    Err("无法分配唯一的输出临时文件名。".into())
}

fn normalize_target_path(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let canonical_parent = parent
        .canonicalize()
        .map_err(|_| "输出目录不存在或无法访问。".to_string())?;
    if !canonical_parent.is_dir() {
        return Err("输出目录不是文件夹。".into());
    }
    let name = path
        .file_name()
        .ok_or_else(|| "输出文件名无效。".to_string())?;
    Ok(canonical_parent.join(name))
}

fn save_project_sync(
    state: &GeoStateInner,
    id: &str,
    config: &GeoConfig,
    output_path: &Path,
) -> Result<String, String> {
    config.validate()?;
    let session = get_session(state, id)?;
    let target = normalize_target_path(output_path)?;
    if target == session.source_path {
        return Err("项目路径不能覆盖原始照片。".into());
    }
    let project = ProjectFile {
        format: PROJECT_FORMAT.into(),
        version: PROJECT_VERSION,
        provider: config.provider,
        mock: config.provider == 0,
        source: ProjectSource {
            path: session.source_path.clone(),
            source_name: session.source_name.clone(),
            width: session.width,
            height: session.height,
            gps_source: session.exif_location.as_ref().map(|_| "exif".into()),
        },
        config: config.clone(),
    };
    let bytes =
        serde_json::to_vec_pretty(&project).map_err(|error| format!("序列化项目失败：{error}"))?;
    let previous = session
        .project_path
        .lock()
        .map_err(|_| "项目路径锁不可用。".to_string())?
        .clone();
    write_project_atomic(
        &target,
        &bytes,
        previous.as_deref() == Some(target.as_path()),
    )?;
    *session
        .project_path
        .lock()
        .map_err(|_| "项目路径锁不可用。".to_string())? = Some(target.clone());
    Ok(target.to_string_lossy().into_owned())
}

fn write_project_atomic(target: &Path, bytes: &[u8], may_replace: bool) -> Result<(), String> {
    let (temporary, mut file) = create_temp_file(target)?;
    let write_result = file.write_all(bytes).and_then(|_| file.sync_all());
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(format!("写入项目临时文件失败：{error}"));
    }
    if may_replace {
        if !target.is_file() {
            let _ = fs::remove_file(&temporary);
            return Err("当前项目文件已不存在，请另存为新项目。".into());
        }
        if let Err(error) = fs::rename(&temporary, target) {
            let _ = fs::remove_file(&temporary);
            return Err(format!("原子更新项目文件失败：{error}"));
        }
        return Ok(());
    }
    if let Err(error) = fs::hard_link(&temporary, target) {
        let _ = fs::remove_file(&temporary);
        return if error.kind() == io::ErrorKind::AlreadyExists {
            Err("目标项目已存在；请打开该项目后再保存，或另存为新文件。".into())
        } else {
            Err(format!("无法安全创建项目文件：{error}"))
        };
    }
    fs::remove_file(&temporary)
        .map_err(|error| format!("项目已保存，但临时文件清理失败：{error}"))?;
    Ok(())
}

fn load_project_sync(
    state: &Arc<GeoStateInner>,
    project_path: &Path,
) -> Result<GeoLoadedProject, String> {
    let path = project_path
        .canonicalize()
        .map_err(|_| "项目文件不存在或无法访问。".to_string())?;
    let metadata = fs::metadata(&path).map_err(|_| "无法读取项目文件信息。".to_string())?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > 2 * 1024 * 1024 {
        return Err("项目文件为空、过大或不是普通文件。".into());
    }
    let project: ProjectFile =
        serde_json::from_reader(File::open(&path).map_err(|_| "无法打开项目文件。".to_string())?)
            .map_err(|_| "项目文件缺少有效配置或版本不受支持。".to_string())?;
    if project.format != PROJECT_FORMAT || project.version != PROJECT_VERSION {
        return Err("项目文件类型或版本不受支持。".into());
    }
    project.config.validate()?;
    if project.provider != project.config.provider || project.mock != (project.config.provider == 0)
    {
        return Err("项目的数据源元信息与叠加配置不一致。".into());
    }
    let photo = open_sync(state, &project.source.path)?;
    if photo.width != project.source.width
        || photo.height != project.source.height
        || photo.source_name != project.source.source_name
        || project.source.gps_source.as_deref() != photo.exif_location.as_ref().map(|_| "exif")
    {
        if let Ok(mut sessions) = state.sessions.lock() {
            sessions.remove(&photo.id);
        }
        return Err("项目记录的源照片尺寸与当前文件不一致。".into());
    }
    let session = get_session(state, &photo.id)?;
    *session
        .project_path
        .lock()
        .map_err(|_| "项目路径锁不可用。".to_string())? = Some(path);
    Ok(GeoLoadedProject {
        photo,
        config: project.config,
    })
}

fn native_error(buffer: &[c_char], fallback: &str) -> String {
    let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .trim()
        .to_string();
    if message.is_empty() {
        fallback.into()
    } else {
        message
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn default_config_matches_overlay_contract_and_serializes_switches_as_bools() {
        let config = GeoConfig::default();
        assert_eq!(config.opacity, 0.18);
        assert_eq!(config.radius_km, 3.0);
        assert_eq!(config.line_width, 1.0);
        assert!(!config.enabled);
        assert!(config.auto_sky_fade && config.labels && config.north_arrow);
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["enabled"], false);
        assert_eq!(json["provider"], 0);
        assert_eq!(json["radius_km"], 3.0);
        assert!(serde_json::from_value::<GeoConfig>(json).is_ok());
        assert_eq!(config.overpass_endpoint, DEFAULT_OVERPASS_ENDPOINT);
        let camel_case = serde_json::json!({"radiusKm": 5.0, "enabled": true});
        let parsed: GeoConfig = serde_json::from_value(camel_case).unwrap();
        assert_eq!(parsed.radius_km, 5.0);
        assert!(parsed.enabled);
    }

    #[test]
    fn config_rejects_non_finite_coordinates_and_native_range_errors() {
        let mut config = GeoConfig::default();
        config.latitude = f64::NAN;
        assert!(config.validate().unwrap_err().contains("NaN"));
        config.latitude = 91.0;
        assert!(config.validate().unwrap_err().contains("纬度"));
        config.latitude = 0.0;
        config.style = 4;
        assert!(config.validate().unwrap_err().contains("样式"));
    }

    #[test]
    fn supported_input_extension_is_explicit_and_case_insensitive() {
        assert_eq!(
            InputKind::from_path(Path::new("image.JPG")).unwrap(),
            InputKind::Jpeg
        );
        assert_eq!(
            InputKind::from_path(Path::new("image.tiff")).unwrap(),
            InputKind::Tiff
        );
        assert_eq!(
            InputKind::from_path(Path::new("image.DNG")).unwrap(),
            InputKind::Dng
        );
        assert!(InputKind::from_path(Path::new("image.gif")).is_err());
    }

    #[test]
    fn gps_coordinates_apply_south_and_west_signs() {
        assert_eq!(gps_decimal_degrees([38.0, 30.0, 0.0], "N"), Some(38.5));
        assert_eq!(gps_decimal_degrees([122.0, 15.0, 0.0], "W"), Some(-122.25));
        assert_eq!(gps_decimal_degrees([1.0, 2.0, 3.0], "X"), None);
    }

    #[test]
    fn transparent_pixels_flatten_to_white_and_opaque_pixels_keep_color() {
        let source = RgbaImage::from_raw(2, 1, vec![12, 34, 56, 0, 200, 100, 50, 255]).unwrap();
        let result = flatten_rgba_over_white(&source);
        assert_eq!(result.get_pixel(0, 0), &Rgb([255, 255, 255]));
        assert_eq!(result.get_pixel(1, 0), &Rgb([200, 100, 50]));
    }

    #[test]
    fn orientation_rotations_swap_dimensions_and_preserve_corner_order() {
        let source = RgbaImage::from_raw(
            2,
            2,
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
            ],
        )
        .unwrap();
        let orientation_one = apply_orientation(source.clone(), 1);
        assert_eq!(orientation_one.get_pixel(0, 0).0, [255, 0, 0, 255]);
        let orientation_six = apply_orientation(source.clone(), 6);
        assert_eq!((orientation_six.width(), orientation_six.height()), (2, 2));
        assert_eq!(orientation_six.get_pixel(1, 0).0, [255, 0, 0, 255]);
        let orientation_eight = apply_orientation(source, 8);
        assert_eq!(orientation_eight.get_pixel(0, 1).0, [255, 0, 0, 255]);
    }

    #[test]
    fn output_creation_is_atomic_and_never_overwrites_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("result.png");
        let source = RgbImage::from_pixel(2, 2, Rgb([25, 50, 75]));
        write_image_no_clobber(&target, OutputFormat::Png, Some(&source), None).unwrap();
        let original = fs::read(&target).unwrap();
        assert!(!original.is_empty());
        let error =
            write_image_no_clobber(&target, OutputFormat::Png, Some(&source), None).unwrap_err();
        assert!(error.contains("已存在"));
        assert_eq!(fs::read(&target).unwrap(), original);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn older_project_without_config_defaults_to_disabled_overlay() {
        let project = r#"{"format":"imprint.geographic-overlay","version":1,"provider":0,"mock":true,"source":{"path":"photo.dng","source_name":"photo.dng","width":2,"height":2,"gps_source":null}}"#;
        let parsed: ProjectFile = serde_json::from_str(project).unwrap();
        assert!(!parsed.config.enabled);
        assert_eq!(parsed.config, GeoConfig::default());
    }

    #[test]
    fn disabled_osm_preview_does_not_fetch_or_require_features() {
        let directory = tempfile::tempdir().unwrap();
        let input_path = directory.path().join("small.png");
        RgbImage::from_pixel(4, 3, Rgb([20, 40, 60]))
            .save_with_format(&input_path, ImageFormat::Png)
            .unwrap();
        let state = Arc::new(GeoStateInner::default());
        let photo = open_sync(&state, &input_path).unwrap();
        let config = GeoConfig {
            provider: 1,
            overpass_endpoint: "http://127.0.0.1:1/api/interpreter".into(),
            ..GeoConfig::default()
        };

        let preview = preview_sync(&state, &photo.id, &config).unwrap();

        assert_eq!(preview.provider, "osm");
        assert_eq!(preview.feature_count, 0);
        assert!(!preview.cache_hit);
        assert!(preview.warning.contains("启用叠加后将查询"));
    }

    #[test]
    fn open_preview_save_load_and_export_keep_full_dimensions_and_session_identity() {
        let directory = tempfile::tempdir().unwrap();
        let input_path = directory.path().join("wide.png");
        let mut input = RgbImage::new(1801, 905);
        for (x, y, pixel) in input.enumerate_pixels_mut() {
            *pixel = Rgb([(x % 251) as u8, (y % 239) as u8, 80]);
        }
        input
            .save_with_format(&input_path, ImageFormat::Png)
            .unwrap();

        let state = Arc::new(GeoStateInner::default());
        let photo = open_sync(&state, &input_path).unwrap();
        assert_eq!((photo.width, photo.height), (1801, 905));
        let config = GeoConfig::default();
        let preview = preview_sync(&state, &photo.id, &config).unwrap();
        assert_eq!(
            (preview.width, preview.height),
            (1801, 905),
            "preview must return every original pixel, not a 1600px thumbnail"
        );
        let encoded = preview
            .image
            .strip_prefix("data:image/png;base64,")
            .unwrap();
        let bytes = BASE64.decode(encoded).unwrap();
        let preview_image = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(preview_image.dimensions(), input.dimensions());
        assert_eq!(
            preview_image, input,
            "disabled full-resolution preview preserves every source pixel"
        );
        assert!(preview.image.starts_with("data:image/png;base64,"));
        assert!(preview.overlay.starts_with("data:image/png;base64,"));

        let project_path = directory.path().join("wide.imprint-geo.json");
        save_project_sync(&state, &photo.id, &config, &project_path).unwrap();
        let saved_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&project_path).unwrap()).unwrap();
        assert_eq!(saved_json["config"]["radius_km"], 3.0);
        assert_eq!(saved_json["config"]["enabled"], false);

        let loaded = load_project_sync(&state, &project_path).unwrap();
        assert_ne!(loaded.photo.id, photo.id);
        assert_eq!(loaded.config, config);
        assert_eq!((loaded.photo.width, loaded.photo.height), (1801, 905));
        let output_path = directory.path().join("wide-output.png");
        let actual_path = export_sync(
            &state,
            &loaded.photo.id,
            &loaded.config,
            &output_path,
            "png",
            false,
        )
        .unwrap();
        assert_eq!(
            Path::new(&actual_path).canonicalize().unwrap(),
            output_path.canonicalize().unwrap()
        );
        let output = ImageReader::open(&output_path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
        assert_eq!((output.width(), output.height()), (1801, 905));
        assert_eq!(output.get_pixel(1700, 900), input.get_pixel(1700, 900));
    }

    #[test]
    fn full_resolution_osm_preview_export_and_disabled_identity_use_one_query() {
        use std::net::TcpListener;
        use std::time::{Duration, Instant};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/api/interpreter", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    Err(error) => panic!("local OSM fixture connection: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = stream.read(&mut chunk).unwrap();
                assert!(n > 0);
                request.extend_from_slice(&chunk[..n]);
                if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]);
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|value| value.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            assert!(request.starts_with(b"POST /api/interpreter"));
            let body = r#"{"elements":[{"type":"way","id":1,"tags":{"highway":"residential","name:en":"TEST ROAD"},"geometry":[{"lat":1.0,"lon":1.999},{"lat":1.0,"lon":2.001}]},{"type":"node","id":2,"tags":{"place":"village","name:en":"TEST PLACE"},"lat":1.0005,"lon":2.0}]}"#;
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
        });
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("finished.png");
        let input = RgbImage::from_pixel(1801, 905, image::Rgb([60, 80, 120]));
        input
            .save_with_format(&source_path, ImageFormat::Png)
            .unwrap();
        let original_bytes = fs::read(&source_path).unwrap();
        let state = Arc::new(GeoStateInner::default());
        let photo = open_sync(&state, &source_path).unwrap();
        let mut config = GeoConfig {
            enabled: true,
            has_location: true,
            provider: 1,
            latitude: 1.0,
            longitude: 2.0,
            radius_km: 0.2,
            auto_sky_fade: false,
            overpass_endpoint: endpoint,
            ..GeoConfig::default()
        };
        let first = preview_sync(&state, &photo.id, &config).unwrap();
        assert_eq!((first.width, first.height), (1801, 905));
        assert_eq!(first.provider, "osm");
        assert_eq!(first.feature_count, 2);
        assert!(!first.cache_hit);
        assert!(first.attribution.contains("OpenStreetMap contributors"));
        let overlay = BASE64
            .decode(
                first
                    .overlay
                    .strip_prefix("data:image/png;base64,")
                    .unwrap(),
            )
            .unwrap();
        let overlay = image::load_from_memory(&overlay).unwrap().to_rgba8();
        assert_eq!(overlay.dimensions(), input.dimensions());
        assert!(
            overlay.pixels().any(|pixel| pixel[3] > 0),
            "OSM geometry must reach the native overlay, not only a status field"
        );
        config.opacity = 0.3;
        let second = preview_sync(&state, &photo.id, &config).unwrap();
        assert!(second.cache_hit);
        let output = directory.path().join("overlay.png");
        export_sync(&state, &photo.id, &config, &output, "png", false).unwrap();
        assert_eq!(
            ImageReader::open(output).unwrap().decode().unwrap().width(),
            1801
        );
        config.enabled = false;
        let disabled = preview_sync(&state, &photo.id, &config).unwrap();
        let disabled = BASE64
            .decode(
                disabled
                    .image
                    .strip_prefix("data:image/png;base64,")
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(image::load_from_memory(&disabled).unwrap().to_rgb8(), input);
        assert_eq!(fs::read(source_path).unwrap(), original_bytes);
        server.join().unwrap();
    }

    #[test]
    #[ignore = "requires IMPRINT_GEO_SMOKE_INPUT to point to a real DNG"]
    fn real_dng_open_and_export_uses_full_resolution_rgb_pixels() {
        let path = std::env::var("IMPRINT_GEO_SMOKE_INPUT").expect("set IMPRINT_GEO_SMOKE_INPUT");
        let decoded = decode_source(Path::new(&path)).unwrap();
        assert!(decoded
            .source_path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("dng")));
        assert!(decoded.rgb.width() > 0 && decoded.rgb.height() > 0);
        assert!(
            decoded
                .rgb
                .pixels()
                .any(|pixel| pixel.0.iter().any(|channel| *channel > 4)),
            "DNG decoder returned a black image"
        );
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("full_resolution.png");
        let id = Uuid::new_v4().to_string();
        let preview = decoded.rgb.clone();
        assert_eq!(preview.dimensions(), decoded.rgb.dimensions());
        let session = Arc::new(GeoSession {
            source_path: decoded.source_path.clone(),
            source_name: decoded.source_name.clone(),
            width: decoded.rgb.width(),
            height: decoded.rgb.height(),
            preview_rgb: preview,
            native_backend: "native test source".into(),
            render_mutex: Mutex::new(()),
            exif_location: decoded.exif_location.clone(),
            project_path: Mutex::new(None),
            preview_generation: AtomicU64::new(0),
        });
        let state = GeoStateInner::default();
        state.sessions.lock().unwrap().insert(id.clone(), session);
        let output = export_sync(
            &state,
            &id,
            &GeoConfig::default(),
            &destination,
            "png",
            false,
        )
        .unwrap();
        assert_eq!(
            Path::new(&output).canonicalize().unwrap(),
            destination.canonicalize().unwrap()
        );
        let exported = ImageReader::open(destination).unwrap().decode().unwrap();
        assert_eq!(
            (exported.width(), exported.height()),
            (decoded.rgb.width(), decoded.rgb.height())
        );
    }

    #[test]
    #[ignore = "requires IMPRINT_GEO_SMOKE_DIR with orientation and alpha fixtures"]
    fn image_io_orientation_alpha_and_tiff_fixtures_from_env() {
        let directory = PathBuf::from(
            std::env::var("IMPRINT_GEO_SMOKE_DIR").expect("set IMPRINT_GEO_SMOKE_DIR"),
        );
        let normal = decode_source(&directory.join("orientation1.png")).unwrap();
        assert_eq!((normal.rgb.width(), normal.rgb.height()), (160, 120));
        let normal_top = normal.rgb.get_pixel(80, 5);
        let normal_bottom = normal.rgb.get_pixel(80, 115);
        assert!(
            normal_top[0] > normal_top[2],
            "orientation 1 top should be red"
        );
        assert!(
            normal_bottom[2] > normal_bottom[0],
            "orientation 1 bottom should be blue"
        );

        let rotated = decode_source(&directory.join("orientation6.jpg")).unwrap();
        assert_eq!((rotated.rgb.width(), rotated.rgb.height()), (120, 160));
        let rotated_left = rotated.rgb.get_pixel(5, 80);
        let rotated_right = rotated.rgb.get_pixel(115, 80);
        assert!(
            rotated_left[2] > rotated_left[0],
            "orientation 6 left should be blue"
        );
        assert!(
            rotated_right[0] > rotated_right[2],
            "orientation 6 right should be red"
        );

        let alpha = decode_source(&directory.join("alpha.png")).unwrap();
        assert_eq!((alpha.rgb.width(), alpha.rgb.height()), (160, 120));
        let alpha_top = alpha.rgb.get_pixel(0, 0);
        assert!(
            alpha_top[0] > 220 && alpha_top[1] > 180 && alpha_top[2] > 170,
            "half-transparent red pixels should flatten over white, got {alpha_top:?}"
        );
        let tiff = decode_source(&directory.join("rgb.tif")).unwrap();
        assert_eq!((tiff.rgb.width(), tiff.rgb.height()), (160, 120));
        let gray16 = decode_source(&directory.join("gray16.tif")).unwrap();
        assert_eq!((gray16.rgb.width(), gray16.rgb.height()), (160, 120));
        assert!(gray16.rgb.pixels().any(|pixel| pixel[0] > 4));
    }
}
