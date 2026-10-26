use reqwest::blocking::{Client, Response};
use reqwest::header::{ACCEPT, CONTENT_TYPE, RETRY_AFTER};
use reqwest::redirect::Policy;
use reqwest::StatusCode;
use reqwest::Url;
use reqwest::{NoProxy, Proxy};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Read;
#[cfg(target_os = "macos")]
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_ENDPOINT: &str = "https://overpass-api.de/api/interpreter";
const FALLBACK_ENDPOINT: &str = "https://overpass.private.coffee/api/interpreter";
const USER_AGENT: &str = "Imprint/3.0.5 (https://github.com/luqie0106/Imprint)";
const MAX_RESPONSE_BYTES: u64 = 20 * 1024 * 1024;
const MAX_FEATURES: usize = 10_000;
const MAX_POINTS: usize = 500_000;
const MAX_CACHED_QUERIES: usize = 4;
const MAX_CACHED_FAILURES: usize = 16;
const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const FAILURE_CACHE_TTL: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GeoPoint {
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GeoFeature {
    /// Native kind values: road=0, water=1, coast=2, place=3.
    pub kind: u32,
    pub points: Vec<GeoPoint>,
    pub label: String,
}

#[derive(Clone, Debug)]
pub(crate) struct OsmFetch {
    pub features: Arc<Vec<GeoFeature>>,
    pub cache_hit: bool,
    pub endpoint_host: String,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    endpoint: String,
    latitude_bits: u64,
    longitude_bits: u64,
    radius_bits: u64,
}

struct CachedFeatures {
    expires_at: Instant,
    features: Arc<Vec<GeoFeature>>,
    endpoint_host: String,
}

struct CachedFailure {
    expires_at: Instant,
    message: String,
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<CacheKey, CachedFeatures>,
    lru: VecDeque<CacheKey>,
    failures: HashMap<CacheKey, CachedFailure>,
    failure_lru: VecDeque<CacheKey>,
    cooldown_until: Option<Instant>,
    cooldown_status: Option<u16>,
}

pub(crate) struct OsmClient {
    client: Result<Client, String>,
    cache: Mutex<CacheState>,
}

impl Default for OsmClient {
    fn default() -> Self {
        let client =
            build_http_client().map_err(|error| format!("创建 Overpass HTTP 客户端失败：{error}"));
        Self {
            client,
            cache: Mutex::new(CacheState::default()),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ProxySettings {
    http: Option<String>,
    https: Option<String>,
}

fn build_http_client() -> Result<Client, String> {
    let mut builder = Client::builder()
        .timeout(Duration::from_secs(25))
        .connect_timeout(Duration::from_secs(10))
        .redirect(Policy::none())
        .user_agent(USER_AGENT);
    let env_exclusions = std::env::var("no_proxy")
        .or_else(|_| std::env::var("NO_PROXY"))
        .unwrap_or_default();
    let exclusions = if env_exclusions.trim().is_empty() {
        "localhost,127.0.0.1,::1".to_string()
    } else {
        format!("{env_exclusions},localhost,127.0.0.1,::1")
    };
    let no_proxy = NoProxy::from_string(&exclusions);
    if let Some(proxy_url) = std::env::var("IMPRINT_GEO_HTTP_PROXY")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        let proxy = Proxy::all(&proxy_url)
            .map_err(|error| format!("OSM HTTP 代理配置无效：{}", error.without_url()))?
            .no_proxy(no_proxy);
        builder = builder.proxy(proxy);
    } else {
        let settings = proxy_settings();
        if let Some(proxy_url) = settings.http {
            let proxy = Proxy::http(&proxy_url)
                .map_err(|error| format!("OSM HTTP 代理配置无效：{}", error.without_url()))?
                .no_proxy(no_proxy.clone());
            builder = builder.proxy(proxy);
        }
        if let Some(proxy_url) = settings.https {
            let proxy = Proxy::https(&proxy_url)
                .map_err(|error| format!("OSM HTTPS 代理配置无效：{}", error.without_url()))?
                .no_proxy(no_proxy);
            builder = builder.proxy(proxy);
        }
    }
    builder
        .build()
        .map_err(|error| format!("初始化 OSM HTTP 连接失败：{}", error.without_url()))
}

fn proxy_settings() -> ProxySettings {
    let env_proxy = |names: &[&str]| {
        names.iter().find_map(|name| {
            std::env::var(name)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .and_then(normalize_proxy_url)
        })
    };
    let all = env_proxy(&["all_proxy", "ALL_PROXY"]);
    let https = env_proxy(&["https_proxy", "HTTPS_PROXY"])
        .or_else(|| all.clone())
        .or_else(|| env_proxy(&["http_proxy", "HTTP_PROXY"]));
    let http = env_proxy(&["http_proxy", "HTTP_PROXY"])
        .or_else(|| all.clone())
        .or_else(|| https.clone());
    if http.is_some() || https.is_some() {
        return ProxySettings { http, https };
    }
    #[cfg(target_os = "macos")]
    {
        return macos_static_proxy_settings();
    }
    #[cfg(not(target_os = "macos"))]
    ProxySettings::default()
}

fn normalize_proxy_url(value: String) -> Option<String> {
    let value = value.trim();
    let url_string = if value.contains("://") {
        value.to_string()
    } else {
        format!("http://{value}")
    };
    let url = Url::parse(&url_string).ok()?;
    matches!(url.scheme(), "http" | "https").then_some(url.to_string())
}

#[cfg(target_os = "macos")]
fn macos_static_proxy_settings() -> ProxySettings {
    let output = Command::new("/usr/sbin/scutil").arg("--proxy").output();
    let Ok(output) = output else {
        return ProxySettings::default();
    };
    if !output.status.success() {
        return ProxySettings::default();
    }
    parse_static_proxy_settings(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(any(target_os = "macos", test))]
fn parse_static_proxy_settings(contents: &str) -> ProxySettings {
    let values = contents
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            Some((key.trim(), value.trim()))
        })
        .collect::<HashMap<_, _>>();
    let static_proxy = |kind: &str| {
        if values.get(format!("{kind}Enable").as_str()).copied() != Some("1") {
            return None;
        }
        let host = values.get(format!("{kind}Proxy").as_str()).copied()?;
        let port = values.get(format!("{kind}Port").as_str()).copied()?;
        let port: u16 = port.parse().ok()?;
        let host = host.trim_matches('"');
        if host.is_empty() || host.chars().any(char::is_whitespace) {
            return None;
        }
        let host = if host.contains(':') && !host.starts_with('[') {
            format!("[{host}]")
        } else {
            host.to_string()
        };
        let url = format!("http://{host}:{port}");
        Url::parse(&url).ok().map(|_| url)
    };
    ProxySettings {
        http: static_proxy("HTTP"),
        https: static_proxy("HTTPS"),
    }
}

impl OsmClient {
    pub(crate) fn fetch(
        &self,
        endpoint: &str,
        latitude: f64,
        longitude: f64,
        radius_km: f64,
    ) -> Result<OsmFetch, String> {
        validate_query(latitude, longitude, radius_km)?;
        let endpoint_url = validate_endpoint(endpoint)?;
        let configured_endpoint = endpoint_url.as_str().to_string();
        let key = CacheKey {
            endpoint: configured_endpoint.clone(),
            latitude_bits: latitude.to_bits(),
            longitude_bits: longitude.to_bits(),
            radius_bits: radius_km.to_bits(),
        };
        let query = build_query(latitude, longitude, radius_km);
        let client = self.client.as_ref().map_err(Clone::clone)?;

        // The lock covers the request as well as lookup/insert. This coalesces concurrent
        // requests for the same query and serializes different requests to respect public limits.
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| "Overpass 查询缓存锁不可用。".to_string())?;
        let now = Instant::now();
        cache.entries.retain(|_, cached| cached.expires_at > now);
        let live_keys = cache.entries.keys().cloned().collect::<HashSet<_>>();
        cache.lru.retain(|candidate| live_keys.contains(candidate));
        cache.failures.retain(|_, cached| cached.expires_at > now);
        let failed_keys = cache.failures.keys().cloned().collect::<HashSet<_>>();
        cache
            .failure_lru
            .retain(|candidate| failed_keys.contains(candidate));
        if let Some(cached) = cache.entries.get(&key) {
            let result = OsmFetch {
                features: Arc::clone(&cached.features),
                cache_hit: true,
                endpoint_host: cached.endpoint_host.clone(),
            };
            cache.lru.retain(|candidate| candidate != &key);
            cache.lru.push_back(key);
            return Ok(result);
        }
        if let Some(cached) = cache.failures.get(&key) {
            return Err(cached.message.clone());
        }
        if let Some(until) = cache.cooldown_until {
            if until > Instant::now() {
                let seconds = until
                    .saturating_duration_since(Instant::now())
                    .as_secs()
                    .max(1);
                let status = cache.cooldown_status.unwrap_or_default();
                return Err(format!(
                    "Overpass 最近返回 HTTP {status}，正在遵守服务冷却期；请在约 {seconds} 秒后重试。"
                ));
            }
            cache.cooldown_until = None;
            cache.cooldown_status = None;
        }

        let result = match request_features(client, &endpoint_url, &query) {
            Ok(features) => Ok((
                features,
                endpoint_url.host_str().unwrap_or("Overpass").to_string(),
            )),
            Err(primary_error)
                if configured_endpoint == DEFAULT_ENDPOINT
                    && should_try_fallback(&primary_error) =>
            {
                let fallback_url = validate_endpoint(FALLBACK_ENDPOINT)?;
                request_features(client, &fallback_url, &query)
                    .map(|features| {
                        (
                            features,
                            fallback_url.host_str().unwrap_or("Overpass").to_string(),
                        )
                    })
                    .map_err(|error| {
                        format!(
                            "默认 Overpass 服务不可用（{primary_error}）；备用公共服务也失败（{error}）。"
                        )
                    })
            }
            Err(error) => Err(error),
        };
        let (features, actual_endpoint_host) = match result {
            Ok(result) => result,
            Err(error) => {
                let mut failure_ttl = FAILURE_CACHE_TTL;
                if error.contains("HTTP 429") || error.contains("HTTP 406") {
                    let cooldown = retry_after_cooldown(&error);
                    cache.cooldown_until = Some(Instant::now() + cooldown);
                    failure_ttl = cooldown;
                    cache.cooldown_status = if error.contains("HTTP 429") {
                        Some(429)
                    } else {
                        Some(406)
                    };
                }
                insert_failure(&mut cache, key.clone(), error.clone(), failure_ttl);
                return Err(error);
            }
        };

        if features.is_empty() {
            let error = "Overpass 查询成功，但该位置和半径内没有可绘制的 OSM 道路、水系、海岸线或英文地名。".to_string();
            insert_failure(&mut cache, key, error.clone(), FAILURE_CACHE_TTL);
            return Err(error);
        }
        let features = Arc::new(features);
        cache.entries.insert(
            key.clone(),
            CachedFeatures {
                expires_at: Instant::now() + CACHE_TTL,
                features: Arc::clone(&features),
                endpoint_host: actual_endpoint_host.clone(),
            },
        );
        cache.lru.retain(|candidate| candidate != &key);
        cache.lru.push_back(key);
        while cache.entries.len() > MAX_CACHED_QUERIES {
            if let Some(oldest) = cache.lru.pop_front() {
                cache.entries.remove(&oldest);
            } else {
                break;
            }
        }
        Ok(OsmFetch {
            features,
            cache_hit: false,
            endpoint_host: actual_endpoint_host,
        })
    }
}

fn insert_failure(cache: &mut CacheState, key: CacheKey, message: String, ttl: Duration) {
    cache.failures.insert(
        key.clone(),
        CachedFailure {
            expires_at: Instant::now() + ttl,
            message,
        },
    );
    cache.failure_lru.retain(|candidate| candidate != &key);
    cache.failure_lru.push_back(key);
    while cache.failures.len() > MAX_CACHED_FAILURES {
        if let Some(oldest) = cache.failure_lru.pop_front() {
            cache.failures.remove(&oldest);
        } else {
            break;
        }
    }
}

fn retry_after_cooldown(error: &str) -> Duration {
    let retry_after = error
        .split_once("Retry-After: ")
        .and_then(|(_, rest)| rest.split('；').next())
        .map(str::trim);
    let seconds = retry_after
        .and_then(|value| {
            value.parse::<u64>().ok().or_else(|| {
                httpdate::parse_http_date(value)
                    .ok()?
                    .duration_since(std::time::SystemTime::now())
                    .ok()
                    .map(|duration| duration.as_secs())
            })
        })
        .unwrap_or(30)
        .clamp(30, 300);
    Duration::from_secs(seconds)
}

fn validate_query(latitude: f64, longitude: f64, radius_km: f64) -> Result<(), String> {
    if !latitude.is_finite()
        || !longitude.is_finite()
        || !radius_km.is_finite()
        || !(-90.0..=90.0).contains(&latitude)
        || !(-180.0..=180.0).contains(&longitude)
    {
        return Err("Overpass 查询坐标必须是有效的经纬度。".into());
    }
    if !(0.05..=20.0).contains(&radius_km) {
        return Err("公共 Overpass 查询范围必须在 0.05 到 20 公里之间；请手动缩小范围。".into());
    }
    Ok(())
}

pub(crate) fn validate_endpoint(endpoint: &str) -> Result<Url, String> {
    let url = Url::parse(endpoint.trim())
        .map_err(|_| "Overpass 地址必须是有效的 HTTP 或 HTTPS URL。".to_string())?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("Overpass 地址只允许有效的 HTTP 或 HTTPS URL。".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("Overpass 地址不能包含用户名或密码。".into());
    }
    if url.query().is_some() {
        return Err("Overpass 地址不能包含 URL query 参数。".into());
    }
    if url.fragment().is_some() {
        return Err("Overpass 地址不能包含 URL fragment。".into());
    }
    Ok(url)
}

fn build_query(latitude: f64, longitude: f64, radius_km: f64) -> String {
    let radius_meters = (radius_km * 1000.0).ceil() as u32;
    format!(
        "[out:json][timeout:20];(way(around:{radius_meters},{latitude:.8},{longitude:.8})[highway];way(around:{radius_meters},{latitude:.8},{longitude:.8})[waterway];way(around:{radius_meters},{latitude:.8},{longitude:.8})[natural=coastline];way(around:{radius_meters},{latitude:.8},{longitude:.8})[natural=water];node(around:{radius_meters},{latitude:.8},{longitude:.8})[place];);out geom tags;"
    )
}

fn request_features(
    client: &Client,
    endpoint: &Url,
    query: &str,
) -> Result<Vec<GeoFeature>, String> {
    let response = client
        .post(endpoint.clone())
        .header(ACCEPT, "application/json")
        .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
        .form(&[("data", query)])
        .send()
        .map_err(|error| {
            let error = error.without_url();
            format!(
                "请求 {} 失败：{error}",
                endpoint.host_str().unwrap_or("Overpass")
            )
        })?;
    read_response(response)
}

fn read_response(response: Response) -> Result<Vec<GeoFeature>, String> {
    let status = response.status();
    if status == StatusCode::TOO_MANY_REQUESTS {
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(|value| format!("服务建议 Retry-After: {value}；"))
            .unwrap_or_default();
        return Err(format!(
            "Overpass 请求被限流（HTTP 429）。{retry_after}请稍后重试。"
        ));
    }
    if status.is_server_error() {
        return Err(format!("Overpass 服务暂时不可用（HTTP {status}）。"));
    }
    if !status.is_success() {
        return Err(format!("Overpass 请求失败（HTTP {status}）。"));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES)
    {
        return Err("Overpass 响应超过 20 MiB 上限。".into());
    }
    let mut bytes = Vec::new();
    response
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("读取 Overpass 响应失败：{error}"))?;
    if bytes.len() as u64 > MAX_RESPONSE_BYTES {
        return Err("Overpass 响应超过 20 MiB 上限。".into());
    }
    let json: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Overpass 返回了无效 JSON：{error}"))?;
    if let Some(remark) = json.get("remark").and_then(Value::as_str) {
        let remark = remark.trim();
        let excerpt = remark.chars().take(500).collect::<String>();
        return Err(format!("Overpass 查询失败：{excerpt}"));
    }
    parse_elements(
        json.get("elements")
            .and_then(Value::as_array)
            .ok_or_else(|| "Overpass 响应缺少 elements 数组。".to_string())?,
    )
}

fn parse_elements(elements: &[Value]) -> Result<Vec<GeoFeature>, String> {
    let mut features = Vec::new();
    let mut point_count = 0usize;
    for element in elements {
        let Some(element_type) = element.get("type").and_then(Value::as_str) else {
            continue;
        };
        let tags = element.get("tags").and_then(Value::as_object);
        let tags = tags.ok_or_else(|| "Overpass 元素缺少 tags 对象。".to_string())?;
        let Some(kind) = feature_kind(element_type, tags) else {
            continue;
        };
        let label = ascii_label(tags);
        let points = if kind == 3 {
            let Some(label) = label.as_ref() else {
                // Places with only non-ASCII labels are omitted instead of drawing mojibake.
                continue;
            };
            let _ = label;
            vec![point_from_element(element)?]
        } else {
            let Some(geometry) = element.get("geometry").and_then(Value::as_array) else {
                continue;
            };
            if geometry.len() < 2 {
                continue;
            }
            geometry
                .iter()
                .map(point_from_element)
                .collect::<Result<Vec<_>, _>>()?
        };
        point_count = point_count
            .checked_add(points.len())
            .ok_or_else(|| "Overpass 点数超出范围。".to_string())?;
        if features.len() >= MAX_FEATURES || point_count > MAX_POINTS {
            return Err(format!(
                "Overpass 数据超出安全上限（最多 {MAX_FEATURES} 个要素、{MAX_POINTS} 个点）。请缩小查询半径。"
            ));
        }
        features.push(GeoFeature {
            kind,
            points,
            label: label.unwrap_or_default(),
        });
    }
    Ok(features)
}

fn feature_kind(element_type: &str, tags: &serde_json::Map<String, Value>) -> Option<u32> {
    match element_type {
        "way" => {
            if tags.contains_key("highway") {
                Some(0)
            } else if tags.contains_key("waterway")
                || tags.get("natural").and_then(Value::as_str) == Some("water")
            {
                Some(1)
            } else if tags.get("natural").and_then(Value::as_str) == Some("coastline") {
                Some(2)
            } else {
                None
            }
        }
        "node" if tags.contains_key("place") => Some(3),
        _ => None,
    }
}

fn ascii_label(tags: &serde_json::Map<String, Value>) -> Option<String> {
    let english = tags.get("name:en").and_then(Value::as_str);
    let fallback = tags.get("name").and_then(Value::as_str);
    [english, fallback]
        .into_iter()
        .flatten()
        .find_map(|candidate| {
            let trimmed = candidate.trim();
            if trimmed.is_empty() || !trimmed.is_ascii() {
                return None;
            }
            let label: String = trimmed.chars().take(256).collect();
            (!label.is_empty()).then_some(label)
        })
}

fn point_from_element(element: &Value) -> Result<GeoPoint, String> {
    let latitude = element
        .get("lat")
        .and_then(Value::as_f64)
        .ok_or_else(|| "Overpass 几何点缺少有效纬度。".to_string())?;
    let longitude = element
        .get("lon")
        .and_then(Value::as_f64)
        .ok_or_else(|| "Overpass 几何点缺少有效经度。".to_string())?;
    if !latitude.is_finite()
        || !longitude.is_finite()
        || !(-90.0..=90.0).contains(&latitude)
        || !(-180.0..=180.0).contains(&longitude)
    {
        return Err("Overpass 几何点包含超出范围的经纬度。".into());
    }
    Ok(GeoPoint {
        latitude,
        longitude,
    })
}

fn should_try_fallback(error: &str) -> bool {
    // A 429 or 406 requires a cooldown on the public endpoint. Do not immediately retry
    // through another service in those cases.
    !error.contains("HTTP 429")
        && !error.contains("HTTP 406")
        && (error.contains("HTTP 5")
            || (error.starts_with("请求 ") && !error.contains("HTTP "))
            || error.contains("读取 Overpass 响应失败"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    fn mock_server(
        status: &'static str,
        body: &'static str,
    ) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let count = stream.read(&mut buffer).unwrap();
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    let headers_end = request
                        .windows(4)
                        .position(|window| window == b"\r\n\r\n")
                        .unwrap()
                        + 4;
                    let header = String::from_utf8_lossy(&request[..headers_end]);
                    let content_length = header
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if request.len() >= headers_end + content_length {
                        break;
                    }
                }
            }
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            String::from_utf8_lossy(&request).into_owned()
        });
        (format!("http://{address}/api/interpreter"), handle)
    }

    #[test]
    fn query_requests_only_supported_geometry_and_no_metadata() {
        let query = build_query(48.8584, 2.2945, 0.2);
        assert!(query.starts_with("[out:json][timeout:20];"));
        assert!(query.contains("around:200,48.85840000,2.29450000"));
        assert!(query.contains("[highway]"));
        assert!(query.contains("[waterway]"));
        assert!(query.contains("[natural=coastline]"));
        assert!(query.contains("[natural=water]"));
        assert!(query.contains("[place]"));
        assert!(query.ends_with("out geom tags;"));
        assert!(!query.contains("meta"));
    }

    #[test]
    fn endpoints_reject_non_http_and_embedded_credentials() {
        assert!(validate_endpoint("file:///tmp/osm").is_err());
        assert!(validate_endpoint("https://alice:secret@example.org/interpreter").is_err());
        assert!(validate_endpoint("https://example.org/interpreter?token=secret").is_err());
        assert!(validate_endpoint("https://example.org/interpreter").is_ok());
    }

    #[test]
    fn parses_only_static_macos_http_proxy_settings() {
        let parsed = parse_static_proxy_settings(
            "HTTPEnable : 1\nHTTPProxy : proxy.local\nHTTPPort : 8080\nHTTPSEnable : 1\nHTTPSProxy : 127.0.0.1\nHTTPSPort : 7897\nProxyAutoConfigEnable : 1\n",
        );
        assert_eq!(parsed.http.as_deref(), Some("http://proxy.local:8080"));
        assert_eq!(parsed.https.as_deref(), Some("http://127.0.0.1:7897"));

        let pac_only = parse_static_proxy_settings(
            "ProxyAutoConfigEnable : 1\nProxyAutoConfigURLString : https://proxy.example/pac\n",
        );
        assert_eq!(pac_only, ProxySettings::default());
    }

    #[test]
    fn query_range_is_bounded_without_silent_clamping() {
        assert!(validate_query(1.0, 2.0, 0.05).is_ok());
        assert!(validate_query(1.0, 2.0, 20.0).is_ok());
        assert!(validate_query(1.0, 2.0, 20.01)
            .unwrap_err()
            .contains("手动缩小"));
        assert!(!should_try_fallback("Overpass 请求被限流（HTTP 429）"));
        assert!(!should_try_fallback("Overpass 请求失败（HTTP 406）"));
        assert!(!should_try_fallback("Overpass 请求失败（HTTP 400）"));
        assert!(should_try_fallback("Overpass 服务暂时不可用（HTTP 502）"));
        assert!(should_try_fallback("请求 overpass-api.de 失败：timed out"));
        assert_eq!(
            retry_after_cooldown("Overpass 请求被限流 Retry-After: 120；稍后重试"),
            Duration::from_secs(120)
        );
        assert_eq!(
            retry_after_cooldown("Overpass 请求被限流 Retry-After: 5；稍后重试"),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn parses_supported_osm_element_kinds_and_ascii_place_names() {
        let json: Value = serde_json::from_str(
            r#"{"elements":[
                {"type":"way","tags":{"highway":"residential","name":"Rue A"},"geometry":[{"lat":1.0,"lon":2.0},{"lat":1.1,"lon":2.1}]},
                {"type":"way","tags":{"waterway":"river"},"geometry":[{"lat":2.0,"lon":3.0},{"lat":2.1,"lon":3.1}]},
                {"type":"way","tags":{"natural":"coastline"},"geometry":[{"lat":3.0,"lon":4.0},{"lat":3.1,"lon":4.1}]},
                {"type":"node","lat":4.0,"lon":5.0,"tags":{"place":"village","name":"村","name:en":"Village"}},
                {"type":"node","lat":5.0,"lon":6.0,"tags":{"place":"village","name":"村"}},
                {"type":"relation","tags":{"waterway":"river"}}
            ]}"#,
        )
        .unwrap();
        let features = parse_elements(json["elements"].as_array().unwrap()).unwrap();
        assert_eq!(features.len(), 4);
        assert_eq!(
            features
                .iter()
                .map(|feature| feature.kind)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(features[0].label, "Rue A");
        assert_eq!(features[3].label, "Village");
        assert_eq!(features[3].points[0].latitude, 4.0);
    }

    #[test]
    fn fetch_uses_http_and_caches_same_query() {
        let body = r#"{"elements":[{"type":"way","tags":{"highway":"residential"},"geometry":[{"lat":1.0,"lon":2.0},{"lat":1.1,"lon":2.1}]}]}"#;
        let (endpoint, server) = mock_server("200 OK", body);
        let client = OsmClient::default();
        let first = client.fetch(&endpoint, 1.0, 2.0, 0.2).unwrap();
        let second = client.fetch(&endpoint, 1.0, 2.0, 0.2).unwrap();
        let request = server.join().unwrap();
        assert!(!first.cache_hit);
        assert!(second.cache_hit);
        assert!(Arc::ptr_eq(&first.features, &second.features));
        assert_eq!(first.features.len(), 1);
        assert!(request.starts_with("POST /api/interpreter "));
        assert!(request
            .to_ascii_lowercase()
            .contains("user-agent: imprint/3.0.5"));
        assert!(request
            .to_ascii_lowercase()
            .contains("content-type: application/x-www-form-urlencoded"));
        assert!(request
            .split_once("\r\n\r\n")
            .is_some_and(|(_, body)| body.starts_with("data=")));
    }

    #[test]
    fn concurrent_identical_queries_share_one_http_request() {
        let body = r#"{"elements":[{"type":"way","tags":{"highway":"residential"},"geometry":[{"lat":1.0,"lon":2.0},{"lat":1.1,"lon":2.1}]}]}"#;
        let (endpoint, server) = mock_server("200 OK", body);
        let client = Arc::new(OsmClient::default());
        let request = |client: Arc<OsmClient>| {
            let endpoint = endpoint.clone();
            thread::spawn(move || client.fetch(&endpoint, 1.0, 2.0, 0.2).unwrap())
        };
        let first = request(Arc::clone(&client));
        let second = request(client);
        let results = [first.join().unwrap(), second.join().unwrap()];
        server.join().unwrap();
        assert_eq!(results.iter().filter(|result| result.cache_hit).count(), 1);
        assert!(Arc::ptr_eq(&results[0].features, &results[1].features));
    }

    #[test]
    fn surfaces_http_limit_server_remark_and_no_data_errors() {
        let (endpoint, server) = mock_server("429 Too Many Requests", "{}");
        let response = request_features(
            &Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap(),
            &validate_endpoint(&endpoint).unwrap(),
            "query",
        )
        .unwrap_err();
        server.join().unwrap();
        assert!(response.contains("HTTP 429"));

        let (endpoint, server) = mock_server("429 Too Many Requests", "{}");
        let client = OsmClient::default();
        assert!(client
            .fetch(&endpoint, 1.0, 2.0, 0.2)
            .unwrap_err()
            .contains("429"));
        let cooldown = client.fetch(&endpoint, 3.0, 4.0, 0.2).unwrap_err();
        server.join().unwrap();
        assert!(cooldown.contains("冷却期"));

        let (endpoint, server) = mock_server("503 Service Unavailable", "{}");
        let client = OsmClient::default();
        let first = client.fetch(&endpoint, 1.0, 2.0, 0.2).unwrap_err();
        let cached = client.fetch(&endpoint, 1.0, 2.0, 0.2).unwrap_err();
        server.join().unwrap();
        assert_eq!(first, cached);
        assert!(cached.contains("HTTP 503"));

        let (endpoint, server) = mock_server("503 Service Unavailable", "{}");
        let response = request_features(
            &Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap(),
            &validate_endpoint(&endpoint).unwrap(),
            "query",
        )
        .unwrap_err();
        server.join().unwrap();
        assert!(response.contains("HTTP 503"));

        let (endpoint, server) = mock_server("200 OK", r#"{"remark":"runtime error: timed out"}"#);
        let response = request_features(
            &Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap(),
            &validate_endpoint(&endpoint).unwrap(),
            "query",
        )
        .unwrap_err();
        server.join().unwrap();
        assert!(response.contains("timed out"));

        assert!(parse_elements(&[]).unwrap().is_empty());

        let (endpoint, server) = mock_server("200 OK", r#"{"elements":[]}"#);
        let error = OsmClient::default()
            .fetch(&endpoint, 1.0, 2.0, 0.2)
            .unwrap_err();
        server.join().unwrap();
        assert!(error.contains("没有可绘制"));
    }
}
