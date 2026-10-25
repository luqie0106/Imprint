#include "imprint_geo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace imprint::geo {
namespace {
constexpr double kEarthRadiusM = 6371008.8;
constexpr double kPi = 3.1415926535897932384626433832795;
inline double radians(double d) noexcept { return d * (kPi / 180.0); }
inline double degrees(double r) noexcept { return r * (180.0 / kPi); }
inline double clamp01(double x) noexcept { return std::max(0.0, std::min(1.0, x)); }

bool valid_coordinate(double lat, double lon) noexcept {
    return std::isfinite(lat) && std::isfinite(lon) && lat >= -90.0 && lat <= 90.0 && lon >= -180.0 && lon <= 180.0;
}
} // namespace

LocalAzimuthalEquidistant::LocalAzimuthalEquidistant(double latitude_deg, double longitude_deg)
    : latitude_rad_(radians(latitude_deg)), longitude_rad_(radians(longitude_deg)),
      valid_(valid_coordinate(latitude_deg, longitude_deg)) {}

bool LocalAzimuthalEquidistant::valid() const noexcept { return valid_; }

bool LocalAzimuthalEquidistant::forward(double latitude_deg, double longitude_deg, MetricPoint *out) const noexcept {
    if (!valid_ || out == nullptr || !valid_coordinate(latitude_deg, longitude_deg)) return false;
    const double phi = radians(latitude_deg);
    const double lambda = radians(longitude_deg);
    const double dlambda = std::remainder(lambda - longitude_rad_, 2.0 * kPi);
    const double sin0 = std::sin(latitude_rad_);
    const double cos0 = std::cos(latitude_rad_);
    const double sinp = std::sin(phi);
    const double cosp = std::cos(phi);
    const double cos_c = std::max(-1.0, std::min(1.0, sin0 * sinp + cos0 * cosp * std::cos(dlambda)));
    const double c = std::acos(cos_c);
    if (c < 1e-12) {
        out->east_m = 0.0;
        out->north_m = 0.0;
        return true;
    }
    if (kPi - c < 1e-8) return false; // antipodal point has no unique azimuth
    const double sin_c = std::sin(c);
    const double k = c / sin_c;
    out->east_m = kEarthRadiusM * k * cosp * std::sin(dlambda);
    out->north_m = kEarthRadiusM * k * (cos0 * sinp - sin0 * cosp * std::cos(dlambda));
    return std::isfinite(out->east_m) && std::isfinite(out->north_m);
}

bool LocalAzimuthalEquidistant::inverse(const MetricPoint &point, double *latitude_deg, double *longitude_deg) const noexcept {
    if (!valid_ || latitude_deg == nullptr || longitude_deg == nullptr ||
        !std::isfinite(point.east_m) || !std::isfinite(point.north_m)) return false;
    const double rho = std::hypot(point.east_m, point.north_m);
    if (rho < 1e-9) {
        *latitude_deg = degrees(latitude_rad_);
        *longitude_deg = degrees(longitude_rad_);
        return true;
    }
    const double c = rho / kEarthRadiusM;
    if (c >= kPi) return false;
    const double sinc = std::sin(c);
    const double cosc = std::cos(c);
    const double sinc0 = std::sin(latitude_rad_);
    const double cosc0 = std::cos(latitude_rad_);
    const double phi = std::asin(std::max(-1.0, std::min(1.0,
        cosc * sinc0 + point.north_m * sinc * cosc0 / rho)));
    const double lambda = longitude_rad_ + std::atan2(
        point.east_m * sinc,
        rho * cosc0 * cosc - point.north_m * sinc0 * sinc);
    *latitude_deg = degrees(phi);
    *longitude_deg = std::remainder(degrees(lambda), 360.0);
    if (*longitude_deg < -180.0) *longitude_deg += 360.0;
    if (*longitude_deg > 180.0) *longitude_deg -= 360.0;
    return std::isfinite(*latitude_deg) && std::isfinite(*longitude_deg);
}

bool MockGeoDataProvider::load(const GeoQuery &query, GeoDataset *out, std::string *error) const {
    if (out == nullptr || !valid_coordinate(query.latitude, query.longitude) ||
        !std::isfinite(query.radius_m) || query.radius_m <= 0.0) {
        if (error) *error = "Invalid query for the mock map provider.";
        return false;
    }
    struct Cache { GeoQuery query{}; GeoDataset dataset; bool valid=false; };
    static thread_local Cache cache;
    if (cache.valid && cache.query.latitude==query.latitude && cache.query.longitude==query.longitude &&
        cache.query.radius_m==query.radius_m) {
        *out=cache.dataset;
        if (error) error->clear();
        return true;
    }
    LocalAzimuthalEquidistant projection(query.latitude, query.longitude);
    const double r = query.radius_m;
    const auto make_feature = [&](FeatureKind kind, const char *label,
                                  std::initializer_list<MetricPoint> points, GeoFeature *feature) -> bool {
        feature->kind = kind;
        feature->label = label;
        feature->points_deg.clear();
        for (const MetricPoint &p : points) {
            double lat = 0.0, lon = 0.0;
            if (!projection.inverse(p, &lat, &lon)) return false;
            feature->points_deg.emplace_back(lat, lon);
        }
        return true;
    };
    GeoDataset result;
    result.mock_data = true;
    result.features.resize(4);
    if (!make_feature(FeatureKind::Road, "MOCK ROAD", {
        {-0.95*r,-0.32*r}, {-0.68*r,-0.10*r}, {-0.35*r,-0.17*r}, {-0.06*r,0.04*r},
        {0.26*r,0.12*r}, {0.57*r,0.04*r}, {0.93*r,0.34*r}}, &result.features[0]) ||
        !make_feature(FeatureKind::Road, "MOCK ROAD", {
        {-0.42*r,-0.96*r}, {-0.25*r,-0.57*r}, {-0.18*r,-0.25*r}, {0.04*r,0.07*r},
        {0.25*r,0.39*r}, {0.31*r,0.70*r}, {0.48*r,0.96*r}}, &result.features[1]) ||
        !make_feature(FeatureKind::Waterway, "MOCK WATER", {
        {-0.98*r,0.62*r}, {-0.65*r,0.45*r}, {-0.39*r,0.47*r}, {-0.16*r,0.30*r},
        {0.18*r,0.27*r}, {0.44*r,0.06*r}, {0.72*r,-0.02*r}, {0.98*r,-0.22*r}}, &result.features[2]) ||
        !make_feature(FeatureKind::Coastline, "MOCK COAST", {
        {-1.0*r,0.82*r}, {-0.66*r,0.72*r}, {-0.34*r,0.76*r}, {-0.02*r,0.68*r},
        {0.31*r,0.72*r}, {0.62*r,0.66*r}, {1.0*r,0.74*r}}, &result.features[3])) {
        if (error) *error = "Could not project mock map geometry.";
        return false;
    }
    result.features.resize(9);
    if (!make_feature(FeatureKind::Road, "MOCK ROAD", {
        {-0.92*r,-0.64*r}, {-0.62*r,-0.57*r}, {-0.30*r,-0.60*r}, {0.03*r,-0.55*r},
        {0.36*r,-0.60*r}, {0.66*r,-0.54*r}, {0.94*r,-0.62*r}}, &result.features[4]) ||
        !make_feature(FeatureKind::Road, "MOCK ROAD", {
        {-0.92*r,0.55*r}, {-0.61*r,0.61*r}, {-0.30*r,0.52*r}, {0.02*r,0.59*r},
        {0.36*r,0.50*r}, {0.65*r,0.56*r}, {0.94*r,0.50*r}}, &result.features[5]) ||
        !make_feature(FeatureKind::Road, "MOCK ROAD", {
        {-0.68*r,-0.92*r}, {-0.61*r,-0.61*r}, {-0.65*r,-0.28*r}, {-0.59*r,0.02*r},
        {-0.63*r,0.34*r}, {-0.58*r,0.64*r}, {-0.67*r,0.94*r}}, &result.features[6]) ||
        !make_feature(FeatureKind::Waterway, "MOCK WATER", {
        {-0.94*r,-0.83*r}, {-0.72*r,-0.69*r}, {-0.53*r,-0.52*r}, {-0.38*r,-0.31*r}},
        &result.features[7]) ||
        !make_feature(FeatureKind::Coastline, "MOCK COAST", {
        {-0.96*r,0.95*r}, {-0.76*r,0.70*r}, {-0.64*r,0.47*r}, {-0.43*r,0.20*r},
        {-0.28*r,-0.02*r}}, &result.features[8])) {
        if (error) *error = "Could not project additional mock map geometry.";
        return false;
    }
    cache.query=query; cache.dataset=result; cache.valid=true;
    *out = std::move(result);
    if (error) error->clear();
    return true;
}

bool OverpassGeoDataProvider::load(const GeoQuery &, GeoDataset *, std::string *error) const {
    if (error) *error = "Overpass provider is unsupported in this native core; no map request was made.";
    return false;
}

} // namespace imprint::geo

namespace {
using imprint::geo::FeatureKind;
using imprint::geo::GeoDataset;
using imprint::geo::GeoFeature;
using imprint::geo::GeoQuery;
using imprint::geo::LocalAzimuthalEquidistant;
using imprint::geo::MetricPoint;
using imprint::geo::clamp01;
using imprint::geo::kPi;
using imprint::geo::radians;

constexpr uint64_t kMaximumPixels = 100000000ULL;

void write_error(char *dst, uint32_t capacity, const char *message) noexcept {
    if (dst == nullptr || capacity == 0) return;
    const size_t n = static_cast<size_t>(capacity);
    std::snprintf(dst, n, "%s", message ? message : "Unknown native geo error.");
}

bool checked_pixel_count(uint32_t width, uint32_t height, size_t *count) noexcept {
    if (width == 0 || height == 0) return false;
    const uint64_t n = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    if (n > kMaximumPixels || n > std::numeric_limits<size_t>::max() / 4) return false;
    *count = static_cast<size_t>(n);
    return true;
}

bool valid_config(const GeoConfig &c, const char **why) noexcept {
    if (c.enabled == 0) return true;
    if (c.has_location == 0) { *why = "GPS location is missing (has_location is false)."; return false; }
    if (!std::isfinite(c.latitude) || !std::isfinite(c.longitude)) {
        *why = "GPS latitude/longitude contains NaN or infinity."; return false;
    }
    if (c.latitude < -90.0 || c.latitude > 90.0 || c.longitude < -180.0 || c.longitude > 180.0) {
        *why = "GPS latitude/longitude is invalid."; return false;
    }
    const double values[] = {c.opacity,c.radius_km,c.offset_x,c.offset_y,c.scale,c.rotation,
                             c.line_width,c.feather,c.coverage,c.manual_horizon};
    for (double v : values) if (!std::isfinite(v)) { *why = "Geo configuration contains NaN or infinity."; return false; }
    if (c.opacity < 0.0 || c.opacity > 1.0) { *why = "opacity must be in [0,1]."; return false; }
    if (c.radius_km < 0.05 || c.radius_km > 1000.0) { *why = "radius_km must be in [0.05,1000]."; return false; }
    if (c.offset_x < -1.0 || c.offset_x > 1.0 || c.offset_y < -1.0 || c.offset_y > 1.0) {
        *why = "offset_x and offset_y must be in [-1,1]."; return false;
    }
    if (c.scale < 0.1 || c.scale > 4.0) { *why = "scale must be in [0.1,4]."; return false; }
    if (c.line_width < 0.25 || c.line_width > 40.0) { *why = "line_width must be in [0.25,40]."; return false; }
    if (c.feather < 0.0 || c.feather > 0.3) { *why = "feather must be in [0,0.3]."; return false; }
    if (c.coverage < 0.05 || c.coverage > 1.0) { *why = "coverage must be in [0.05,1]."; return false; }
    if (c.manual_horizon < 0.0 || c.manual_horizon > 1.0) { *why = "manual_horizon must be in [0,1]."; return false; }
    if (c.style < 0 || c.style > 3) { *why = "Unknown geographic overlay style."; return false; }
    if (c.provider < 0 || c.provider > 1) { *why = "Unknown geographic data provider."; return false; }
    return true;
}

using SkyRegion = imprint::geo::SkyMask;

inline int luma(int r, int g, int b) noexcept { return (77*r + 150*g + 29*b) >> 8; }

bool sky_candidate(const uint8_t *rgb, uint32_t x, uint32_t y, uint32_t width,
                   uint8_t ref_r, uint8_t ref_g, uint8_t ref_b) noexcept {
    const size_t i = (static_cast<size_t>(y) * width + x) * 3;
    const int r=rgb[i], g=rgb[i+1], b=rgb[i+2];
    const int maxc=std::max({r,g,b}), minc=std::min({r,g,b});
    const int bright=luma(r,g,b);
    const int dr=r-ref_r, dg=g-ref_g, db=b-ref_b;
    const int color_distance=std::abs(dr)+std::abs(dg)+std::abs(db);
    const bool blue_sky = b >= r + 7 && b >= g - 12 && b > 95 && bright > 90;
    const bool blue_reference = ref_b >= ref_r + 7;
    const bool low_luma_blue_sky = blue_reference && color_distance <= 78 && bright > 24 &&
                                   (b >= r + 4 || color_distance <= 24);
    /* Neutral clouds are admitted only when they closely match a blue sky reference. */
    const bool cloud_or_haze = blue_reference && (maxc-minc <= 32) && bright > 190 && color_distance <= 82;
    if (!(blue_sky || low_luma_blue_sky || cloud_or_haze) || color_distance > 145) return false;
    int edge=0;
    if (x > 0) {
        const size_t left=i-3;
        edge+=std::abs(r-rgb[left])+std::abs(g-rgb[left+1])+std::abs(b-rgb[left+2]);
    }
    if (y > 0) {
        const size_t above=i-static_cast<size_t>(width)*3;
        edge+=std::abs(r-rgb[above])+std::abs(g-rgb[above+1])+std::abs(b-rgb[above+2]);
    }
    /* Strong local boundaries stop the mask at buildings, trees, and shore edges. */
    const int edge_limit=bright<90?72:105;
    return edge < edge_limit || color_distance < 24;
}

SkyRegion find_top_connected_sky(const uint8_t *rgb, uint32_t width, uint32_t height) {
    const size_t count = static_cast<size_t>(width) * height;
    SkyRegion sky;
    sky.width=width; sky.height=height;
    sky.pixels.assign(count, 0);
    sky.bottom_by_column.assign(width, -1);
    sky.min_x = static_cast<int32_t>(width);
    sky.min_y = static_cast<int32_t>(height);
    if (count == 0) return sky;

    const uint32_t sample_count = std::min<uint32_t>(width, std::max<uint32_t>(1, width / 20));
    std::array<int,3> sums{0,0,0};
    for (uint32_t x=0; x<sample_count; ++x) {
        const uint32_t sx = sample_count <= 1 ? 0 :
            (width == sample_count ? x : static_cast<uint32_t>((static_cast<uint64_t>(x)*(width-1))/(sample_count-1)));
        const size_t i=static_cast<size_t>(sx)*3;
        sums[0]+=rgb[i]; sums[1]+=rgb[i+1]; sums[2]+=rgb[i+2];
    }
    const uint8_t ref_r=static_cast<uint8_t>(sums[0]/static_cast<int>(sample_count));
    const uint8_t ref_g=static_cast<uint8_t>(sums[1]/static_cast<int>(sample_count));
    const uint8_t ref_b=static_cast<uint8_t>(sums[2]/static_cast<int>(sample_count));
    std::vector<uint8_t> candidate(count,0);
    for (uint32_t y=0; y<height; ++y) {
        for (uint32_t x=0; x<width; ++x) {
            candidate[static_cast<size_t>(y)*width+x] =
                sky_candidate(rgb,x,y,width,ref_r,ref_g,ref_b) ? 1 : 0;
        }
    }

    struct Span { int32_t y, x0, x1; };
    std::vector<Span> stack;
    const auto fill_span = [&](int32_t x, int32_t y, auto &&self) -> void {
        (void)self;
        int32_t left=x, right=x;
        const size_t row=static_cast<size_t>(y)*width;
        while (left>0 && candidate[row+static_cast<size_t>(left-1)] && !sky.pixels[row+static_cast<size_t>(left-1)]) --left;
        while (right+1<static_cast<int32_t>(width) && candidate[row+static_cast<size_t>(right+1)] && !sky.pixels[row+static_cast<size_t>(right+1)]) ++right;
        for (int32_t sx=left; sx<=right; ++sx) sky.pixels[row+static_cast<size_t>(sx)]=1;
        stack.push_back({y,left,right});
    };
    if (height > 0) {
        uint32_t x=0;
        while (x<width) {
            const size_t i=x;
            if (!candidate[i] || sky.pixels[i]) { ++x; continue; }
            fill_span(static_cast<int32_t>(x),0,fill_span);
            while (x<width && sky.pixels[x]) ++x;
        }
    }
    while (!stack.empty()) {
        const Span span=stack.back();
        stack.pop_back();
        for (int dy : {-1,1}) {
            const int32_t ny=span.y+dy;
            if (ny<0 || ny>=static_cast<int32_t>(height)) continue;
            const size_t row=static_cast<size_t>(ny)*width;
            int32_t x=std::max<int32_t>(0,span.x0-1);
            const int32_t end=std::min<int32_t>(static_cast<int32_t>(width)-1,span.x1+1);
            while (x<=end) {
                const size_t i=row+static_cast<size_t>(x);
                if (!candidate[i] || sky.pixels[i]) { ++x; continue; }
                fill_span(x,ny,fill_span);
                while (x<=end && sky.pixels[row+static_cast<size_t>(x)]) ++x;
            }
        }
    }
    for (uint32_t y=0; y<height; ++y) {
        const size_t row=static_cast<size_t>(y)*width;
        for (uint32_t x=0; x<width; ++x) {
            if (!sky.pixels[row+x]) continue;
            ++sky.area;
            sky.min_x=std::min(sky.min_x,static_cast<int32_t>(x));
            sky.max_x=std::max(sky.max_x,static_cast<int32_t>(x));
            sky.min_y=std::min(sky.min_y,static_cast<int32_t>(y));
            sky.max_y=std::max(sky.max_y,static_cast<int32_t>(y));
            sky.bottom_by_column[x]=static_cast<int32_t>(y);
        }
    }
    if (sky.area==0) { sky.min_x=0; sky.min_y=0; sky.max_x=-1; sky.max_y=-1; }
    /* 3/4 chamfer distance transform gives a two-dimensional subject-edge fade. */
    sky.distance_inside.assign(count,0);
    constexpr uint16_t kInf=std::numeric_limits<uint16_t>::max();
    for (size_t i=0;i<count;++i) if (sky.pixels[i]) sky.distance_inside[i]=kInf;
    const auto relax=[&](size_t i,size_t j,uint16_t weight) {
        const uint32_t candidate=static_cast<uint32_t>(sky.distance_inside[j])+weight;
        if (candidate<sky.distance_inside[i]) sky.distance_inside[i]=static_cast<uint16_t>(candidate);
    };
    for (uint32_t y=0;y<height;++y) for (uint32_t x=0;x<width;++x) {
        const size_t i=static_cast<size_t>(y)*width+x;
        if (!sky.pixels[i]) continue;
        if (y>0) relax(i,i-width,3);
        if (x>0) relax(i,i-1,3);
        if (y>0&&x>0) relax(i,i-width-1,4);
        if (y>0&&x+1<width) relax(i,i-width+1,4);
    }
    for (uint32_t yy=height;yy>0;--yy) {
        const uint32_t y=yy-1;
        for (uint32_t xx=width;xx>0;--xx) {
            const uint32_t x=xx-1;
            const size_t i=static_cast<size_t>(y)*width+x;
            if (!sky.pixels[i]) continue;
            if (y+1<height) relax(i,i+width,3);
            if (x+1<width) relax(i,i+1,3);
            if (y+1<height&&x+1<width) relax(i,i+width+1,4);
            if (y+1<height&&x>0) relax(i,i+width-1,4);
        }
    }
    return sky;
}

struct Color { uint8_t r,g,b; };
Color rgb_color(uint32_t rgb) noexcept {
    return {static_cast<uint8_t>((rgb>>16)&255),static_cast<uint8_t>((rgb>>8)&255),static_cast<uint8_t>(rgb&255)};
}

class Canvas {
public:
    Canvas(uint8_t *rgba, uint32_t w, uint32_t h, double cx, double cy, double radius, double opacity)
        : rgba_(rgba), width_(w), height_(h), cx_(cx), cy_(cy), radius_(radius), opacity_(opacity) {}

    void pixel(int x, int y, Color c, double alpha) noexcept {
        if (x<0 || y<0 || x>=static_cast<int>(width_) || y>=static_cast<int>(height_) || alpha<=0.0) return;
        const double dx=(x+0.5)-cx_, dy=(y+0.5)-cy_;
        const double edge=radius_+0.5-std::hypot(dx,dy);
        if (edge<=0.0) return;
        const double clip=std::min(1.0,edge);
        const double sa=clamp01(alpha*opacity_*clip);
        if (sa<=0.0) return;
        const size_t i=(static_cast<size_t>(y)*width_+static_cast<size_t>(x))*4;
        const double da=rgba_[i+3]/255.0;
        const double oa=sa+da*(1.0-sa);
        if (oa<=0.0) return;
        rgba_[i]=static_cast<uint8_t>(std::lround((c.r*sa+rgba_[i]*da*(1.0-sa))/oa));
        rgba_[i+1]=static_cast<uint8_t>(std::lround((c.g*sa+rgba_[i+1]*da*(1.0-sa))/oa));
        rgba_[i+2]=static_cast<uint8_t>(std::lround((c.b*sa+rgba_[i+2]*da*(1.0-sa))/oa));
        rgba_[i+3]=static_cast<uint8_t>(std::lround(oa*255.0));
    }

    void stamp(double x, double y, double radius, Color c, double alpha=1.0) noexcept {
        const int r=std::max(0,static_cast<int>(std::ceil(radius+1.0)));
        const int x0=static_cast<int>(std::floor(x)), y0=static_cast<int>(std::floor(y));
        for (int py=y0-r; py<=y0+r; ++py) for (int px=x0-r; px<=x0+r; ++px) {
            const double d=std::hypot((px+0.5)-x,(py+0.5)-y);
            const double coverage=clamp01(radius+0.5-d);
            if (coverage>0.0) pixel(px,py,c,alpha*coverage);
        }
    }

    void line(double x0,double y0,double x1,double y1,double width,Color c,double alpha=1.0) noexcept {
        const double dx=x1-x0, dy=y1-y0;
        const double length=std::hypot(dx,dy);
        const int steps=std::max(1,static_cast<int>(std::ceil(length*1.4)));
        const double radius=std::max(0.0,width*0.5);
        for (int i=0; i<=steps; ++i) {
            const double t=static_cast<double>(i)/steps;
            stamp(x0+dx*t,y0+dy*t,radius,c,alpha);
        }
    }

    void circle(double x,double y,double radius,double width,Color c,double alpha=1.0) noexcept {
        const double circumference=2.0*kPi*std::max(0.1,radius);
        const int steps=std::max(16,static_cast<int>(std::ceil(circumference*1.2)));
        for (int i=0;i<steps;++i) {
            const double a=2.0*kPi*i/steps;
            stamp(x+radius*std::cos(a),y+radius*std::sin(a),std::max(0.0,width*0.5),c,alpha);
        }
    }

    void fill_circle(double x,double y,double radius,Color c,double alpha) noexcept {
        const int x0=std::max(0,static_cast<int>(std::floor(x-radius-1.0)));
        const int x1=std::min(static_cast<int>(width_)-1,static_cast<int>(std::ceil(x+radius+1.0)));
        const int y0=std::max(0,static_cast<int>(std::floor(y-radius-1.0)));
        const int y1=std::min(static_cast<int>(height_)-1,static_cast<int>(std::ceil(y+radius+1.0)));
        for (int py=y0;py<=y1;++py) for (int px=x0;px<=x1;++px) {
            const double d=std::hypot((px+0.5)-x,(py+0.5)-y);
            const double coverage=clamp01(radius+0.5-d);
            if (coverage>0.0) pixel(px,py,c,alpha*coverage);
        }
    }

    void text(double x,double y,const char *s,int scale,Color c,double alpha=1.0) noexcept;
private:
    uint8_t *rgba_;
    uint32_t width_,height_;
    double cx_,cy_,radius_,opacity_;
};

std::array<uint8_t,7> glyph(char ch) noexcept {
    if (ch>='a'&&ch<='z') ch=static_cast<char>(ch-'a'+'A');
    switch(ch) {
#define G(ch_,a,b,c,d,e,f,g) case ch_: return {{a,b,c,d,e,f,g}}
        G('A',14,17,17,31,17,17,17); G('B',30,17,17,30,17,17,30); G('C',14,17,16,16,16,17,14);
        G('D',30,17,17,17,17,17,30); G('E',31,16,16,30,16,16,31); G('F',31,16,16,30,16,16,16);
        G('G',14,17,16,23,17,17,15); G('H',17,17,17,31,17,17,17); G('I',14,4,4,4,4,4,14);
        G('J',7,2,2,2,18,18,12); G('K',17,18,20,24,20,18,17); G('L',16,16,16,16,16,16,31);
        G('M',17,27,21,21,17,17,17); G('N',17,25,21,19,17,17,17); G('O',14,17,17,17,17,17,14);
        G('P',30,17,17,30,16,16,16); G('Q',14,17,17,17,21,18,13); G('R',30,17,17,30,20,18,17);
        G('S',15,16,16,14,1,1,30); G('T',31,4,4,4,4,4,4); G('U',17,17,17,17,17,17,14);
        G('V',17,17,17,17,17,10,4); G('W',17,17,17,21,21,21,10); G('X',17,17,10,4,10,17,17);
        G('Y',17,17,10,4,4,4,4); G('Z',31,1,2,4,8,16,31);
        G('0',14,17,19,21,25,17,14); G('1',4,12,4,4,4,4,14); G('2',14,17,1,2,4,8,31);
        G('3',30,1,1,14,1,1,30); G('4',2,6,10,18,31,2,2); G('5',31,16,16,30,1,1,30);
        G('6',14,16,16,30,17,17,14); G('7',31,1,2,4,8,8,8); G('8',14,17,17,14,17,17,14);
        G('9',14,17,17,15,1,1,14); G('-',0,0,0,31,0,0,0); G('.',0,0,0,0,0,12,12);
        G(':',0,12,12,0,12,12,0); G('/',1,2,2,4,8,8,16); G('+',0,4,4,31,4,4,0);
        G(',',0,0,0,0,0,4,8); G(' ',0,0,0,0,0,0,0); G('_',0,0,0,0,0,0,31);
#undef G
        default: return {{0,0,0,0,0,0,0}};
    }
}

void Canvas::text(double x,double y,const char *s,int scale,Color c,double alpha) noexcept {
    if (!s || scale<1) return;
    const double start=x;
    for (;*s;++s,x+=6.0*scale) {
        const auto rows=glyph(*s);
        for (int row=0;row<7;++row) for (int col=0;col<5;++col) {
            if ((rows[row] & (1u << (4-col)))==0) continue;
            for (int py=0;py<scale;++py) for (int px=0;px<scale;++px)
                pixel(static_cast<int>(start+(x-start)+col*scale+px),static_cast<int>(y+row*scale+py),c,alpha);
        }
    }
}

struct Palette {
    Color bg, grid, road, water, coast, accent, text, marker;
};
Palette make_palette(bool dark, uint32_t custom) noexcept {
    Palette p;
    if (dark) p={{13,24,39},{112,132,145},{168,188,198},{132,165,180},{205,215,221},{168,188,198},{226,234,238},{235,68,72}};
    else p={{246,242,226},{130,145,151},{68,91,103},{82,124,143},{114,136,146},{68,91,103},{52,69,78},{207,54,56}};
    if (custom!=0) {
        p.accent=rgb_color(custom);
        p.road=p.water=p.coast=p.accent;
        p.text=p.accent;
    }
    return p;
}

struct TextItem {
    double x=0.0, y=0.0;
    std::string text;
    int scale=1;
    Color color{255,255,255};
    double alpha=1.0;
};

bool safe_text_box(const TextItem &item, uint32_t width, uint32_t height,
                   const SkyRegion &sky, bool check_sky, bool manual, double manual_y,
                   double feather_px) noexcept {
    const int x0=static_cast<int>(std::floor(item.x));
    const int y0=static_cast<int>(std::floor(item.y));
    const int x1=x0+static_cast<int>(item.text.size())*6*item.scale;
    const int y1=y0+7*item.scale;
    if (x0<0 || y0<0 || x1>static_cast<int>(width) || y1>static_cast<int>(height)) return false;
    if (manual && y1+feather_px>manual_y) return false;
    if (!check_sky) return true;
    const double safe_distance=feather_px+item.scale;
    for (int y=y0;y<y1;++y) for (int x=x0;x<x1;++x) {
        const size_t i=static_cast<size_t>(y)*width+static_cast<size_t>(x);
        if (!sky.pixels[i] || sky.distance_inside[i]/3.0<safe_distance) return false;
    }
    return true;
}

bool render_impl(const uint8_t *rgb,uint32_t width,uint32_t height,const GeoConfig &c,uint8_t *out,
                 const imprint::geo::IGeoDataProvider &data_provider,
                 const imprint::geo::ISkyMaskProvider &sky_provider,const char **why) {
    size_t count=0;
    if (!checked_pixel_count(width,height,&count)) { *why="Image dimensions are zero or exceed the native pixel limit."; return false; }
    if (rgb==nullptr || out==nullptr) { *why="RGB input and RGBA output buffers are required."; return false; }
    std::memset(out,0,count*4);
    if (c.enabled==0) return true;
    if (!valid_config(c,why)) return false;
    if (c.style==static_cast<int32_t>(imprint::geo::Style::Terrain)) {
        *why="Terrain style is unsupported because this core has no verified DEM source."; return false;
    }
    if (c.provider==static_cast<int32_t>(imprint::geo::ProviderKind::Overpass)) {
        *why="Overpass provider is unsupported in this native core; no network request was made."; return false;
    }

    SkyRegion sky;
    if (c.auto_sky_fade!=0) {
        std::string sky_error;
        if (!sky_provider.build(rgb,width,height,&sky,&sky_error)) {
            *why="Sky mask provider failed."; return false;
        }
        if (sky.area==0) return true;
    }

    double cx=static_cast<double>(width)*0.5 + c.offset_x*static_cast<double>(width)*0.25;
    double cy=static_cast<double>(height)*0.5 + c.offset_y*static_cast<double>(height)*0.25;
    double diameter=static_cast<double>(std::min(width,height))*c.coverage*c.scale;
    if (c.auto_sky_fade!=0 && c.adaptive!=0) {
        const double sky_w=std::max(1,sky.max_x-sky.min_x+1);
        const double sky_h=std::max(1,sky.max_y-sky.min_y+1);
        const double available=std::min(sky_w,sky_h);
        diameter=available*c.coverage*c.scale;
        cx=(sky.min_x+sky.max_x+1.0)*0.5 + c.offset_x*std::max(0.0,(sky_w-diameter)*0.5);
        cy=(sky.min_y+sky.max_y+1.0)*0.5 + c.offset_y*std::max(0.0,(sky_h-diameter)*0.5);
    }
    const double panel_radius=std::max(1.0,diameter*0.5);
    const double meters_per_pixel=(c.radius_km*1000.0)/panel_radius;
    const double px_per_m=1.0/meters_per_pixel;
    const double angle=radians(std::remainder(c.rotation,360.0));
    const double cs=std::cos(angle), sn=std::sin(angle);
    /* Accumulate drawing coverage at full layer strength; config opacity is applied once below. */
    Canvas canvas(out,width,height,cx,cy,panel_radius,1.0);

    uint64_t luma_sum=0;
    const uint32_t sample_step=std::max<uint32_t>(1,std::min(width,height)/64);
    size_t luma_n=0;
    for (uint32_t y=0;y<height;y+=sample_step) for (uint32_t x=0;x<width;x+=sample_step) {
        const size_t i=(static_cast<size_t>(y)*width+x)*3;
        luma_sum+=static_cast<uint64_t>(luma(rgb[i],rgb[i+1],rgb[i+2])); ++luma_n;
    }
    const bool dark=(luma_n==0 || luma_sum/luma_n<112);
    Palette palette=make_palette(dark,c.color_rgb);
    std::vector<TextItem> pending_labels;

    GeoQuery query{c.latitude,c.longitude,c.radius_km*1000.0};
    GeoDataset dataset;
    std::string provider_error;
    if (!data_provider.load(query,&dataset,&provider_error)) {
        *why="Geographic data provider failed or is unsupported."; return false;
    }
    imprint::geo::Style style=static_cast<imprint::geo::Style>(c.style);
    if (style==imprint::geo::Style::Auto) {
        style=imprint::geo::Style::Urban;
        for (const auto &feature:dataset.features) if (feature.kind==FeatureKind::Coastline) { style=imprint::geo::Style::Coastal; break; }
    }
    const LocalAzimuthalEquidistant projection(c.latitude,c.longitude);
    const auto map_point=[&](double lat,double lon,double *x,double *y)->bool {
        MetricPoint p;
        if (!projection.forward(lat,lon,&p)) return false;
        const double east=p.east_m*cs-p.north_m*sn;
        const double north=p.east_m*sn+p.north_m*cs;
        *x=cx+east*px_per_m;
        *y=cy-north*px_per_m;
        return std::isfinite(*x)&&std::isfinite(*y);
    };
    const double detail_scale=std::min(width,height)/1000.0;
    const double lw=c.line_width*detail_scale;
    for (const GeoFeature &feature:dataset.features) {
        if (feature.kind==FeatureKind::Coastline && style!=imprint::geo::Style::Coastal) continue;
        Color color=feature.kind==FeatureKind::Road?palette.road:
                    feature.kind==FeatureKind::Waterway?palette.water:palette.coast;
        const double feature_width=feature.kind==FeatureKind::Road?lw:
                                   feature.kind==FeatureKind::Coastline?lw*1.35:lw*1.15;
        double last_x=0,last_y=0;
        bool have_last=false;
        for (const auto &point:feature.points_deg) {
            double x=0,y=0;
            if (!map_point(point.first,point.second,&x,&y)) { have_last=false; continue; }
            if (have_last) canvas.line(last_x,last_y,x,y,feature_width,color,0.95);
            last_x=x; last_y=y; have_last=true;
        }
        if (c.labels!=0 && feature.points_deg.size()>2) {
            const size_t pi=feature.points_deg.size()/2;
            double x=0,y=0;
            if (map_point(feature.points_deg[pi].first,feature.points_deg[pi].second,&x,&y)) {
                const int ts=std::max(1,std::min(4,static_cast<int>(std::lround(panel_radius/350.0))));
                pending_labels.push_back({x+4,y-9.0*ts,feature.label,ts,palette.text,0.96});
            }
        }
    }

    if (c.labels!=0) {
        char coords[96];
        std::snprintf(coords,sizeof(coords),"%c %.4f %c %.4f",
                      c.latitude<0.0?'S':'N',std::fabs(c.latitude),
                      c.longitude<0.0?'W':'E',std::fabs(c.longitude));
        const int ts=std::max(1,std::min(4,static_cast<int>(std::lround(panel_radius/400.0))));
        pending_labels.push_back({cx-panel_radius*0.70,cy-panel_radius*0.76,coords,ts,palette.text,0.98});
        pending_labels.push_back({cx-panel_radius*0.70,cy-panel_radius*0.76+10.0*ts,
                                  "MOCK DATA - EXAMPLE ONLY",ts,palette.marker,1.0});
    } else {
        const int ts=std::max(1,std::min(3,static_cast<int>(std::lround(panel_radius/500.0))));
        pending_labels.push_back({cx-panel_radius*0.35,cy-panel_radius*0.78,"MOCK MAP",ts,palette.marker,0.95});
    }

    if (c.photo_marker!=0) {
        const double m=std::max(1.5*detail_scale,lw*3.0);
        canvas.fill_circle(cx,cy,m*0.34,palette.marker,1.0);
        canvas.circle(cx,cy,m,1.8*detail_scale,palette.marker,1.0);
        canvas.line(cx-m*1.8,cy,cx+m*1.8,cy,1.2*detail_scale,palette.marker,0.95);
        canvas.line(cx,cy-m*1.8,cx,cy+m*1.8,1.2*detail_scale,palette.marker,0.95);
        canvas.circle(cx,cy,1.8*detail_scale,1.0*detail_scale,palette.text,1.0);
    }
    if (c.north_arrow!=0) {
        const double ax=cx+panel_radius*0.70, ay=cy-panel_radius*0.55;
        const double len=std::max(8.0,panel_radius*0.12);
        const double nx=-sn, ny=-cs; // true north after map rotation
        const double px=cs, py=-sn;
        const double tx=ax+nx*len, ty=ay+ny*len;
        canvas.line(ax-nx*len,ay-ny*len,tx,ty,2.0*detail_scale,palette.text,0.95);
        canvas.line(tx,ty,tx-nx*len*0.32+px*len*0.42,ty-ny*len*0.32+py*len*0.42,2.0*detail_scale,palette.text,0.95);
        canvas.line(tx,ty,tx-nx*len*0.32-px*len*0.42,ty-ny*len*0.32-py*len*0.42,2.0*detail_scale,palette.text,0.95);
        const int ns=std::max(1,std::min(3,static_cast<int>(std::lround(panel_radius/450.0))));
        pending_labels.push_back({tx-2,ty-9.0*ns,"N",ns,palette.text,1.0});
    }
    if (c.scale_bar!=0) {
        const double desired_px=panel_radius*0.46;
        const double target_m=desired_px/px_per_m;
        double scale_m=1.0;
        const double pow10=std::pow(10.0,std::floor(std::log10(std::max(target_m,1.0))));
        for (double mantissa : {1.0,2.0,5.0}) if (mantissa*pow10<=target_m) scale_m=mantissa*pow10;
        const double bar_px=scale_m*px_per_m;
        const double bx=cx-panel_radius*0.64, by=cy+panel_radius*0.68;
        canvas.line(bx,by,bx+bar_px,by,2.0*detail_scale,palette.text,0.98);
        canvas.line(bx,by-4*detail_scale,bx,by+4*detail_scale,1.6*detail_scale,palette.text,0.98);
        canvas.line(bx+bar_px,by-4*detail_scale,bx+bar_px,by+4*detail_scale,1.6*detail_scale,palette.text,0.98);
        char label[32];
        if (scale_m>=1000.0) std::snprintf(label,sizeof(label),"%.0f KM",scale_m/1000.0);
        else std::snprintf(label,sizeof(label),"%.0f M",scale_m);
        const int ts=std::max(1,std::min(3,static_cast<int>(std::lround(panel_radius/450.0))));
        pending_labels.push_back({bx,by+4.0,label,ts,palette.text,0.98});
    }

    const double feather_px=c.feather*std::min(width,height);
    const bool manual=c.manual_horizon>0.0;
    const double manual_y=manual?c.manual_horizon*height:static_cast<double>(height);
    bool mock_notice_safe=false;
    for (const TextItem &item:pending_labels) {
        if (item.text.rfind("MOCK",0)==0 &&
            safe_text_box(item,width,height,sky,c.auto_sky_fade!=0,manual,manual_y,feather_px)) {
            mock_notice_safe=true;
            break;
        }
    }
    if (!mock_notice_safe) {
        const double x=std::max(0.0,std::min(static_cast<double>(width)-24.0,cx-12.0));
        const double y=c.auto_sky_fade!=0?static_cast<double>(std::max(0,sky.min_y)):
                       std::max(0.0,cy-panel_radius*0.75);
        TextItem compact_notice{x,y,"MOCK",1,palette.marker,1.0};
        if (!safe_text_box(compact_notice,width,height,sky,c.auto_sky_fade!=0,manual,manual_y,feather_px)) {
            std::memset(out,0,count*4);
            return true;
        }
        pending_labels.push_back(std::move(compact_notice));
    }
    Canvas label_canvas(out,width,height,cx,cy,std::hypot(static_cast<double>(width),static_cast<double>(height)),1.0);
    for (const TextItem &item:pending_labels) {
        if (!safe_text_box(item,width,height,sky,c.auto_sky_fade!=0,manual,manual_y,feather_px)) continue;
        label_canvas.text(item.x,item.y,item.text.c_str(),item.scale,item.color,item.alpha);
    }
    for (size_t i=0;i<count;++i) {
        const uint32_t y=static_cast<uint32_t>(i/width);
        double mask=c.opacity;
        if (c.auto_sky_fade!=0) {
            if (!sky.pixels[i]) mask=0.0;
            else if (feather_px>0.0 && !sky.distance_inside.empty())
                mask*=clamp01((sky.distance_inside[i]/3.0)/feather_px);
        }
        if (manual) {
            const double depth=manual_y-(static_cast<double>(y)+0.5);
            if (depth<0.0) mask=0.0;
            else if (feather_px>0.0) mask*=clamp01(depth/feather_px);
        }
        if (out[i*4+3]!=0) {
            out[i*4+3]=static_cast<uint8_t>(std::lround(out[i*4+3]*mask));
            if (out[i*4+3]==0) out[i*4]=out[i*4+1]=out[i*4+2]=0;
        }
    }
    return true;
}
} // namespace

bool imprint::geo::TopConnectedRgbSkyMaskProvider::build(const uint8_t *rgb,uint32_t width,uint32_t height,
                                                         imprint::geo::SkyMask *out,std::string *error) const {
    if (error) error->clear();
    if (rgb==nullptr || out==nullptr || width==0 || height==0 ||
        static_cast<uint64_t>(width)*height>kMaximumPixels ||
        static_cast<uint64_t>(width)*height>std::numeric_limits<size_t>::max()/3) {
        if (error) *error="Invalid RGB buffer or dimensions for sky segmentation.";
        return false;
    }
    const size_t bytes=static_cast<size_t>(width)*height*3;
    struct Cache {
        uint64_t hash=0;
        uint32_t width=0,height=0;
        std::vector<uint8_t> exact_rgb;
        imprint::geo::SkyMask mask;
        bool valid=false;
    };
    static thread_local Cache cache;
    uint64_t hash=1469598103934665603ULL ^ (static_cast<uint64_t>(width)<<32) ^ height;
    for (size_t i=0;i<bytes;++i) {
        hash^=rgb[i];
        hash*=1099511628211ULL;
    }
    constexpr size_t kCachePixelLimit=16000000;
    if (static_cast<size_t>(width)*height<=kCachePixelLimit && cache.valid &&
        cache.hash==hash && cache.width==width && cache.height==height &&
        cache.exact_rgb.size()==bytes && std::memcmp(cache.exact_rgb.data(),rgb,bytes)==0) {
        *out=cache.mask;
        return true;
    }
    imprint::geo::SkyMask computed=find_top_connected_sky(rgb,width,height);
    if (static_cast<size_t>(width)*height<=kCachePixelLimit) {
        Cache next;
        next.hash=hash; next.width=width; next.height=height; next.valid=true;
        next.exact_rgb.assign(rgb,rgb+bytes);
        next.mask=computed;
        cache=std::move(next);
    } else {
        cache=Cache{};
    }
    *out=std::move(computed);
    return true;
}

bool imprint::geo::render_with_sky_provider(const uint8_t *rgb,uint32_t width,uint32_t height,
                                            const GeoConfig &config,uint8_t *overlay_rgba,
                                            const imprint::geo::ISkyMaskProvider &sky_provider,
                                            std::string *error) {
    if (config.provider==static_cast<int32_t>(imprint::geo::ProviderKind::Overpass)) {
        if (overlay_rgba && static_cast<uint64_t>(width)*height<=kMaximumPixels)
            std::memset(overlay_rgba,0,static_cast<size_t>(width)*height*4);
        if (error) *error="Overpass provider is unsupported in this native core; no map request was made.";
        return false;
    }
    imprint::geo::MockGeoDataProvider data_provider;
    return imprint::geo::render_with_providers(rgb,width,height,config,overlay_rgba,data_provider,sky_provider,error);
}

bool imprint::geo::render_with_providers(const uint8_t *rgb,uint32_t width,uint32_t height,
                                          const GeoConfig &config,uint8_t *overlay_rgba,
                                          const imprint::geo::IGeoDataProvider &data_provider,
                                          const imprint::geo::ISkyMaskProvider &sky_provider,
                                          std::string *error) {
    if (error) error->clear();
    size_t count=0;
    if (!checked_pixel_count(width,height,&count) || rgb==nullptr || overlay_rgba==nullptr) {
        if (error) *error="Invalid image dimensions or RGB/RGBA buffer.";
        return false;
    }
    std::memset(overlay_rgba,0,count*4);
    try {
        const char *why=nullptr;
        if (!valid_config(config,&why)) { if (error) *error=why; return false; }
        if (config.enabled==0) return true;
        if (config.style==3) { if (error) *error="Terrain style is unsupported because this core has no verified DEM source."; return false; }
        if (!render_impl(rgb,width,height,config,overlay_rgba,data_provider,sky_provider,&why)) {
            std::memset(overlay_rgba,0,count*4);
            if (error) *error=why?why:"Native geographic rendering failed.";
            return false;
        }
        return true;
    } catch (const std::bad_alloc &) {
        std::memset(overlay_rgba,0,count*4);
        if (error) *error="Native geographic overlay ran out of memory.";
        return false;
    } catch (...) {
        std::memset(overlay_rgba,0,count*4);
        if (error) *error="Unexpected native geographic overlay failure.";
        return false;
    }
}

extern "C" int32_t imprint_geo_render(const uint8_t *rgb,uint32_t width,uint32_t height,
                                       const GeoConfig *config,uint8_t *overlay_rgba,
                                       char *error,uint32_t error_capacity) {
    if (error && error_capacity) error[0]='\0';
    size_t count=0;
    if (!checked_pixel_count(width,height,&count)) {
        write_error(error,error_capacity,"Image dimensions are zero or exceed the native pixel limit."); return 1;
    }
    if (config==nullptr || rgb==nullptr || overlay_rgba==nullptr) {
        write_error(error,error_capacity,"RGB input, GeoConfig, and RGBA output buffers are required."); return 1;
    }
    std::memset(overlay_rgba,0,count*4);
    try {
        const char *why=nullptr;
        if (!valid_config(*config,&why)) { write_error(error,error_capacity,why); return 2; }
        if (config->enabled==0) {
            std::memset(overlay_rgba,0,count*4);
            return 0;
        }
        if (config->style==3) {
            std::memset(overlay_rgba,0,count*4);
            write_error(error,error_capacity,"Terrain style is unsupported because this core has no verified DEM source."); return 3;
        }
        if (config->provider==1) {
            std::memset(overlay_rgba,0,count*4);
            write_error(error,error_capacity,"Overpass provider is unsupported in this native core; no network request was made."); return 4;
        }
        imprint::geo::TopConnectedRgbSkyMaskProvider sky_provider;
        imprint::geo::MockGeoDataProvider data_provider;
        if (!render_impl(rgb,width,height,*config,overlay_rgba,data_provider,sky_provider,&why)) {
            std::memset(overlay_rgba,0,count*4);
            write_error(error,error_capacity,why); return 5;
        }
        return 0;
    } catch (const std::bad_alloc &) {
        std::memset(overlay_rgba,0,count*4);
        write_error(error,error_capacity,"Native geographic overlay ran out of memory.");
        return 6;
    } catch (...) {
        std::memset(overlay_rgba,0,count*4);
        write_error(error,error_capacity,"Unexpected native geographic overlay failure.");
        return 7;
    }
}

extern "C" int32_t imprint_geo_composite(const uint8_t *rgb,const uint8_t *overlay_rgba,
                                          uint32_t width,uint32_t height,uint8_t *out_rgb) {
    size_t count=0;
    if (!checked_pixel_count(width,height,&count) || rgb==nullptr || overlay_rgba==nullptr || out_rgb==nullptr) return 1;
    if (out_rgb==rgb) {
        for (size_t i=0;i<count;++i) {
            const uint8_t a=overlay_rgba[i*4+3];
            if (a==0) continue;
            const uint32_t inv=255u-a;
            out_rgb[i*3]=static_cast<uint8_t>((overlay_rgba[i*4]*a+rgb[i*3]*inv+127u)/255u);
            out_rgb[i*3+1]=static_cast<uint8_t>((overlay_rgba[i*4+1]*a+rgb[i*3+1]*inv+127u)/255u);
            out_rgb[i*3+2]=static_cast<uint8_t>((overlay_rgba[i*4+2]*a+rgb[i*3+2]*inv+127u)/255u);
        }
    } else {
        for (size_t i=0;i<count;++i) {
            const uint8_t a=overlay_rgba[i*4+3];
            if (a==0) {
                out_rgb[i*3]=rgb[i*3]; out_rgb[i*3+1]=rgb[i*3+1]; out_rgb[i*3+2]=rgb[i*3+2];
                continue;
            }
            const uint32_t inv=255u-a;
            out_rgb[i*3]=static_cast<uint8_t>((overlay_rgba[i*4]*a+rgb[i*3]*inv+127u)/255u);
            out_rgb[i*3+1]=static_cast<uint8_t>((overlay_rgba[i*4+1]*a+rgb[i*3+1]*inv+127u)/255u);
            out_rgb[i*3+2]=static_cast<uint8_t>((overlay_rgba[i*4+2]*a+rgb[i*3+2]*inv+127u)/255u);
        }
    }
    return 0;
}
