#include "imprint_geo.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
int failures=0;
void check(bool condition,const char *message) {
    if (condition) return;
    std::fprintf(stderr,"FAIL: %s\n",message);
    ++failures;
}

GeoConfig config() {
    GeoConfig c{};
    c.latitude=37.7749; c.longitude=-122.4194; c.opacity=0.9; c.radius_km=10.0;
    c.offset_x=0.0; c.offset_y=0.0; c.scale=1.0; c.rotation=0.0;
    c.line_width=2.0; c.feather=0.0; c.coverage=0.8; c.manual_horizon=0.0;
    c.enabled=1; c.has_location=1; c.style=0; c.auto_sky_fade=0;
    c.labels=1; c.north_arrow=1; c.scale_bar=1; c.photo_marker=1;
    c.adaptive=1; c.provider=0;
    return c;
}

std::vector<uint8_t> solid(uint32_t w,uint32_t h,uint8_t r,uint8_t g,uint8_t b) {
    std::vector<uint8_t> v(static_cast<size_t>(w)*h*3);
    for (size_t i=0;i<v.size();i+=3) { v[i]=r; v[i+1]=g; v[i+2]=b; }
    return v;
}

bool render_ok(const std::vector<uint8_t> &rgb,uint32_t w,uint32_t h,const GeoConfig &c,
               std::vector<uint8_t> *rgba,std::string *error=nullptr) {
    rgba->assign(static_cast<size_t>(w)*h*4,0xA5);
    char err[256]{};
    const int result=imprint_geo_render(rgb.data(),w,h,&c,rgba->data(),err,sizeof(err));
    if (error) *error=err;
    return result==0;
}

size_t alpha_count(const std::vector<uint8_t> &rgba) {
    size_t n=0;
    for (size_t i=3;i<rgba.size();i+=4) if (rgba[i]) ++n;
    return n;
}
uint64_t alpha_sum(const std::vector<uint8_t> &rgba) {
    uint64_t n=0;
    for (size_t i=3;i<rgba.size();i+=4) n+=rgba[i];
    return n;
}

struct SimpleSky final : imprint::geo::ISkyMaskProvider {
    bool build(const uint8_t *,uint32_t w,uint32_t h,imprint::geo::SkyMask *out,std::string *error) const override {
        if (!out) return false;
        out->width=w; out->height=h; out->pixels.assign(static_cast<size_t>(w)*h,0);
        out->distance_inside.assign(static_cast<size_t>(w)*h,0);
        out->bottom_by_column.assign(w,-1); out->min_x=0; out->min_y=0;
        out->max_x=static_cast<int32_t>(w)-1; out->max_y=static_cast<int32_t>(h/2)-1; out->area=0;
        for (uint32_t y=0;y<h/2;++y) for (uint32_t x=0;x<w;++x) {
            const size_t i=static_cast<size_t>(y)*w+x;
            out->pixels[i]=1; out->distance_inside[i]=300; ++out->area;
            out->bottom_by_column[x]=static_cast<int32_t>(h/2)-1;
        }
        if (error) error->clear();
        return true;
    }
};

void test_projection() {
    imprint::geo::LocalAzimuthalEquidistant p(0.0,0.0);
    imprint::geo::MetricPoint east,north;
    check(p.forward(0.0,0.00899321606,&east),"projection accepts a nearby east coordinate");
    check(std::fabs(east.east_m-1000.0)<0.15 && std::fabs(east.north_m)<0.15,
          "local projection preserves the true 1 km eastward metric distance");
    check(p.forward(0.00899321606,0.0,&north),"projection accepts a nearby north coordinate");
    check(std::fabs(north.north_m-1000.0)<0.15 && std::fabs(north.east_m)<0.15,
          "local projection preserves the true 1 km northward metric distance");
    double lat=0,lon=0;
    check(p.inverse(east,&lat,&lon) && std::fabs(lon-0.00899321606)<1e-8,
          "projection inverse round-trips coordinates");
    check(!imprint::geo::LocalAzimuthalEquidistant(91.0,0.0).valid(),"projection rejects invalid origins");
}

void test_validation_and_identity() {
    const uint32_t w=320,h=240;
    auto rgb=solid(w,h,40,80,130);
    std::vector<uint8_t> rgba;
    auto c=config();
    std::string error;
    c.has_location=0;
    check(!render_ok(rgb,w,h,c,&rgba,&error) && error.find("missing")!=std::string::npos,
          "missing GPS fails explicitly");
    c=config(); c.latitude=0.0; c.longitude=0.0;
    check(render_ok(rgb,w,h,c,&rgba,&error) && alpha_count(rgba)>0,
          "(0,0) is accepted as a real GPS coordinate");
    c=config(); c.latitude=91.0;
    check(!render_ok(rgb,w,h,c,&rgba,&error) && error.find("invalid")!=std::string::npos,
          "out-of-range GPS coordinates fail explicitly");
    c=config(); c.longitude=std::numeric_limits<double>::quiet_NaN();
    check(!render_ok(rgb,w,h,c,&rgba,&error) && error.find("NaN")!=std::string::npos,
          "NaN configuration fails at the C ABI");
    c=config(); c.enabled=0; c.has_location=0; c.latitude=std::numeric_limits<double>::quiet_NaN();
    check(render_ok(rgb,w,h,c,&rgba,&error) && alpha_count(rgba)==0,
          "disabled overlay returns an all-zero identity layer");
    check(std::all_of(rgba.begin(),rgba.end(),[](uint8_t x){return x==0;}),
          "disabled overlay clears every RGBA output byte");
    c=config(); c.style=3;
    check(!render_ok(rgb,w,h,c,&rgba,&error) && error.find("Terrain")!=std::string::npos,
          "Terrain is reported unsupported without DEM");
    c=config(); c.provider=1;
    check(!render_ok(rgb,w,h,c,&rgba,&error) && error.find("Overpass")!=std::string::npos,
          "Overpass is reported unsupported without making a request");
    imprint::geo::OverpassGeoDataProvider overpass;
    imprint::geo::GeoDataset dataset;
    check(!overpass.load({0.0,0.0,1000.0},&dataset,&error) && error.find("no map request")!=std::string::npos,
          "Overpass provider itself is an explicit unsupported stub");
    imprint::geo::MockGeoDataProvider mock;
    check(mock.load({0.0,0.0,1000.0},&dataset,&error) && dataset.mock_data && dataset.features.size()>=8,
          "Mock provider labels and returns a dense synthetic roads/water/coast dataset");
}

void test_sky_mask_and_buildings() {
    constexpr uint32_t w=160,h=100;
    auto rgb=solid(w,50,50,68,84);
    auto lower=solid(w,h-50,175,178,181);
    rgb.insert(rgb.end(),lower.begin(),lower.end());
    imprint::geo::TopConnectedRgbSkyMaskProvider provider;
    imprint::geo::SkyMask mask;
    std::string error;
    check(provider.build(rgb.data(),w,h,&mask,&error),"default sky provider accepts a valid image");
    check(mask.area>static_cast<size_t>(w*h*0.15),"top-connected low-luma blue-gray sky exceeds 15 percent");
    check(mask.max_y<50,"gray building boundary does not leak into top-connected sky mask");
    auto c=config(); c.auto_sky_fade=1; c.adaptive=1; c.feather=0.06; c.labels=1;
    std::vector<uint8_t> rgba;
    check(render_ok(rgb,w,h,c,&rgba,&error),"sky-masked overlay renders on low-luma blue-gray sky");
    size_t sky_alpha=0, building_alpha=0;
    for (uint32_t y=0;y<h;++y) for (uint32_t x=0;x<w;++x) {
        const size_t a=(static_cast<size_t>(y)*w+x)*4+3;
        if (!rgba[a]) continue;
        if (y<50) ++sky_alpha; else ++building_alpha;
    }
    check(sky_alpha>0,"overlay uses available sky pixels");
    check(building_alpha==0,"overlay never enters the gray building region");
    check(provider.build(solid(w,h,80,80,80).data(),w,h,&mask,&error) && mask.area==0,
          "same-sized unrelated image does not reuse a stale sky mask");
    auto no_sky=solid(w,h,80,80,80);
    c=config(); c.auto_sky_fade=1;
    check(render_ok(no_sky,w,h,c,&rgba,&error) && alpha_count(rgba)==0,
          "no-sky image produces an empty transparent layer");
    auto sky_only=solid(200,100,47,65,83);
    check(provider.build(sky_only.data(),200,100,&mask,&error) && mask.area>3000,
          "large connected sky is retained without a 15 percent cutoff");

    auto side_subject=solid(w,50,50,68,84);
    for (uint32_t y=19;y<50;++y) for (uint32_t x=0;x<31;++x) {
        const size_t i=(static_cast<size_t>(y)*w+x)*3;
        side_subject[i]=175; side_subject[i+1]=178; side_subject[i+2]=181;
    }
    side_subject.insert(side_subject.end(),lower.begin(),lower.end());
    check(provider.build(side_subject.data(),w,h,&mask,&error),"sky provider handles a side-entry subject");
    const size_t near_subject=static_cast<size_t>(25)*w+31;
    const size_t open_sky=static_cast<size_t>(25)*w+80;
    check(mask.pixels[near_subject] && mask.distance_inside[near_subject]<mask.distance_inside[open_sky],
          "feather distance measures side boundaries in two dimensions");
}

void test_manual_boundary_and_injection() {
    constexpr uint32_t w=240,h=160;
    auto rgb=solid(w,h,65,86,110);
    auto c=config(); c.manual_horizon=0.25; c.auto_sky_fade=0; c.feather=0.03;
    std::vector<uint8_t> rgba;
    std::string error;
    check(render_ok(rgb,w,h,c,&rgba,&error),"manual horizon render succeeds");
    size_t below=0;
    for (uint32_t y=static_cast<uint32_t>(h*c.manual_horizon);y<h;++y)
        for (uint32_t x=0;x<w;++x) below+=rgba[(static_cast<size_t>(y)*w+x)*4+3]!=0;
    check(below==0,"manual horizon is a hard protection boundary below its feather");
    c=config(); c.auto_sky_fade=1;
    SimpleSky custom;
    std::fill(rgba.begin(),rgba.end(),0);
    check(imprint::geo::render_with_sky_provider(rgb.data(),w,h,c,rgba.data(),custom,&error),
          "C++ render entry accepts a replaceable sky-mask provider");
    check(alpha_count(rgba)>0,"injected sky provider controls sky visibility");
}

void test_scale_color_aspect_and_narrow_images() {
    auto c=config();
    std::vector<uint8_t> rgba;
    std::string error;
    for (const auto dims : {std::pair<uint32_t,uint32_t>{1,1},{2,2},{2,40},{39,8},{240,800},{800,240}}) {
        auto rgb=solid(dims.first,dims.second,60,82,103);
        c=config(); c.auto_sky_fade=1;
        check(render_ok(rgb,dims.first,dims.second,c,&rgba,&error),"narrow/aspect-ratio image renders without sampling faults");
    }
    constexpr uint32_t w=320,h=240;
    auto rgb=solid(w,h,40,80,130);
    c=config(); c.color_rgb=0xE02030;
    check(render_ok(rgb,w,h,c,&rgba,&error),"custom color render succeeds");
    const auto red_layer=rgba;
    c.color_rgb=0x20D040;
    check(render_ok(rgb,w,h,c,&rgba,&error),"second custom color render succeeds");
    bool found_red=false,found_green=false;
    for (size_t i=0;i<rgba.size();i+=4) {
        if (red_layer[i+3] && red_layer[i]>red_layer[i+1]*2) found_red=true;
        if (rgba[i+3] && rgba[i+1]>rgba[i]*2) found_green=true;
    }
    check(found_red && found_green,"color_rgb controls visible paths and text");

    auto small=solid(320,240,40,80,130);
    auto large=solid(640,480,40,80,130);
    c=config(); c.labels=0; c.north_arrow=0; c.scale_bar=0;
    std::vector<uint8_t> small_rgba,large_rgba;
    check(render_ok(small,320,240,c,&small_rgba,&error) && render_ok(large,640,480,c,&large_rgba,&error),
          "normalized preview and full-size render both succeed");
    const double area_ratio=static_cast<double>(alpha_sum(large_rgba))/std::max<uint64_t>(1,alpha_sum(small_rgba));
    check(area_ratio>3.2 && area_ratio<4.8,
          "line widths, marker, and geometry scale consistently with image dimensions");

    c=config(); c.opacity=0.18;
    check(render_ok(rgb,w,h,c,&rgba,&error),"low-opacity vector render succeeds");
    const auto low=rgba;
    c.opacity=1.0;
    check(render_ok(rgb,w,h,c,&rgba,&error),"full-opacity vector render succeeds");
    uint64_t low_sum=0,full_sum=0;
    uint8_t low_max=0;
    for (size_t i=3;i<rgba.size();i+=4) {
        low_sum+=low[i]; full_sum+=rgba[i]; low_max=std::max(low_max,low[i]);
    }
    check(full_sum>0 && std::fabs(static_cast<double>(low_sum)/full_sum-0.18)<0.015,
          "18 percent global opacity scales accumulated line coverage linearly");
    check(low_max<=46,"overlapping strokes never exceed the configured layer opacity");
}

void test_composite() {
    const uint8_t rgb[]={10,20,30,200,150,100};
    const uint8_t rgba[]={255,0,255,0, 0,255,0,128};
    uint8_t out[6]{};
    check(imprint_geo_composite(rgb,rgba,2,1,out)==0,"composite succeeds");
    check(out[0]==rgb[0] && out[1]==rgb[1] && out[2]==rgb[2],"alpha-zero composite is byte-exact identity");
    check(out[3]==100 && out[4]==203 && out[5]==50,"straight-alpha source-over uses expected sRGB code-value result");
    uint8_t inplace[]={10,20,30,200,150,100};
    check(imprint_geo_composite(inplace,rgba,2,1,inplace)==0 && std::memcmp(inplace,out,sizeof(out))==0,
          "in-place source-over preserves alpha-zero identity");
}

void test_performance() {
    constexpr uint32_t w=1600,h=1200;
    auto rgb=solid(w,h,42,64,82);
    auto c=config(); c.auto_sky_fade=1; c.adaptive=1;
    std::vector<uint8_t> rgba;
    const auto start=std::chrono::steady_clock::now();
    const bool ok=render_ok(rgb,w,h,c,&rgba);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    check(ok,"large preview render completes");
    check(seconds<8.0,"large preview render stays under the standalone 8 second budget");
}
}

int main() {
    test_projection();
    test_validation_and_identity();
    test_sky_mask_and_buildings();
    test_manual_boundary_and_injection();
    test_scale_color_aspect_and_narrow_images();
    test_composite();
    test_performance();
    if (failures) {
        std::fprintf(stderr,"%d native geo check(s) failed\n",failures);
        return 1;
    }
    std::puts("All native geo checks passed.");
    return 0;
}
