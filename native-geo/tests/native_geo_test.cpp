#include "imprint_geo.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
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

bool render_features_ok(const std::vector<uint8_t> &rgb,uint32_t w,uint32_t h,const GeoConfig &c,
                        const GeoFeatureView *features,uint32_t feature_count,
                        std::vector<uint8_t> *rgba,std::string *error=nullptr) {
    rgba->assign(static_cast<size_t>(w)*h*4,0xA5);
    char err[256]{};
    const int result=imprint_geo_render_features(rgb.data(),w,h,&c,features,feature_count,
                                                  rgba->data(),err,sizeof(err));
    if (error) *error=err;
    return result==0;
}

GeoVertex local_vertex(const GeoConfig &c,double east_m,double north_m) {
    imprint::geo::LocalAzimuthalEquidistant projection(c.latitude,c.longitude);
    double lat=0.0,lon=0.0;
    if (!projection.inverse({east_m,north_m},&lat,&lon)) return {NAN,NAN};
    return {lat,lon};
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

void test_real_feature_abi_and_provider_routing() {
    constexpr uint32_t w=320,h=240;
    auto rgb=solid(w,h,45,70,92);
    auto c=config(); c.provider=1; c.north_arrow=0; c.scale_bar=0; c.photo_marker=0;
    const GeoVertex road_points[]={local_vertex(c,-5000.0,0.0),local_vertex(c,0.0,0.0),local_vertex(c,5000.0,0.0)};
    const GeoVertex water_points[]={local_vertex(c,-3500.0,2500.0),local_vertex(c,0.0,2200.0),local_vertex(c,3500.0,2500.0)};
    const GeoVertex place_points[]={local_vertex(c,1200.0,1800.0)};
    const GeoFeatureView features[]={
        {0,3,road_points,"Market Street"},
        {1,3,water_points,"Lagoon"},
        {3,1,place_points,"Sausalito"}};
    std::vector<uint8_t> rgba;
    std::string error;
    check(render_features_ok(rgb,w,h,c,features,3,&rgba,&error),
          "C ABI renders caller-supplied OSM roads, water, and a point place");
    check(alpha_count(rgba)>0,"real feature input produces visible map pixels without auto-sky gating");

    c=config(); c.provider=1; c.labels=0; c.north_arrow=0; c.scale_bar=0; c.photo_marker=0;
    const GeoVertex real_road[]={local_vertex(c,-5000.0,0.0),local_vertex(c,5000.0,0.0)};
    const GeoFeatureView real_view{0,2,real_road,""};
    check(render_features_ok(rgb,w,h,c,&real_view,1,&rgba,&error),"minimal real dataset renders successfully");
    bool real_has_mock_stamp_pixels=false;
    for (uint32_t y=40;y<56;++y) for (uint32_t x=120;x<184;++x)
        real_has_mock_stamp_pixels|=rgba[(static_cast<size_t>(y)*w+x)*4+3]!=0;
    check(!real_has_mock_stamp_pixels,"real dataset does not receive the MOCK MAP stamp");
    c.provider=0;
    std::vector<uint8_t> mock_rgba;
    check(render_ok(rgb,w,h,c,&mock_rgba,&error),"legacy C ABI still renders its synthetic provider");
    bool mock_has_stamp_pixels=false;
    for (uint32_t y=40;y<56;++y) for (uint32_t x=120;x<184;++x)
        mock_has_stamp_pixels|=mock_rgba[(static_cast<size_t>(y)*w+x)*4+3]!=0;
    check(mock_has_stamp_pixels,"Mock provider retains an explicit MOCK MAP stamp");

    c.provider=1;
    check(!render_ok(rgb,w,h,c,&rgba,&error)&&error.find("caller-supplied features")!=std::string::npos,
          "legacy provider=1 reports missing map data instead of silently using Mock");
    check(alpha_count(rgba)==0,"provider=1 without features leaves a transparent error layer");

    struct FixedDataProvider final : imprint::geo::IGeoDataProvider {
        imprint::geo::GeoDataset data;
        bool load(const imprint::geo::GeoQuery &,imprint::geo::GeoDataset *out,std::string *load_error) const override {
            if (!out) return false;
            *out=data;
            if (load_error) load_error->clear();
            return true;
        }
    } provider;
    provider.data.mock_data=false;
    imprint::geo::GeoFeature path;
    path.kind=imprint::geo::FeatureKind::Road;
    path.points_deg={{real_road[0].latitude,real_road[0].longitude},{real_road[1].latitude,real_road[1].longitude}};
    provider.data.features.push_back(path);
    imprint::geo::TopConnectedRgbSkyMaskProvider sky_provider;
    check(imprint::geo::render_with_providers(rgb.data(),w,h,c,rgba.data(),provider,sky_provider,&error),
          "render_with_providers accepts a real dataset for provider=1");
    check(alpha_count(rgba)>0,"injected real provider draws paths without Mock branding");
}

void test_feature_validation_clipping_and_label_fallback() {
    constexpr uint32_t w=320,h=240;
    auto rgb=solid(w,h,45,70,92);
    auto c=config(); c.provider=1; c.labels=0; c.north_arrow=0; c.scale_bar=0; c.photo_marker=0;
    std::vector<uint8_t> rgba;
    std::string error;
    check(!render_features_ok(rgb,w,h,c,nullptr,0,&rgba,&error)&&error.find("No geographic features")!=std::string::npos,
          "empty feature input returns an explicit no-data error");
    check(alpha_count(rgba)==0,"empty feature input does not fall back to Mock geometry");
    c.enabled=0;
    check(render_features_ok(rgb,w,h,c,nullptr,0,&rgba,&error),
          "disabled OSM overlay does not require a map download");
    check(std::all_of(rgba.begin(),rgba.end(),[](uint8_t value){return value==0;}),
          "disabled OSM overlay is a byte-exact empty RGBA layer");

    c=config(); c.provider=1;
    const GeoVertex valid_points[]={local_vertex(c,-1000.0,0.0),local_vertex(c,1000.0,0.0)};
    GeoFeatureView invalid{0,2,valid_points,"valid"};
    GeoVertex bad_points[]={{NAN,-122.0},{37.0,-122.0}};
    invalid.points=bad_points;
    check(!render_features_ok(rgb,w,h,c,&invalid,1,&rgba,&error)&&error.find("finite WGS84")!=std::string::npos,
          "feature coordinates reject NaN before projection");
    invalid={3,2,valid_points,"place"};
    check(!render_features_ok(rgb,w,h,c,&invalid,1,&rgba,&error)&&error.find("Place features need one point")!=std::string::npos,
          "place geometry must contain exactly one point");
    invalid={0,500001,valid_points,"too many"};
    check(!render_features_ok(rgb,w,h,c,&invalid,1,&rgba,&error)&&error.find("500,000 point limit")!=std::string::npos,
          "feature point total is bounded before geometry is read");
    invalid={0,2,valid_points,nullptr};
    std::string long_label(257,'A'); invalid.label=long_label.c_str();
    check(!render_features_ok(rgb,w,h,c,&invalid,1,&rgba,&error)&&error.find("256 bytes")!=std::string::npos,
          "feature labels are capped at 256 UTF-8 bytes");
    GeoFeatureView too_many[1]{};
    check(!render_features_ok(rgb,w,h,c,too_many,10001,&rgba,&error)&&error.find("10,000")!=std::string::npos,
          "feature count is capped at 10,000");

    c=config(); c.provider=1; c.labels=0; c.north_arrow=0; c.scale_bar=0; c.photo_marker=0;
    const GeoVertex long_path[]={local_vertex(c,-1000000.0,0.0),local_vertex(c,1000000.0,0.0)};
    const GeoFeatureView long_view{0,2,long_path,""};
    check(render_features_ok(rgb,w,h,c,&long_view,1,&rgba,&error),
          "long off-screen geometry is clipped before path stamping");
    check(alpha_count(rgba)>0,"viewport-crossing long road remains visible after clipping");

    c=config(); c.provider=1; c.offset_y=-1.0; c.manual_horizon=0.01;
    c.auto_sky_fade=0; c.labels=1; c.north_arrow=0; c.scale_bar=0; c.photo_marker=0;
    const double panel_radius=std::min(w,h)*c.coverage*c.scale*0.5;
    const double center_y=static_cast<double>(h)*0.5+c.offset_y*static_cast<double>(h)*0.25;
    const double north=(center_y-1.0)/(panel_radius/(c.radius_km*1000.0));
    const GeoVertex edge_path[]={local_vertex(c,-5000.0,north),local_vertex(c,5000.0,north)};
    const GeoFeatureView edge_view{0,2,edge_path,"Unplaceable label"};
    check(render_features_ok(rgb,w,h,c,&edge_view,1,&rgba,&error),
          "road rendering succeeds when all text boxes are outside the safe area");
    check(alpha_count(rgba)>0,"an unplaceable optional label never clears visible road pixels");
    c.opacity=0.0;
    check(render_features_ok(rgb,w,h,c,&edge_view,1,&rgba,&error),"zero-opacity real feature render succeeds");
    check(std::all_of(rgba.begin(),rgba.end(),[](uint8_t value){return value==0;}),
          "zero-opacity feature rendering leaves a transparent identity layer");
}

void test_shared_sky_cache_cross_thread() {
    constexpr uint32_t w=256,h=160;
    auto rgb=solid(w,80,54,76,98);
    auto lower=solid(w,h-80,170,174,178);
    rgb.insert(rgb.end(),lower.begin(),lower.end());
    imprint::geo::TopConnectedRgbSkyMaskProvider provider;
    imprint::geo::SkyMask first,second;
    std::string error;
    check(provider.build(rgb.data(),w,h,&first,&error),"main thread builds a full sky mask");
    bool thread_ok=false;
    std::thread worker([&] { thread_ok=provider.build(rgb.data(),w,h,&second,&error); });
    worker.join();
    check(thread_ok,"worker thread builds or reuses the process-shared mask");
    check(first.width==second.width&&first.height==second.height&&first.pixels==second.pixels&&
          first.distance_inside==second.distance_inside,
          "cross-thread cache reuse preserves exact mask and distance field contents");
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
    test_real_feature_abi_and_provider_routing();
    test_feature_validation_clipping_and_label_fallback();
    test_shared_sky_cache_cross_thread();
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
