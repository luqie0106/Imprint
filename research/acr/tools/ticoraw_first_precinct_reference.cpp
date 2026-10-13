// Research-only API harness for pinned redlamp-libraw 4abfcd2.
// Calls the external decoder stages; does not call or link Adobe software.
#include "src/decoders/nikon_he/nikon_he_picture_header.h"
#include "src/decoders/nikon_he/nikon_he_subband_config.h"
#include "src/decoders/nikon_he/nikon_he_precinct_header.h"
#include "src/decoders/nikon_he/nikon_he_precinct_decode.h"
#include "src/decoders/nikon_he/nikon_he_gcli_decode.h"
#include "src/decoders/nikon_he/nikon_he_coefficient_decode.h"
#include "src/decoders/nikon_he/nikon_he_dequantize.h"
#include "src/decoders/nikon_he/nikon_he_gtli_table.h"
#include "src/decoders/nikon_he/nikon_he_predict_lut.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
using namespace nikon_he;
template<class T> void array(const std::vector<T>& values) {
    std::printf("[");
    for (size_t i=0;i<values.size();++i) std::printf("%s%d",i?",":"",int(values[i]));
    std::printf("]");
}
int main(int argc,char**argv) {
    if(argc!=4) {std::fprintf(stderr,"usage: reference NEF strip_offset strip_size\n");return 64;}
    std::ifstream file(argv[1],std::ios::binary);
    const auto offset=std::strtoull(argv[2],nullptr,10), size=std::strtoull(argv[3],nullptr,10);
    if (!file || size>100000000 || size<100) return 2;
    std::vector<uint8_t> strip(size);file.seekg(offset);file.read(reinterpret_cast<char*>(strip.data()),size);
    if(!file) return 3;
    PictureHeader ph{};
    if(!parse_picture_header(strip.data(),strip.size(),ph) || !is_supported_picture_header(ph,strip.size())) return 4;
    if(ph.precinct_offset > strip.size() || strip.size()-ph.precinct_offset < 12) return 5;
    set_active_picture_header(&ph);
    SubbandConfig cfg[26];LayoutInfo layout;compute_subband_layout(ph.hdr_width/2,cfg,layout);
    const uint8_t* data=strip.data()+ph.precinct_offset;
    const size_t full_size=(size_t(data[0])<<16 | size_t(data[1])<<8 | data[2])+12;
    if(ph.precinct_offset+full_size>strip.size()) return 5;
    PrecinctSizes sizes{};
    if(!parse_precinct_header(data,full_size,layout.sig_bytes,sizes)) return 6;
    PrecinctPredecessorState state;state.init(cfg);
    const auto predict=build_prediction_lookup_table();
    size_t capacity=0;
    for(auto&c:cfg)capacity=std::max(capacity,size_t(c.x24*4+c.ng*4));
    std::vector<int32_t> bufA(capacity),bufB(capacity);
    std::printf("{\"source_commit\":\"4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812\",\"width\":%u,\"height\":%u,\"Bp\":%u,\"Br\":%u,\"lbs\":[",ph.hdr_width,ph.hdr_height,sizes.Bp,sizes.Br);
    for(int lb=0;lb<8;++lb)std::printf("%s{\"lb\":%d,\"raw\":%u,\"sig_bytes\":%u,\"gcli_bytes\":%u,\"data_bytes\":%u,\"sign_bytes\":%u,\"sig_offset\":%u}",lb?",":"",lb,sizes.lb_f20_sign[lb],sizes.lb_sig_bytes[lb],sizes.lb_gcli_bytes[lb],sizes.lb_data_bytes[lb],sizes.lb_sign_bytes[lb],sizes.lb_sig_offset[lb]);
    std::printf("],\"bands\":[");
    const int starts[9]={0,6,12,13,19,21,23,24,26};
    for(int lb=0;lb<8;++lb) {
        const uint8_t* sig=data+sizes.lb_sig_offset[lb];
        const uint8_t* gcli=sig+sizes.lb_sig_bytes[lb];
        const uint8_t* coef=gcli+sizes.lb_gcli_bytes[lb];
        const uint8_t* sign=coef+sizes.lb_data_bytes[lb];
        BitReader sr(sig,sizes.lb_sig_bytes[lb]),gr(gcli,sizes.lb_gcli_bytes[lb]),dr(coef,sizes.lb_data_bytes[lb]),nr(sign,sizes.lb_sign_bytes[lb]);
        for(int sb=starts[lb];sb<starts[lb+1];++sb) {
            int ng=cfg[sb].ng,gtli=lookup_gtli_for_sub_band(sizes.Bp,sizes.Br,sb), mode=sizes.Dpb[sb]|0x70;
            std::vector<uint8_t> gc(ng);std::vector<int32_t> vals(ng*4);
            if(sizes.lb_f20_sign[lb]) {for(auto&g:gc)g=gr.read_bits(4);}
            else if(!decode_gcli_values(sr,gr,mode,ng,gtli,predict,state.get_previous_gcli(sb),gc.data()))return 7;
            unpack_coefficient_magnitudes(dr,gc.data(),gtli,ng,vals.data());
            std::printf("%s{\"sb\":%d,\"lb\":%d,\"ng\":%d,\"gtli\":%d,\"mode\":%d,\"gcli\":",sb?",":"",sb,lb,ng,gtli,mode);array(gc);
            std::printf(",\"magnitudes\":");array(vals);
            apply_sign_bits(nr,vals.data(),ng*4);std::printf(",\"signed\":");array(vals);
            dequantize_coefficient_array(vals.data(),ng*4,gc.data(),ng,gtli,sb==12||sb==23);
            std::printf(",\"dequantized\":");array(vals);
            std::printf("}");
            state.save_gcli(sb,gc.data());
            auto&buf=cfg[sb].buffer_idx?bufB:bufA;
            std::copy(vals.begin(),vals.end(),buf.begin()+cfg[sb].x24*4);
        }
    }
    state.destroy();
    PrecinctPredecessorState check;check.init(cfg);
    std::vector<int32_t> checkA(capacity),checkB(capacity);
    auto result=decode_precinct(data,full_size,ph.hdr_width,cfg,check,predict,checkA.data(),checkB.data());
    check.destroy();set_active_picture_header(nullptr);
    bool equal=result.success && checkA==bufA && checkB==bufB;
    std::printf("],\"orchestration_parity\":%s}\n",equal?"true":"false");
    return equal?0:8;
}
