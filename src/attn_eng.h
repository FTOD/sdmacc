#include "dimensions.h"
#include "hls_stream.h"
#define FMUL_PER_2pRAM 8

void attn_eng_row(hls::stream<float> &new_Q,
                  hls::stream<float> &new_K,
                  hls::stream<float> &new_V,
                  unsigned &global_idx,
                  hls::stream<float> &Z_out);

// template <int N>
void attn_eng_core_S(hls::stream<float> &Q,
                     hls::stream<float> &K,
                     hls::stream<float> &V,
                     hls::stream<float> &Z_stream,
                     float &S);

void attn_eng_core_mem(float QBuf[HEAD_DIM],
                       float KBuf[HEAD_DIM],
                       float VBuf[HEAD_DIM],
                       float ZBuf[HEAD_DIM],
                       float &S);

void MAC_S(hls::stream<float> &a,
           hls::stream<float> &b,
           float &acc);

void MAC_8(float &acc,
           float a[8],
           float b[8]);

void MUL_SV(float &S,
            hls::stream<float> &v,
            hls::stream<float> &res);

void MUL_8(float a[8],
           float b[8],
           float res[8]);

void MUL_8(float a,
           float b[8],
           float res[8]);

// template <int N, int M>
// void stream_gen(float in[HEAD_DIM],
//                 hls::stream<float> &in_stream);

void SV_stream_to_ram(hls::stream<float> &SV_res,
                      float ZBuf[HEAD_DIM]);

void Z_stream_reduction(hls::stream<float> Z_stream[DIA_KERNAL_WIDTH],
                        hls::stream<float> &Z_out);