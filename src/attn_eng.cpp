#include "attn_eng.h"
#include <hls_math.h>
#include <hls_stream.h>

//!! FIXME XXX HEAD_DIM and DIA_KERNAL_WIDTH must be same, synthesis error otherwise

// you should take the stream per vector (per HEAD_DIM x FP32)
void attn_eng_row(hls::stream<float> &new_Q,
                  hls::stream<float> &new_K,
                  hls::stream<float> &new_V,
                  unsigned &global_idx,
                  hls::stream<float> &Z_out)
{
    // DIA_KERNEL_WIDTH = DIA_KERNAL_WIDTH
    float VBuf[DIA_KERNAL_WIDTH][HEAD_DIM];
    float KBuf[DIA_KERNAL_WIDTH][HEAD_DIM];
#pragma HLS array_partition variable = VBuf type = complete dim = 1
#pragma HLS array_partition variable = KBuf type = complete dim = 1

    float QBuf[HEAD_DIM];
#pragma HLS array_partition variable = QBuf type = complete dim = 0

    float S[DIA_KERNAL_WIDTH];
#pragma HLS array_partition variable = QBuf type = complete dim = 0

    // LOAD the new vector
    unsigned idx = global_idx % DIA_KERNAL_WIDTH;

    for (int i = 0; i < HEAD_DIM; i++)
    {
        float tmp_Q, tmp_K, tmp_V;
        new_Q.read(tmp_Q);
        new_K.read(tmp_K);
        new_V.read(tmp_V);
        KBuf[idx][i] = tmp_K;
        VBuf[idx][i] = tmp_V;
        QBuf[i] = tmp_Q;
    }

    hls::stream<float> Q_stream[DIA_KERNAL_WIDTH];
    hls::stream<float> K_stream[DIA_KERNAL_WIDTH];
    hls::stream<float> V_stream[DIA_KERNAL_WIDTH];
    hls::stream<float> Z_stream[DIA_KERNAL_WIDTH];
    for (int i = 0; i < DIA_KERNAL_WIDTH; i++)
    {
#pragma HLS unroll
        for (int j = 0; j < HEAD_DIM; j++)
        {
            Q_stream[i].write(QBuf[i]);
            K_stream[i].write(KBuf[i][j]);
            V_stream[i].write(VBuf[i][j]);
        }
    }
    for (int i = 0; i < DIA_KERNAL_WIDTH; i++)
    {
#pragma HLS unroll
        attn_eng_core_S(Q_stream[i], K_stream[i], V_stream[i], Z_stream[i], S[i]);
    }
    Z_stream_reduction(Z_stream, Z_out);
}

void attn_eng_core_S(hls::stream<float> &Q,
                     hls::stream<float> &K,
                     hls::stream<float> &V,
                     hls::stream<float> &Z_stream,
                     float &S)
{
    float S_tmp;
    MAC_S(Q, K, S_tmp);
    S_tmp = expf(S_tmp);
    S = S_tmp;
    MUL_SV(S_tmp, V, Z_stream);
}

void MAC_S(hls::stream<float> &a,
           hls::stream<float> &b,
           float &acc)
{
    float sum_tmp = 0;
    for (int i = 0; i < 8; i++)
    {
#pragma HLS unroll
        float tmp_a, tmp_b, tmp;
        a.read(tmp_a);
        b.read(tmp_b);
        tmp = tmp_a * tmp_b;
        sum_tmp += tmp;
    }
    acc += sum_tmp;
}

void MUL_SV(float &S,
            hls::stream<float> &v,
            hls::stream<float> &res)
{
    float sum_tmp = 0;
    for (int i = 0; i < 8; i++)
    {
#pragma HLS unroll
        float tmp_v, tmp;
        v.read(tmp_v);
        tmp = tmp_v * S;
        res.write(tmp);
    }
}

void MAC_8(float &acc,
           float a[8],
           float b[8])
{
#pragma HLS array_partition variable = a type = complete dim = 0
#pragma HLS array_partition variable = b type = complete dim = 0
#pragma HLS interface mode = ap_fifo port = a
#pragma HLS interface mode = ap_fifo port = b
    float tmp_sum;
    for (int i = 0; i < 8; i++)
    {
#pragma HLS pipeline
        float tmp;
        tmp = a[i] * b[i];
        tmp_sum += tmp;
    }
    acc += tmp_sum;
}

void MUL_8(float a[8],
           float b[8],
           float res[8])
{
#pragma HLS array_partition variable = a type = complete dim = 0
#pragma HLS array_partition variable = b type = complete dim = 0
#pragma HLS array_partition variable = res type = complete dim = 0
#pragma HLS interface mode = ap_fifo port = a
#pragma HLS interface mode = ap_fifo port = b
    for (int i = 0; i < 8; i++)
    {
#pragma HLS unroll
        float tmp;
        tmp = a[i] * b[i];
        res[i] = tmp;
    }
}

void MUL_8(float a,
           float b[8],
           float res[8])
{
#pragma HLS array_partition variable = a type = complete dim = 0
#pragma HLS array_partition variable = b type = complete dim = 0
#pragma HLS array_partition variable = res type = complete dim = 0
#pragma HLS interface mode = ap_fifo port = a
#pragma HLS interface mode = ap_fifo port = b
    for (int i = 0; i < 8; i++)
    {
#pragma HLS unroll
        float tmp;
        tmp = a * b[i];
        res[i] = tmp;
    }
}

void Z_stream_reduction(hls::stream<float> Z_stream[DIA_KERNAL_WIDTH],
                        float& scaler,
                        hls::stream<float> &Z_out)
{
    int sum = 0;
    for (int i = 0; i < DIA_KERNAL_WIDTH; i++)
    {
#pragma HLS unroll
        float tmp;
        Z_stream[i].read(tmp);
        sum += tmp;
    }
    float tmp1;
    tmp1 = sum/scaler;
    Z_out.write(tmp1);
}