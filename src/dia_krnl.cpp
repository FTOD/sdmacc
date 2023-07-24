#include "dia_krnl.h"
#include <hls_math.h>

#define MODEL_DIM 1024
#define VDATA_SIZE MODEL_DIM
#define DIA_KERNAL_WIDTH 32

typedef struct
{
	float v[VDATA_SIZE];
} v_dt;

void dia_krnl(const v_dt *QCol,
			const v_dt *KRow,
			const v_dt *VCol,
			v_dt *out)
{
	// QK
	float dia[DIA_KERNAL_WIDTH][MODEL_DIM];
	for (int d = 0; d < DIA_KERNAL_WIDTH; d++)
		for (int i = 0; i < VDATA_SIZE; i++)
		{
#pragma HLS unroll
			dia[d][i] = QCol->v[i] * KRow->v[i];
		}
	// SV
	for (int d = 0; d < DIA_KERNAL_WIDTH; d++)
		for (int i = 0; i < VDATA_SIZE; i++)
		{
#pragma HLS unroll
			out[d][i] = dia[d][i] * VCol->v[i];
		}
}
// 	void half_mul(const half &a, const half &b, half &c)
// 	{
// 		half tmp;
// #pragma HLS BIND_OP variable = tmp op = fmul impl = dsp
// 		tmp = a * b;
// 		c = tmp;
// 	}
//
// 	void vec_vec_mul(const v_dt *QCol,
// 					 const v_dt *KRow,
// 					 v_dt *out)
// 	{
// 	loop_vec_vec:
// 		for (int i = 0; i < VDATA_SIZE; i++)
// 		{
// #pragma HLS unroll impl = dsp
// 			half tmp;
// #pragma HLS BIND_OP variable = tmp op = fmul impl = dsp
// 			tmp = QCol->v[i] * KRow->v[i];
// 			out->v[i] = tmp;
// 		}
// 	}
//
// 	void QK_slice_mul(const v_dt *QCol,
// 					  const v_dt *KRow,
// 					  const uint8_t shift_offset,
// 					  v_dt *out)
// 	{
// 	loop_QK_slice:
// 		for (int i = 0; i < VDATA_SIZE; i++)
// 		{
// #pragma HLS unroll
// 			uint16_t QIndex;
// 			if (i + shift_offset > VDATA_SIZE)
// 				QIndex = i + shift_offset - VDATA_SIZE;
// 			else
// 				QIndex = i + shift_offset;
// 			half q, k, tmp;
// 			q = QCol->v[QIndex];
// 			k = KRow->v[i];
// 			tmp = q * k;
// 			out->v[i] = tmp;
// 		}
// 	}

// void QK_mul(const v_dt *QCol,
// 			const v_dt *KRow,
// 			dia_dt *Dia)
// {
// loop_QK:
// 	for (uint8_t i = 0; i < DIA_KERNAL_WIDTH; i++)
// 	{
// #pragma HLS unroll                           \
	// 		QK_slice_mul(QCol, KRow, i, Dia->dia[i]); \
	// 	}                                          \
	// }