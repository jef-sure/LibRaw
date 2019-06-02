/* -*- C++ -*-
 * File: dht_nn_demosaic.cpp
 * Copyright 2019 Anton Petrusevich
 * Created: Tue Apr  14, 2019
 *
 * This code is licensed under one of three licenses as you choose:
 *
 * 1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
 *    (See file LICENSE.LGPL provided in LibRaw distribution archive for details).
 *
 * 2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
 *    (See file LICENSE.CDDL provided in LibRaw distribution archive for details).
 *
 * 3. LibRaw Software License 27032010
 *    (See file LICENSE.LibRaw.pdf provided in LibRaw distribution archive for details).
 *
 */

#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

#include <iostream>

// @formatter:off
using namespace std;
using namespace dynet; /* @suppress("Symbol is not resolved") */
// @formatter:on
struct DhtNn {
	typedef float float3[3];
	enum calc_t {
		CC_UD = 0, // Computation using Chroma
		CC_LR = 1,
		CC_LU = 2,
		CC_LD = 3,
		CC_RU = 4,
		CC_RD = 5,
		CC_U = 6,
		CC_L = 7,
		CC_D = 8,
		CC_R = 9,
	};
	enum {
		LD_UD = 0, // LD == lightness directed
		LD_LR = 1,
		LD_LU = 2,
		LD_LD = 3,
		LD_RU = 4,
		LD_RD = 5,
		SG_UD = 6, // SG == single gradient
		SG_LR = 7,
		SG_DU = 8,
		SG_RL = 9,
		SG_ULDR = 10, // diagonal directed gradient
		SG_DRUL = 11,
		SG_URDL = 12,
		SG_DLUR = 13,
		MG_UD = 14, // MG == multi co-directed gradient
		MG_LR = 15,
		MG_DU = 16,
		MG_RL = 17,
		CI_R = 18, // Color index
		CI_G = 19,
		CI_B = 20,
	};
	static const int HDMV = LD_RD + 1;
	typedef float ld_raw_t[HDMV + 2];
	static const int NUM_FEATURES = CI_B + 1;
	typedef uchar features_t[21];
	typedef uchar ld_map_t[6];
	typedef float nnt_answers_t[10];

	static const int ompTileSize = 128;
	static constexpr const float JND = 0.03f; // Just Noticable Difference
	static constexpr const float TLC = 1.3f; // Threshold for LC difference

	int iHeight, iWidth, sd4Height, sd4Width, sd2Height, sd2Width;
	const int margin = 4;
	static constexpr const float zeroColor = sqrt(0.25f);
	static const int nnTile = 5;
	static const int nnMargin = nnTile / 2;
// @formatter:off
	static constexpr int nnLayers = {
			nnTile * nnTile * NUM_FEATURES - HDMV * (nnTile * nnTile / 2), // input

			10 // output
	};
// @formatter:on

	float3 *floatRaw;
	float3 *floatSd4;
	float3 *floatSd4Blured;
	float3 *floatSd2;
	float3 *floatSd2Blured;
	float3 *floatSd2Raw;
	float3 *floatSd4Raw;
	features_t *ftSd2, *ftRaw;
	ld_map_t *ldmSd2, *ldmRaw;
	ld_raw_t *gSd2, *gRaw;
	nnt_answers_t *nntAnswers;

	float dataMaximum;

	LibRaw &libraw;

	~DhtNn();
	DhtNn(LibRaw &_libraw);
	/*
	 * This functions probalby will use bitmap to store 1|0 values
	 */
	static inline void set_feature(features_t * fm, int idx, uchar value) {
		fm[0][idx] = value;
	}

	static inline uchar get_feature(features_t * fm, int idx) {
		return fm[0][idx];
	}

	/*
	 * Green aproximation with hue transition for red:
	 *      G_0 = (G_-1 + G_1) / 2 - (R_-2 + R_2 - 2 * R_0) / 4
	 */

	static inline float hue_aprox(float gl, float gr, float r0, float rl, float rr) {
		return (gl + gr) / 2.0f - (rl + rr - 2.0f * r0) / 4.0f;
	}

	static inline float hue_aprox_half(float gl, float r0, float rl) {
		return gl + (r0 - rl) / 2.0f;
	}

	static inline void inc_ld_ft(float g0, float gl, float r0, float rl, float maxL, float maxC, uchar &cnt) {
		if (fabs(gl - g0) <= maxL && maxC >= fabs(gl - rl - g0 + r0))
			++cnt;
	}
	static inline void min_vec_max_half(float g0, float gl, float gr, float r0, float rl, float rr, float &minVec,
			float &maxL, float &maxC) {
		const float ll = (fabs(gl - g0) + 1.0f) * (fabs(gl - rl - g0 + r0) + 1.0f);
		const float lr = (fabs(gr - g0) + 1.0f) * (fabs(gr - rr - g0 + r0) + 1.0f);
		const float lcvec = ll + lr;
		if (minVec < 0 || lcvec < minVec) {
			minVec = lcvec;
			if (ll < lr) {
				maxL = fabs(gl - g0);
				maxC = fabs(gl - rl - g0 + r0);
			} else {
				maxL = fabs(gr - g0);
				maxC = fabs(gr - rr - g0 + r0);
			}
		}
	}

	static inline float hue_aprox_vec(float gl, float gr, float r0, float rl, float rr, float &minVec, float &maxL,
			float &maxC) {
		float g0 = hue_aprox(gl, gr, r0, rl, rr);
		min_vec_max_half(g0, gl, gr, r0, rl, rr, minVec, maxL, maxC);
		return g0;
	}

	static inline float raw2float(unsigned short c) {
		if (c == (ushort) 0)
			return zeroColor;
		if (c <= (ushort) 16)
			return sqrt((float) c);
		return log2((float) c);
	}

	unsigned short float2raw(float f) {
		if (f < 0.0f)
			return 0u;
		if (f <= 4.0f)
			return round(f * f);
		float r = round(pow(2, f));
		if (r > dataMaximum)
			return dataMaximum;
		return round(r);
	}
	float float2nn(float f) {
		return f / dataMaximum;
	}
	float nn2float(float n) {
		return n * dataMaximum;
	}
	int frOffset(int x, int y) {
		return iWidth * y + x;
	}
	static inline float calc_gtype(calc_t t, float3 * const fr0, const int fw, uchar lc) {
		switch (t) {
		case CC_UD:
			return hue_aprox(fr0[-fw][1], fr0[+fw][1], fr0[0][lc], fr0[-2 * fw][lc], fr0[+2 * fw][lc]);
		case CC_LR:
			return hue_aprox(fr0[-1][1], fr0[+1][1], fr0[0][lc], fr0[-2][lc], fr0[+2][lc]);
		case CC_LU:
			return hue_aprox(fr0[-1][1], fr0[-fw][1], fr0[0][lc], fr0[-2][lc], fr0[-2 * fw][lc]);
		case CC_LD:
			return hue_aprox(fr0[-1][1], fr0[+fw][1], fr0[0][lc], fr0[-2][lc], fr0[+2 * fw][lc]);
		case CC_RU:
			return hue_aprox(fr0[+1][1], fr0[-fw][1], fr0[0][lc], fr0[+2][lc], fr0[-2 * fw][lc]);
		case CC_RD:
			return hue_aprox(fr0[+1][1], fr0[+fw][1], fr0[0][lc], fr0[+2][lc], fr0[+2 * fw][lc]);
		case CC_U:
			return hue_aprox_half(fr0[-fw][1], fr0[0][lc], fr0[-2 * fw][lc]);
		case CC_L:
			return hue_aprox_half(fr0[-1][1], fr0[0][lc], fr0[-2][lc]);
		case CC_D:
			return hue_aprox_half(fr0[+fw][1], fr0[0][lc], fr0[+2 * fw][lc]);
		case CC_R:
			return hue_aprox_half(fr0[+1][1], fr0[0][lc], fr0[+2][lc]);
		default:
			throw LIBRAW_EXCEPTION_IO_CORRUPT;
		}
	}
	void out_only_green();
	void out_image();
	void out_sd2();
	void scale_down2();
	void blur_down2();
	void out_sd4();
	void scale_down4();
	void blur_down4();
	void make_sd2fRaw();
	void make_ld_features(float3 * const fr, const int fw, const int fh, ld_raw_t * const ldr, ld_map_t * const ldm,
			features_t *fld);
	void out_ld_features(float3 * const frOut, const int fw, const int fh, ld_raw_t * const ldr, features_t *fld);
	void make_ld_greens(float * const ldOut, const int fw, const int fh, ld_raw_t * const ldr, features_t *fld);
	void make_ld_gmap_point(float3 * const fr, const int fw, ld_raw_t * const ldr, const uchar cc);
	void make_sg_features(float3 * const fr, const int fw, const int fh, ld_raw_t* const ldr, features_t *fld);
	void make_mg_features(const int fw, const int fh, features_t *fld);
	void make_nnt_answers(float3 * const fr, const int fw, const int fh);
	void make_features();
};

DhtNn::DhtNn(LibRaw& _libraw) :
		libraw(_libraw) {
	floatRaw = 0;
	floatSd4 = 0;
	floatSd4Raw = 0;
	floatSd4Blured = 0;
	floatSd2 = 0;
	floatSd2Raw = 0;
	floatSd2Blured = 0;
	ftSd2 = 0;
	ftRaw = 0;
	ldmSd2 = 0;
	ldmRaw = 0;
	ftRaw = 0;
	gSd2 = 0;
	gRaw = 0;
	nntAnswers = 0;
	iHeight = libraw.imgdata.sizes.iheight + margin * 2;
	iWidth = libraw.imgdata.sizes.iwidth + margin * 2;
	iHeight += (4 - (iHeight % 4)) % 4;
	iWidth += (4 - (iWidth % 4)) % 4;
	sd2Height = (libraw.imgdata.sizes.iheight + 1) / 2 + margin * 2;
	sd2Width = (libraw.imgdata.sizes.iwidth + 1) / 2 + margin * 2;
	sd2Height += (4 - sd2Height % 4) % 4;
	sd2Width += (4 - sd2Width % 4) % 4;
	sd4Height = (libraw.imgdata.sizes.iheight + 3) / 4 + margin * 2;
	sd4Width = (libraw.imgdata.sizes.iwidth + 3) / 4 + margin * 2;
	sd4Height += (4 - sd4Height % 4) % 4;
	sd4Width += (4 - sd4Width % 4) % 4;
	floatRaw = (float3*) malloc(iHeight * iWidth * sizeof(float3));
	if (!floatRaw)
		throw LIBRAW_EXCEPTION_ALLOC;
	floatSd4 = (float3*) malloc(sd4Height * sd4Width * sizeof(float3));
	if (!floatSd4)
		throw LIBRAW_EXCEPTION_ALLOC;
	floatSd4Raw = (float3*) malloc(sd4Height * sd4Width * sizeof(float3));
	if (!floatSd4Raw)
		throw LIBRAW_EXCEPTION_ALLOC;
	floatSd4Blured = (float3*) malloc(sd4Height * sd4Width * sizeof(float3));
	if (!floatSd4Blured)
		throw LIBRAW_EXCEPTION_ALLOC;
	floatSd2 = (float3*) malloc(sd2Height * sd2Width * sizeof(float3));
	if (!floatSd2)
		throw LIBRAW_EXCEPTION_ALLOC;
	floatSd2Raw = (float3*) malloc(sd2Height * sd2Width * sizeof(float3));
	if (!floatSd2Raw)
		throw LIBRAW_EXCEPTION_ALLOC;
	floatSd2Blured = (float3*) malloc(sd2Height * sd2Width * sizeof(float3));
	if (!floatSd2Blured)
		throw LIBRAW_EXCEPTION_ALLOC;
	ftSd2 = (features_t*) calloc(sd2Height * sd2Width, sizeof(features_t));
	if (!ftSd2)
		throw LIBRAW_EXCEPTION_ALLOC;
	ftRaw = (features_t*) calloc(iHeight * iWidth, sizeof(features_t));
	if (!ftRaw)
		throw LIBRAW_EXCEPTION_ALLOC;
	ldmSd2 = (ld_map_t*) calloc(sd2Height * sd2Width, sizeof(ld_map_t));
	if (!ldmSd2)
		throw LIBRAW_EXCEPTION_ALLOC;
	ldmRaw = (ld_map_t*) calloc(iHeight * iWidth, sizeof(ld_map_t));
	if (!ldmRaw)
		throw LIBRAW_EXCEPTION_ALLOC;
	gRaw = (ld_raw_t*) calloc(iHeight * iWidth, sizeof(ld_raw_t));
	if (!gRaw)
		throw LIBRAW_EXCEPTION_ALLOC;
	gSd2 = (ld_raw_t*) calloc(sd2Height * sd2Width, sizeof(ld_raw_t));
	if (!gSd2)
		throw LIBRAW_EXCEPTION_ALLOC;
	for (int i = 0; i < iHeight * iWidth; ++i)
		floatRaw[i][0] = floatRaw[i][1] = floatRaw[i][2] = zeroColor;
	dataMaximum = 0;
	for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i) {
		unsigned char col_cache[32];
		for (int j = 0; j < 32; ++j) {
			unsigned char l = libraw.COLOR(i, j);
			if (l == 3)
				l = 1;
			col_cache[j] = l;
		}
		for (int j = 0; j < libraw.imgdata.sizes.iwidth; ++j) {
			unsigned char l = col_cache[(unsigned) j % 32];
			unsigned short c = libraw.imgdata.image[i * libraw.imgdata.sizes.iwidth + j][l];
			if (c > dataMaximum)
				dataMaximum = c;
			floatRaw[frOffset(j + margin, i + margin)][l] = raw2float(c);
		}
	}
}

void DhtNn::out_only_green() {
	const ushort iwidth = libraw.imgdata.sizes.iwidth;
	for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i) {
		for (int j = 0; j < iwidth; ++j) {
			libraw.imgdata.image[i * iwidth + j][0] = 0;
			libraw.imgdata.image[i * iwidth + j][3] = libraw.imgdata.image[i * iwidth + j][1] = float2raw(
					floatRaw[frOffset(j + margin, i + margin)][1]);
			libraw.imgdata.image[i * iwidth + j][2] = 0;
		}
	}
}

void DhtNn::out_image() {
	const ushort iwidth = libraw.imgdata.sizes.iwidth;
	for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i) {
		for (int j = 0; j < iwidth; ++j) {
			libraw.imgdata.image[i * iwidth + j][0] = float2raw(floatRaw[frOffset(j + margin, i + margin)][0]);
			libraw.imgdata.image[i * iwidth + j][3] = libraw.imgdata.image[i * iwidth + j][1] = float2raw(
					floatRaw[frOffset(j + margin, i + margin)][1]);
			libraw.imgdata.image[i * iwidth + j][2] = float2raw(floatRaw[frOffset(j + margin, i + margin)][2]);
		}
	}
}

void DhtNn::out_sd4() {
	const int halfH = (libraw.imgdata.sizes.iheight + 3) / 4;
	for (int y = 0; y < sd4Height - 2 * margin; ++y) {
		for (int x = 0; x < sd4Width - 2 * margin; ++x) {
			floatRaw[frOffset(x + margin, y + margin)][0] = floatSd4[(y + margin) * sd4Width + x + margin][0];
			floatRaw[frOffset(x + margin, y + margin)][1] = floatSd4[(y + margin) * sd4Width + x + margin][1];
			floatRaw[frOffset(x + margin, y + margin)][2] = floatSd4[(y + margin) * sd4Width + x + margin][2];
			floatRaw[frOffset(x + margin, y + margin + halfH)][0] =
					floatSd4Blured[(y + margin) * sd4Width + x + margin][0];
			floatRaw[frOffset(x + margin, y + margin + halfH)][1] =
					floatSd4Blured[(y + margin) * sd4Width + x + margin][1];
			floatRaw[frOffset(x + margin, y + margin + halfH)][2] =
					floatSd4Blured[(y + margin) * sd4Width + x + margin][2];
		}
	}
}

void DhtNn::out_sd2() {
	const int halfH = (libraw.imgdata.sizes.iheight + 1) / 2;
	const int halfW = (libraw.imgdata.sizes.iwidth + 1) / 2;
	for (int y = 0; y < sd2Height - 2 * margin; ++y) {
		for (int x = 0; x < sd2Width - 2 * margin; ++x) {
			floatRaw[frOffset(x + margin + halfW, y + margin)][0] = floatSd2[(y + margin) * sd2Width + x + margin][0];
			floatRaw[frOffset(x + margin + halfW, y + margin)][1] = floatSd2[(y + margin) * sd2Width + x + margin][1];
			floatRaw[frOffset(x + margin + halfW, y + margin)][2] = floatSd2[(y + margin) * sd2Width + x + margin][2];
			floatRaw[frOffset(x + margin + halfW, y + margin + halfH)][0] = floatSd2Blured[(y + margin) * sd2Width + x
					+ margin][0];
			floatRaw[frOffset(x + margin + halfW, y + margin + halfH)][1] = floatSd2Blured[(y + margin) * sd2Width + x
					+ margin][1];
			floatRaw[frOffset(x + margin + halfW, y + margin + halfH)][2] = floatSd2Blured[(y + margin) * sd2Width + x
					+ margin][2];
		}
	}
}

void DhtNn::scale_down2() {
	const int qH = (libraw.imgdata.sizes.iheight + 1) / 2;
	const int qW = (libraw.imgdata.sizes.iwidth + 1) / 2;
	const int sds = sd2Width * sd2Height;
	for (int o = 0; o < sds; ++o) {
		for (uchar i = 0; i < 3; ++i) {
			floatSd2[o][i] = zeroColor;
			floatSd2Blured[o][i] = zeroColor;
		}
	}
	uchar js = libraw.COLOR(0, 0) & 1; // js == 1 only for green
	uchar lc = libraw.COLOR(0, js); // non-green line color
	for (int y = 0; y < qH; ++y) {
		int srY = y * 2 + margin;
		for (int x = 0; x < qW; ++x) {
			int srX = x * 2 + margin;
			uchar cc = lc; // current non-green color
			uchar gp = js; // non-green color first position
			float3 sdp = { 0.0f, 0.0f, 0.0f };
			for (int l = 0; l < 2; ++l, cc ^= 2, gp ^= 1)
				for (int k = 0; k < 2; ++k) {
					uchar pc = (uchar)(k & 1) == gp ? cc : 1;
					sdp[pc] += floatRaw[frOffset(srX + k, srY + l)][pc];
				}
			floatSd2[(y + margin) * sd2Width + x + margin][0] = sdp[0];
			floatSd2[(y + margin) * sd2Width + x + margin][1] = sdp[1] / 2.0f;
			floatSd2[(y + margin) * sd2Width + x + margin][2] = sdp[2];
		}
	}
	blur_down2();
}
void DhtNn::blur_down2() {
	float blurKernel[3][3] = { { 1, 2, 1 }, { 2, 36, 2 }, { 1, 2, 1 }, };
	const float norm = 48.0f;
	for (int y = 0; y < sd2Height - 2 * margin; ++y) {
		for (int x = 0; x < sd2Width - 2 * margin; ++x) {
			float3 sdp = { 0.0f, 0.0f, 0.0f };
			for (uchar ci = 0; ci < 3; ++ci) {
				sdp[ci] += floatSd2[(y + margin - 1) * sd2Width + x + margin - 1][ci] * blurKernel[0][0];
				sdp[ci] += floatSd2[(y + margin - 1) * sd2Width + x + margin][ci] * blurKernel[0][1];
				sdp[ci] += floatSd2[(y + margin - 1) * sd2Width + x + margin + 1][ci] * blurKernel[0][2];
				sdp[ci] += floatSd2[(y + margin) * sd2Width + x + margin - 1][ci] * blurKernel[1][0];
				sdp[ci] += floatSd2[(y + margin) * sd2Width + x + margin][ci] * blurKernel[1][1];
				sdp[ci] += floatSd2[(y + margin) * sd2Width + x + margin + 1][ci] * blurKernel[1][2];
				sdp[ci] += floatSd2[(y + margin + 1) * sd2Width + x + margin - 1][ci] * blurKernel[2][0];
				sdp[ci] += floatSd2[(y + margin + 1) * sd2Width + x + margin][ci] * blurKernel[2][1];
				sdp[ci] += floatSd2[(y + margin + 1) * sd2Width + x + margin + 1][ci] * blurKernel[2][2];
				sdp[ci] /= norm;
			}
			floatSd2Blured[(y + margin) * sd2Width + x + margin][0] = sdp[0];
			floatSd2Blured[(y + margin) * sd2Width + x + margin][1] = sdp[1];
			floatSd2Blured[(y + margin) * sd2Width + x + margin][2] = sdp[2];
		}
	}
}

void DhtNn::scale_down4() {
	const int qH = (libraw.imgdata.sizes.iheight + 3) / 4;
	const int qW = (libraw.imgdata.sizes.iwidth + 3) / 4;
	const int sds = sd4Width * sd4Height;
	for (int o = 0; o < sds; ++o) {
		for (uchar i = 0; i < 3; ++i) {
			floatSd4[o][i] = zeroColor;
			floatSd4Blured[o][i] = zeroColor;
		}
	}
	uchar js = libraw.COLOR(0, 0) & 1; // js == 1 only for green
	uchar lc = libraw.COLOR(0, js); // non-green line color
	for (int y = 0; y < qH; ++y) {
		int srY = y * 4 + margin;
		for (int x = 0; x < qW; ++x) {
			int srX = x * 4 + margin;
			uchar cc = lc; // current non-green color
			uchar gp = js; // non-green color first position
			float3 sdp = { 0.0f, 0.0f, 0.0f };
			for (int l = 0; l < 4; ++l, cc ^= 2, gp ^= 1)
				for (int k = 0; k < 4; ++k) {
					uchar pc = (uchar)(k & 1) == gp ? cc : 1;
					sdp[pc] += floatRaw[frOffset(srX + k, srY + l)][pc];
				}
			floatSd4[(y + margin) * sd4Width + x + margin][0] = sdp[0] / 4.0f;
			floatSd4[(y + margin) * sd4Width + x + margin][1] = sdp[1] / 8.0f;
			floatSd4[(y + margin) * sd4Width + x + margin][2] = sdp[2] / 4.0f;
		}
	}
	blur_down4();
}

void DhtNn::blur_down4() {
	float blurKernel[3][3] = { { 1, 2, 1 }, { 2, 12, 2 }, { 1, 2, 1 }, };
	const float norm = 24.0f;
	for (int y = 0; y < sd4Height - 2 * margin; ++y) {
		for (int x = 0; x < sd4Width - 2 * margin; ++x) {
			float3 sdp = { 0.0f, 0.0f, 0.0f };
			for (uchar ci = 0; ci < 3; ++ci) {
				sdp[ci] += floatSd4[(y + margin - 1) * sd4Width + x + margin - 1][ci] * blurKernel[0][0];
				sdp[ci] += floatSd4[(y + margin - 1) * sd4Width + x + margin][ci] * blurKernel[0][1];
				sdp[ci] += floatSd4[(y + margin - 1) * sd4Width + x + margin + 1][ci] * blurKernel[0][2];
				sdp[ci] += floatSd4[(y + margin) * sd4Width + x + margin - 1][ci] * blurKernel[1][0];
				sdp[ci] += floatSd4[(y + margin) * sd4Width + x + margin][ci] * blurKernel[1][1];
				sdp[ci] += floatSd4[(y + margin) * sd4Width + x + margin + 1][ci] * blurKernel[1][2];
				sdp[ci] += floatSd4[(y + margin + 1) * sd4Width + x + margin - 1][ci] * blurKernel[2][0];
				sdp[ci] += floatSd4[(y + margin + 1) * sd4Width + x + margin][ci] * blurKernel[2][1];
				sdp[ci] += floatSd4[(y + margin + 1) * sd4Width + x + margin + 1][ci] * blurKernel[2][2];
				sdp[ci] /= norm;
			}
			floatSd4Blured[(y + margin) * sd4Width + x + margin][0] = sdp[0];
			floatSd4Blured[(y + margin) * sd4Width + x + margin][1] = sdp[1];
			floatSd4Blured[(y + margin) * sd4Width + x + margin][2] = sdp[2];
		}
	}
}

void DhtNn::make_sd2fRaw() {
	const int sds = sd2Width * sd2Height;
	for (int o = 0; o < sds; ++o) {
		for (uchar i = 0; i < 3; ++i) {
			floatSd2Raw[o][i] = zeroColor;
		}
	}
	for (int y = 0; y < sd2Height - 2 * margin; ++y) {
		for (int x = 0; x < sd2Width - 2 * margin; ++x) {
			uchar cc = libraw.COLOR(y, x);
			int o = (y + margin) * sd2Width + x + margin;
			floatSd2Raw[o][cc] = floatSd2Blured[o][cc];
		}
	}
}

void DhtNn::make_ld_gmap_point(float3* const fr, const int fw, ld_raw_t* const ldr, const uchar cc) {
	float cc0 = fr[0][cc];
	float ccU = fr[-fw * 2][cc];
	float ccR = fr[+2][cc];
	float ccD = fr[+fw * 2][cc];
	float ccL = fr[-2][cc];
	float gU = fr[-fw][1];
	float gR = fr[+1][1];
	float gD = fr[+fw][1];
	float gL = fr[-1][1];
	float minVec = -1.0f;
	float maxL = 0.0f, maxC = 0.0f;
	ldr[0][LD_UD] = hue_aprox_vec(gU, gD, cc0, ccU, ccD, minVec, maxL, maxC);
	ldr[0][LD_LR] = hue_aprox_vec(gL, gR, cc0, ccL, ccR, minVec, maxL, maxC);
	ldr[0][LD_LU] = hue_aprox_vec(gL, gU, cc0, ccL, ccU, minVec, maxL, maxC);
	ldr[0][LD_LD] = hue_aprox_vec(gL, gD, cc0, ccL, ccD, minVec, maxL, maxC);
	ldr[0][LD_RU] = hue_aprox_vec(gR, gU, cc0, ccR, ccU, minVec, maxL, maxC);
	ldr[0][LD_RD] = hue_aprox_vec(gR, gD, cc0, ccR, ccD, minVec, maxL, maxC);
	ldr[0][HDMV] = maxL + JND;
	ldr[0][HDMV + 1] = maxC + JND;
}

void DhtNn::make_ld_features(float3* const fr, const int fw, const int fh, ld_raw_t* const ldr, ld_map_t * const ldm,
		features_t *fld) {
	const int sds = fw * fh;
	for (int o = 0; o < sds; ++o) {
		ldr[o][LD_UD] = ldr[o][LD_LR] = ldr[o][LD_LU] = ldr[o][LD_LD] = ldr[o][LD_RU] = ldr[o][LD_RD] = fr[o][1];
	}
	for (int y = margin; y < fh - margin; ++y) {
		uchar js = libraw.COLOR(y - margin, 0) & 1; // js == 1 only for green
		uchar lc = libraw.COLOR(y - margin, js); // non-green line color
		for (int x = margin + js; x < fw - margin; x += 2) { // loop for non-green points
			int o = y * fw + x;
			make_ld_gmap_point(fr + o, fw, ldr + o, lc);
		}
	}
	// There will be no LD features for greens
	for (int y = margin; y < fh - margin; ++y) {
		uchar js = libraw.COLOR(y - margin, 0) & 1; // js == 1 only for green
		uchar lc = libraw.COLOR(y - margin, js); // non-green line color
		for (int x = margin + js; x < fw - margin; x += 2) { // loop for non-green points
			int o = y * fw + x;
			float cc0 = fr[o][lc];
			float ccU = fr[o - fw * 2][lc];
			float ccR = fr[o + 2][lc];
			float ccD = fr[o + fw * 2][lc];
			float ccL = fr[o - 2][lc];
			float gU = fr[o - fw][1];
			float gR = fr[o + 1][1];
			float gD = fr[o + fw][1];
			float gL = fr[o - 1][1];
			float hdmvL = ldr[o][HDMV], hdmvC = ldr[o][HDMV + 1];
			uchar mls = 0;
			for (int i = 0; i < HDMV; ++i) {
				float g0 = ldr[o][i];
				inc_ld_ft(g0, gL, cc0, (ccL + cc0) / 2, hdmvL, hdmvC, ldm[o][i]);
				inc_ld_ft(g0, gR, cc0, (ccR + cc0) / 2, hdmvL, hdmvC, ldm[o][i]);
				inc_ld_ft(g0, gU, cc0, (ccU + cc0) / 2, hdmvL, hdmvC, ldm[o][i]);
				inc_ld_ft(g0, gD, cc0, (ccD + cc0) / 2, hdmvL, hdmvC, ldm[o][i]);
				if (ldm[o][i] > mls)
					mls = ldm[o][i];
			}
			for (int i = 0; i < HDMV; ++i) {
				if (mls == ldm[o][i])
					set_feature(fld + o, i, 1);
			}
		}
	}
}

void DhtNn::out_ld_features(float3 * const frOut, const int fw, const int fh, ld_raw_t* const ldr, features_t *fld) {
	float fdm = raw2float(dataMaximum);
	for (int y = margin; y < fh - margin; ++y) {
		uchar js = libraw.COLOR(y - margin, 0) & 1; // js == 1 only for green
		uchar lc = libraw.COLOR(y - margin, js); // non-green line color
		for (int x = margin; x < fw - margin; ++x) {
			int o = y * fw + x;
//			frOut[o][lc ^ 2] = zeroColor;
			if ((uchar)((x - margin) & 1) == js) {
				// non-green
				float num = 0, sum = 0;
				for (int i = 0; i < HDMV; ++i) {
					if (get_feature(fld + o, i)) {
						++num;
						sum += ldr[o][i];
					}
				}
				frOut[o][1] = sum / num;
			} else {
				// green
//				frOut[o][0] = frOut[o][2] = zeroColor;
			}
		}
	}
}

void DhtNn::make_ld_greens(float * const ldOut, const int fw, const int fh, ld_raw_t* const ldr, features_t *fld) {
	for (int y = margin; y < fh - margin; ++y) {
		uchar js = libraw.COLOR(y - margin, 0) & 1; // js == 1 only for green
		uchar lc = libraw.COLOR(y - margin, js); // non-green line color
		for (int x = margin; x < fw - margin; ++x) {
			int o = y * fw + x;
			if ((uchar)((x - margin) & 1) == js) {
				// non-green
				float num = 0, sum = 0;
				for (int i = 0; i < HDMV; ++i) {
					if (get_feature(fld + o, i)) {
						++num;
						sum += ldr[o][i];
					}
				}
				ldOut[o] = sum / num;
			} else {
				ldOut[o] = ldr[o][0];
			}
		}
	}
}

void DhtNn::make_sg_features(float3 * const fr, const int fw, const int fh, ld_raw_t* const ldr, features_t *fld) {
	float * ldOut = (float*) malloc(fh * fw * sizeof(float));
	if (!ldOut)
		throw LIBRAW_EXCEPTION_ALLOC;
	const int sds = fw * fh;
	for (int o = 0; o < sds; ++o) {
		ldOut[o] = zeroColor;
	}
	make_ld_greens(ldOut, fw, fh, ldr, fld); // use previously determined greens from LD features
	for (int y = margin; y < fh - margin; ++y) {
		uchar js = libraw.COLOR(y - margin, 0) & 1; // js == 1 only for green
		uchar lc = libraw.COLOR(y - margin, js); // non-green line color
		uchar ac = lc ^ 2;
		for (int x = margin; x < fw - margin; ++x) {
			int o = y * fw + x;
			if ((uchar)((x - margin) & 1) == js) {
				// non-green
//				 SG_UD = 6, // SG == single gradient
//				 SG_LR = 7,
//				 SG_DU = 8,
//				 SG_RL = 9,
				float cc0 = fr[o][lc];
				float ccU = fr[o - fw * 2][lc];
				float ccR = fr[o + 2][lc];
				float ccD = fr[o + fw * 2][lc];
				float ccL = fr[o - 2][lc];
				float gU = fr[o - fw][1];
				float gR = fr[o + 1][1];
				float gD = fr[o + fw][1];
				float gL = fr[o - 1][1];
				float g0h = hue_aprox(gL, gR, cc0, ccL, ccR);
				float g0v = hue_aprox(gU, gD, cc0, ccU, ccD);
				float lch = (fabs(gL - g0h) + 1.0f) * (fabs(gL - ccL - g0h + cc0) + 1.0f)
						+ (fabs(gR - g0h) + 1.0f) * (fabs(gR - ccR - g0h + cc0) + 1.0f);
				float lcv = (fabs(gU - g0v) + 1.0f) * (fabs(gU - ccU - g0v + cc0) + 1.0f)
						+ (fabs(gD - g0v) + 1.0f) * (fabs(gD - ccD - g0v + cc0) + 1.0f);
				float lcd = (lch < lcv) ? lcv / lch : lch / lcv;
				uchar fud, fdu, flr, frl;
				fud = gU < gD || fabs(gU - gD) < JND * 2.0f;
				fdu = gD < gU || fabs(gU - gD) < JND * 2.0f;
				flr = gL < gR || fabs(gL - gR) < JND * 2.0f;
				frl = gR < gL || fabs(gL - gR) < JND * 2.0f;
				if (lcd < TLC) {
					set_feature(fld + o, SG_UD, fud);
					set_feature(fld + o, SG_LR, flr);
					set_feature(fld + o, SG_DU, fdu);
					set_feature(fld + o, SG_RL, frl);
				} else {
					if (lch < lcv) {
						set_feature(fld + o, SG_LR, flr);
						set_feature(fld + o, SG_RL, frl);
					} else {
						set_feature(fld + o, SG_UD, fud);
						set_feature(fld + o, SG_DU, fdu);
					}
				}
				/*
				 // ccUL |      | ccU  |      | ccUR
				 //------+------+------+------+------
				 //      | acUL |  gU  |  |
				 //------+------+------+------+------
				 // ccL  |  gL  | cc0  |  gR  | ccR
				 //------+------+------+------+------
				 //      | acDL |  gD  | acDR |
				 //------+------+------+------+------
				 // ccDL |      | ccD  |      | ccDR
				 */
				float ccUL = fr[o - fw * 2 - 2][lc];
				float ccUR = fr[o - fw * 2 + 2][lc];
				float ccDL = fr[o + fw * 2 - 2][lc];
				float ccDR = fr[o + fw * 2 + 2][lc];
				float acUL = fr[o - fw - 1][ac];
				float acUR = fr[o - fw + 1][ac];
				float acDL = fr[o + fw - 1][ac];
				float acDR = fr[o + fw + 1][ac];
				float cc0ULF = (ccUL + ccU + ccL + cc0) / 4.0f;
				float cc0URF = (ccUR + ccU + ccR + cc0) / 4.0f;
				float cc0DLF = (ccDL + ccD + ccL + cc0) / 4.0f;
				float cc0DRF = (ccDR + ccD + ccR + cc0) / 4.0f;
				float ac0ULDR = (acUL + acDR + 2.0f * cc0 - cc0ULF - cc0DRF) / 2.0f;
				float ac0DLUR = (acDL + acUR + 2.0f * cc0 - cc0DLF - cc0URF) / 2.0f;
				float culdr = (fabs(acUL - ac0ULDR) + 1.0f) * (fabs(cc0ULF - cc0) + 1.0f)
						+ (fabs(acDR - ac0ULDR) + 1.0f) * (fabs(cc0DRF - cc0) + 1.0f);
				float cdlur = (fabs(acDL - ac0DLUR) + 1.0f) * (fabs(cc0DLF - cc0) + 1.0f)
						+ (fabs(acUR - ac0DLUR) + 1.0f) * (fabs(cc0URF - cc0) + 1.0f);
				float cvd = (culdr < cdlur) ? cdlur / culdr : culdr / cdlur;
//				SG_ULDR = 10, // diagonal directed gradient
//				SG_DRUL = 11,
//				SG_URDL = 12,
//				SG_DLUR = 13,
				uchar fuldr, fdlur, furdl, fdrul;
				fuldr = acUL < acDR || fabs(acUL - acDR) < JND * 2.0f;
				fdrul = acDR < acUL || fabs(acDR - acUL) < JND * 2.0f;
				furdl = acUR < acDL || fabs(acUR - acDL) < JND * 2.0f;
				fdlur = acDL < acUR || fabs(acDL - acUR) < JND * 2.0f;
				if (cvd < TLC) {
					set_feature(fld + o, SG_ULDR, fuldr);
					set_feature(fld + o, SG_DRUL, fdrul);
					set_feature(fld + o, SG_URDL, furdl);
					set_feature(fld + o, SG_DLUR, fdlur);
				} else {
					if (culdr < cdlur) {
						set_feature(fld + o, SG_ULDR, fuldr);
						set_feature(fld + o, SG_DRUL, fdrul);
					} else {
						set_feature(fld + o, SG_URDL, furdl);
						set_feature(fld + o, SG_DLUR, fdlur);
					}
				}
			} else {
				// green
				float gU = ldOut[o - fw];
				float gD = ldOut[o + fw];
				float gL = ldOut[o - 1];
				float gR = ldOut[o + 1];
				float g0 = fr[o][1];
				float lch = (fabs(gL - g0) + 1.0f) + (fabs(gR - g0) + 1.0f);
				float lcv = (fabs(gU - g0) + 1.0f) + (fabs(gD - g0) + 1.0f);
				float lcd = (lch < lcv) ? lcv / lch : lch / lcv;
				uchar fud, fdu, flr, frl;
				fud = gU < gD || fabs(gU - gD) < JND * 2.0f;
				fdu = gD < gU || fabs(gU - gD) < JND * 2.0f;
				flr = gL < gR || fabs(gL - gR) < JND * 2.0f;
				frl = gR < gL || fabs(gL - gR) < JND * 2.0f;
				if (lcd < TLC) {
					set_feature(fld + o, SG_UD, fud);
					set_feature(fld + o, SG_LR, flr);
					set_feature(fld + o, SG_DU, fdu);
					set_feature(fld + o, SG_RL, frl);
				} else {
					if (lch < lcv) {
						set_feature(fld + o, SG_LR, flr);
						set_feature(fld + o, SG_RL, frl);
					} else {
						set_feature(fld + o, SG_UD, fud);
						set_feature(fld + o, SG_DU, fdu);
					}
				}
				float gUL = fr[o - fw - 1][1];
				float gDL = fr[o + fw - 1][1];
				float gUR = fr[o - fw + 1][1];
				float gDR = fr[o + fw + 1][1];
				float luldr = (fabs(gUL - g0) + 1.0f) + (fabs(gDR - g0) + 1.0f);
				float lurdl = (fabs(gUR - g0) + 1.0f) + (fabs(gDL - g0) + 1.0f);
				float lld = (luldr < lurdl) ? lurdl / luldr : luldr / lurdl;
				uchar fuldr, fdlur, furdl, fdrul;
				fuldr = gUL < gDR || fabs(gUL - gDR) < JND * 2.0f;
				fdrul = gDR < gUL || fabs(gDR - gUL) < JND * 2.0f;
				furdl = gUR < gDL || fabs(gUR - gDL) < JND * 2.0f;
				fdlur = gDL < gUR || fabs(gDL - gUR) < JND * 2.0f;
				if (lld < TLC) {
					set_feature(fld + o, SG_ULDR, fuldr);
					set_feature(fld + o, SG_DRUL, fdrul);
					set_feature(fld + o, SG_URDL, furdl);
					set_feature(fld + o, SG_DLUR, fdlur);
				} else {
					if (luldr < lurdl) {
						set_feature(fld + o, SG_ULDR, fuldr);
						set_feature(fld + o, SG_DRUL, fdrul);
					} else {
						set_feature(fld + o, SG_URDL, furdl);
						set_feature(fld + o, SG_DLUR, fdlur);
					}
				}
			}
		}
	}
	free(ldOut);
}

void DhtNn::make_mg_features(const int fw, const int fh, features_t *fld) {
	struct {
		int sg, mg;
	} mmg[] = { { SG_UD, MG_UD }, { SG_LR, MG_LR }, { SG_DU, MG_DU }, { SG_RL, MG_RL } };
	for (int y = margin; y < fh - margin; ++y) {
		uchar js = libraw.COLOR(y - margin, 0) & 1; // js == 1 only for green
		uchar lc = libraw.COLOR(y - margin, js); // non-green line color
		for (int x = margin; x < fw - margin; ++x) {
			int o = y * fw + x;
			for (auto m : mmg) {
				int num = 0;
				if (get_feature(fld + o, m.sg)) {
					if (get_feature(fld + o - 1, m.sg)) // left
						++num;
					if (get_feature(fld + o + 1, m.sg)) // right
						++num;
					if (get_feature(fld + o - fw, m.sg)) // up
						++num;
					if (get_feature(fld + o + fw, m.sg)) // down
						++num;
					if (num > 1)
						set_feature(fld + o, m.mg, 1);
				}
			}
			if ((uchar)((x - margin) & 1) == js) {
				// non-green
				set_feature(fld + o, CI_R + lc, 1);
			} else {
				set_feature(fld + o, CI_G, 1);
			}
		}
	}
}

void DhtNn::make_features() {

}

void DhtNn::make_nnt_answers(float3 * const frReal, const int fw, const int fh) {
	if (!nntAnswers) {
		nntAnswers = (nnt_answers_t*) calloc(fh * fw, sizeof(nnt_answers_t));
		if (!nntAnswers)
			throw LIBRAW_EXCEPTION_ALLOC;
	}
	for (int y = margin; y < fh - margin; ++y) {
		uchar js = libraw.COLOR(y - margin, 0) & 1; // js == 1 only for green
		uchar lc = libraw.COLOR(y - margin, js); // non-green line color
		for (int x = margin + js; x < fw - margin; x += 2) { // loop for non-green points
			int o = y * fw + x;
			float realG = frReal[o][1];
			float3 * const fr0 = frReal + o;
			nnt_answers_t aG;
			aG[CC_UD] = hue_aprox(fr0[-fw][1], fr0[+fw][1], fr0[0][lc], fr0[-2 * fw][lc], fr0[+2 * fw][lc]);
			aG[CC_LR] = hue_aprox(fr0[-1][1], fr0[+1][1], fr0[0][lc], fr0[-2][lc], fr0[+2][lc]);
			aG[CC_LU] = hue_aprox(fr0[-1][1], fr0[-fw][1], fr0[0][lc], fr0[-2][lc], fr0[-2 * fw][lc]);
			aG[CC_LD] = hue_aprox(fr0[-1][1], fr0[+fw][1], fr0[0][lc], fr0[-2][lc], fr0[+2 * fw][lc]);
			aG[CC_RU] = hue_aprox(fr0[+1][1], fr0[-fw][1], fr0[0][lc], fr0[+2][lc], fr0[-2 * fw][lc]);
			aG[CC_RD] = hue_aprox(fr0[+1][1], fr0[+fw][1], fr0[0][lc], fr0[+2][lc], fr0[+2 * fw][lc]);
			aG[CC_U] = hue_aprox_half(fr0[-fw][1], fr0[0][lc], fr0[-2 * fw][lc]);
			aG[CC_L] = hue_aprox_half(fr0[-1][1], fr0[0][lc], fr0[-2][lc]);
			aG[CC_D] = hue_aprox_half(fr0[+fw][1], fr0[0][lc], fr0[+2 * fw][lc]);
			aG[CC_R] = hue_aprox_half(fr0[+1][1], fr0[0][lc], fr0[+2][lc]);
			int bestI = CC_UD;
			float bestDiff = fabs(realG - aG[CC_UD]);
			for (int ai = 0; ai < sizeof(aG) / sizeof(aG[0]); ++ai) {
				float diff = fabs(realG - aG[ai]);
				if (diff < bestDiff) {
					bestI = ai;
					bestDiff = diff;
				}
				if (diff < JND)
					nntAnswers[o][ai] = 0.90f + 0.1f * ((JND - diff) / JND);
				else
					nntAnswers[o][ai] = 0.0f;
			}
			nntAnswers[o][bestI] = 1.0f; // at least one "right" answer must be there
		}
	}
}

DhtNn::~DhtNn() {
	free(floatRaw);
	free(floatSd2);
	free(floatSd2Blured);
	free(floatSd2Raw);
	free(floatSd4);
	free(floatSd4Blured);
	free(floatSd4Raw);
	free(ldmSd2);
	free(ldmRaw);
	free(ftSd2);
	free(ftRaw);
	free(gSd2);
	free(gRaw);
	free(nntAnswers);
}

void LibRaw::dht_nn_interpolate() {
#ifdef DCRAW_VERBOSE
// hide it later
#endif
	printf("DHT-NN interpolating\n");
	DhtNn dht(*this);
	dht.scale_down2();
//	dht.scale_down4();
	dht.make_sd2fRaw();
	dht.make_ld_features(dht.floatSd2Raw, dht.sd2Width, dht.sd2Height, dht.gSd2, dht.ldmSd2, dht.ftSd2);
	dht.make_sg_features(dht.floatSd2Raw, dht.sd2Width, dht.sd2Height, dht.gSd2, dht.ftSd2);
	dht.make_mg_features(dht.sd2Width, dht.sd2Height, dht.ftSd2);
//	dht.out_ld_features(dht.floatSd2, dht.sd2Width, dht.sd2Height, dht.gSd2, dht.ftSd2);
//	dht.out_sd2();
//	dht.out_sd4();
	dht.out_image();
//	dht.out_only_green();
}

