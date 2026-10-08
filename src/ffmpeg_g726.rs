// SPDX-License-Identifier: LGPL-2.1-or-later
// Port of FFmpeg 2da55bf libavcodec/g726.c: the decoder (g726_decode,
// inverse_quant, the Float11 arithmetic, the rate tables, g726_reset).
// Copyright (c) 2004 Roman Shaposhnik.
//
// This file is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 2.1, or (at your option) any later version.
// It is distributed WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE-LGPL.

//! G.726 decoded as FFmpeg decodes it: its rendition of the Recommendation's
//! section 4, which departs from the ITU reference in places (it says so in
//! its comments), so its output is not the conformance vectors'. The
//! registry decoder's linear output uses it; [`crate::g726`] keeps the
//! ITU-exact codec for the A-law / µ-law interfaces and the encoder.

use crate::g726::Rate;

/// G.726's 11-bit float: sign, 4-bit exponent, 6-bit mantissa.
#[derive(Clone, Copy, Debug, Default)]
struct Float11 {
    sign: u8,
    exp: u8,
    mant: u8,
}

/// `av_log2_16bit`: floor(log2) of the low 16 bits, 0 for 0.
fn log2_16bit(v: u32) -> u32 {
    let (v, n) = if v & 0xff00 != 0 { ((v >> 8) & 0xff, 8) } else { (v & 0xff, 0) };
    n + if v == 0 { 0 } else { 31 - v.leading_zeros() }
}

/// `i2f`.
fn i2f(i: i32) -> Float11 {
    let sign = u8::from(i < 0);
    let i = i.wrapping_abs();
    let exp = log2_16bit(i as u32) + u32::from(i != 0);
    let mant = if i != 0 { (((i << 6) >> exp) & 0xff) as u8 } else { 1 << 5 };
    Float11 { sign, exp: exp as u8, mant }
}

/// `mult`, its result truncated to 16 bits as FFmpeg's `int16_t` is.
fn mult(f1: Float11, f2: Float11) -> i16 {
    let exp = i32::from(f1.exp) + i32::from(f2.exp);
    let mut res = (i32::from(f1.mant) * i32::from(f2.mant) + 0x30) >> 4;
    res = if exp > 19 { res << (exp - 19) } else { res >> (19 - exp) };
    (if f1.sign ^ f2.sign != 0 { -res } else { res }) as i16
}

fn sgn(value: i32) -> i32 {
    if value < 0 {
        -1
    } else {
        1
    }
}

struct Tables {
    iquant: &'static [i16],
    w: &'static [i16],
    f: &'static [u8],
}

const IQUANT16: [i16; 4] = [116, 365, 365, 116];
const W16: [i16; 4] = [-22, 439, 439, -22];
const F16: [u8; 4] = [0, 7, 7, 0];

const IQUANT24: [i16; 8] = [i16::MIN, 135, 273, 373, 373, 273, 135, i16::MIN];
const W24: [i16; 8] = [-4, 30, 137, 582, 582, 137, 30, -4];
const F24: [u8; 8] = [0, 1, 2, 7, 7, 2, 1, 0];

const IQUANT32: [i16; 16] = [i16::MIN, 4, 135, 213, 273, 323, 373, 425, 425, 373, 323, 273, 213, 135, 4, i16::MIN];
const W32: [i16; 16] = [-12, 18, 41, 64, 112, 198, 355, 1122, 1122, 355, 198, 112, 64, 41, 18, -12];
const F32: [u8; 16] = [0, 0, 0, 1, 1, 1, 3, 7, 7, 3, 1, 1, 1, 0, 0, 0];

const IQUANT40: [i16; 32] = [
    i16::MIN, -66, 28, 104, 169, 224, 274, 318, 358, 395, 429, 459, 488, 514, 539, 566, 566, 539, 514, 488, 459, 429,
    395, 358, 318, 274, 224, 169, 104, 28, -66, i16::MIN,
];
const W40: [i16; 32] = [
    14, 14, 24, 39, 40, 41, 58, 100, 141, 179, 219, 280, 358, 440, 529, 696, 696, 529, 440, 358, 280, 219, 179, 141,
    100, 58, 41, 40, 39, 24, 14, 14,
];
const F40: [u8; 32] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

fn tables(rate: Rate) -> Tables {
    match rate.bits() {
        2 => Tables { iquant: &IQUANT16, w: &W16, f: &F16 },
        3 => Tables { iquant: &IQUANT24, w: &W24, f: &F24 },
        4 => Tables { iquant: &IQUANT32, w: &W32, f: &F32 },
        _ => Tables { iquant: &IQUANT40, w: &W40, f: &F40 },
    }
}

/// One channel of FFmpeg's G.726 decoder (`G726Context`).
pub struct Decoder {
    tbls: Tables,
    code_size: u32,
    sr: [Float11; 2],
    dq: [Float11; 6],
    a: [i32; 2],
    b: [i32; 6],
    pk: [i32; 2],
    ap: i32,
    yu: i32,
    yl: i32,
    dms: i32,
    dml: i32,
    td: i32,
    se: i32,
    sez: i32,
    y: i32,
}

impl Decoder {
    /// A decoder at `rate`, as FFmpeg's context starts (zeroed, then
    /// `g726_reset`).
    pub fn new(rate: Rate) -> Self {
        let mut d = Decoder {
            tbls: tables(rate),
            code_size: u32::from(rate.bits()),
            sr: Default::default(),
            dq: Default::default(),
            a: [0; 2],
            b: [0; 6],
            pk: [0; 2],
            ap: 0,
            yu: 0,
            yl: 0,
            dms: 0,
            dml: 0,
            td: 0,
            se: 0,
            sez: 0,
            y: 0,
        };
        d.reset();
        d
    }

    /// `g726_reset`: the fields the zeroed context starts from otherwise.
    fn reset(&mut self) {
        for i in 0..2 {
            self.sr[i].mant = 1 << 5;
            self.pk[i] = 1;
        }
        for dq in &mut self.dq {
            dq.mant = 1 << 5;
        }
        self.yu = 544;
        self.yl = 34816;
        self.y = 544;
    }

    /// `inverse_quant`, truncated to 16 bits as FFmpeg's is.
    fn inverse_quant(&self, i: usize) -> i32 {
        let dql = i32::from(self.tbls.iquant[i]) + (self.y >> 2);
        let dex = (dql >> 7) & 0xf;
        let dqt = (1 << 7) + (dql & 0x7f);
        i32::from((if dql < 0 { 0 } else { (dqt << dex) >> 7 }) as i16)
    }

    /// `g726_decode`: one code word to a 16-bit sample.
    pub fn decode(&mut self, code: u8) -> i16 {
        let i = usize::from(code);
        let i_sig = i32::from(code) >> (self.code_size - 1);
        let mut dq = self.inverse_quant(i);

        // Transition detect.
        let ylint = self.yl >> 15;
        let ylfrac = (self.yl >> 10) & 0x1f;
        let thr2 = if ylint > 9 { 0x1f << 10 } else { (0x20 + ylfrac) << ylint };
        let tr = self.td == 1 && dq > ((3 * thr2) >> 2);

        if i_sig != 0 {
            dq = -dq;
        }
        let re_signal = i32::from((self.se + dq) as i16);

        // Second order predictor coefficients A2 and A1.
        let pk0 = if self.sez + dq != 0 { sgn(self.sez + dq) } else { 0 };
        let dq0 = if dq != 0 { sgn(dq) } else { 0 };
        if tr {
            self.a = [0; 2];
            self.b = [0; 6];
        } else {
            let fa1 = ((-self.a[0] * self.pk[0] * pk0) >> 5).clamp(-256, 255);
            self.a[1] += 128 * pk0 * self.pk[1] + fa1 - (self.a[1] >> 7);
            self.a[1] = self.a[1].clamp(-12288, 12288);
            self.a[0] += 64 * 3 * pk0 * self.pk[0] - (self.a[0] >> 8);
            self.a[0] = self.a[0].clamp(-(15360 - self.a[1]), 15360 - self.a[1]);
            for k in 0..6 {
                self.b[k] += 128 * dq0 * sgn(-i32::from(self.dq[k].sign)) - (self.b[k] >> 8);
            }
        }

        // Dq, Sr and Pk.
        self.pk[1] = self.pk[0];
        self.pk[0] = if pk0 != 0 { pk0 } else { 1 };
        self.sr[1] = self.sr[0];
        self.sr[0] = i2f(re_signal);
        for k in (1..6).rev() {
            self.dq[k] = self.dq[k - 1];
        }
        self.dq[0] = i2f(dq);
        self.dq[0].sign = i_sig as u8;

        self.td = i32::from(self.a[1] < -11776);

        // Ap.
        let f = i32::from(self.tbls.f[i]);
        self.dms += (f << 4) + ((-self.dms) >> 5);
        self.dml += (f << 4) + ((-self.dml) >> 7);
        if tr {
            self.ap = 256;
        } else {
            self.ap += (-self.ap) >> 4;
            if self.y <= 1535 || self.td != 0 || ((self.dms << 2) - self.dml).abs() >= (self.dml >> 3) {
                self.ap += 0x20;
            }
        }

        // Yu and Yl.
        self.yu = (self.y + i32::from(self.tbls.w[i]) + ((-self.y) >> 5)).clamp(544, 5120);
        self.yl += self.yu + ((-self.yl) >> 6);

        // Next Y.
        let al = if self.ap >= 256 { 1 << 6 } else { self.ap >> 2 };
        self.y = (self.yl + (self.yu - (self.yl >> 6)) * al) >> 6;

        // Next SE and SEZ.
        self.se = 0;
        for k in 0..6 {
            self.se += i32::from(mult(i2f(self.b[k] >> 2), self.dq[k]));
        }
        self.sez = self.se >> 1;
        for k in 0..2 {
            self.se += i32::from(mult(i2f(self.a[k] >> 2), self.sr[k]));
        }
        self.se >>= 1;

        (re_signal * 4).clamp(-0xffff, 0xffff) as i16
    }
}
