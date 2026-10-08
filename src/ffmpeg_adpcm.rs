// SPDX-License-Identifier: LGPL-2.1-or-later
// Port of FFmpeg 2da55bf libavcodec/adpcm.c: the ADPCM_IMA_QT block state
// rule, adpcm_ima_oki_expand_nibble, and the ADPCM_CT and ADPCM_SBPRO_2/3/4
// decoders (adpcm_ct_expand_nibble, adpcm_sbpro_expand_nibble, their
// get_nb_samples and decode cases, adpcm_flush's start state).
// Copyright (c) 2001-2003 The FFmpeg project.
//
// This file is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 2.1, or (at your option) any later version.
// It is distributed WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE-LGPL.

//! Decoding as FFmpeg's ADPCM decoders do, where that differs from the
//! formats' own descriptions:
//!
//! - QuickTime IMA (`adpcm_ima_qt`) keeps each channel's predictor and step
//!   index from block to block. A block header only replaces them when its
//!   step index differs, or its predictor (whose low 7 bits the header
//!   cannot hold) is more than 127 away. A header step index above 88 is
//!   an error.
//! - OKI (`adpcm_dialogic`, FFmpeg's `adpcm_ima_oki`) steps by
//!   `((2 * magnitude + 1) * step) >> 3`, which rounds differently from the
//!   sum of shifted steps in the Dialogic application note.
//! - Creative ADPCM (`adpcm_ct`) and the Sound Blaster Pro 2-, 2.6- and
//!   4-bit codecs of VOC files (`adpcm_sbpro_2`/`_3`/`_4`), which have no
//!   other description.

use oxideav_core::{
    AudioFormat, AudioFrame, CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result, SampleFormat,
};

use crate::dialogic;
use crate::ima_qt::{QT_BLOCK_SIZE, QT_SAMPLES_PER_BLOCK};
use crate::ima_wav::ima_expand_nibble;
use crate::tables::{OKI_INDEX_ADJUST, OKI_STEP_SIZE};

/// Canonical codec id for Creative ADPCM (WAV tag 0x0200, VOC codec 0x0200).
pub const CODEC_ID_CT: &str = "adpcm_ct";
/// Canonical codec id for Sound Blaster Pro 2-bit ADPCM (VOC codec 3).
pub const CODEC_ID_SBPRO_2: &str = "adpcm_sbpro_2";
/// Canonical codec id for Sound Blaster Pro 2.6-bit ADPCM (VOC codec 2).
pub const CODEC_ID_SBPRO_3: &str = "adpcm_sbpro_3";
/// Canonical codec id for Sound Blaster Pro 4-bit ADPCM (VOC codec 1).
pub const CODEC_ID_SBPRO_4: &str = "adpcm_sbpro_4";

// ───────────────────────── QuickTime IMA ─────────────────────────

/// One channel's running QuickTime IMA state.
#[derive(Clone, Copy, Debug, Default)]
pub struct QtChannel {
    predictor: i32,
    step_index: i32,
}

/// Decodes one QuickTime IMA packet of `state.len()` 34-byte channel blocks
/// with each channel's running state, as FFmpeg does: interleaved samples,
/// 64 per channel.
pub fn decode_qt_block(data: &[u8], state: &mut [QtChannel]) -> Result<Vec<i16>> {
    let channels = state.len();
    if data.len() < QT_BLOCK_SIZE * channels {
        return Err(Error::invalid(format!(
            "adpcm_ima_qt: need {} bytes for {channels}ch, got {}",
            QT_BLOCK_SIZE * channels,
            data.len()
        )));
    }
    let mut out = vec![0i16; QT_SAMPLES_PER_BLOCK * channels];
    for (ch, cs) in state.iter_mut().enumerate() {
        let block = &data[ch * QT_BLOCK_SIZE..(ch + 1) * QT_BLOCK_SIZE];
        let header = i32::from(i16::from_be_bytes([block[0], block[1]]));
        let step_index = header & 0x7F;
        let predictor = header & !0x7F;
        if cs.step_index != step_index || (predictor - cs.predictor).abs() > 0x7F {
            cs.step_index = step_index;
            cs.predictor = predictor;
        }
        if !(0..=88).contains(&cs.step_index) {
            return Err(Error::invalid(format!("adpcm_ima_qt: step index {} on channel {ch}", cs.step_index)));
        }
        for (i, &byte) in block[2..].iter().enumerate() {
            out[2 * i * channels + ch] = ima_expand_nibble(&mut cs.predictor, &mut cs.step_index, byte & 0x0F);
            out[(2 * i + 1) * channels + ch] = ima_expand_nibble(&mut cs.predictor, &mut cs.step_index, byte >> 4);
        }
    }
    Ok(out)
}

// ───────────────────────── OKI ─────────────────────────

/// `adpcm_ima_oki_expand_nibble`: the 12-bit predictor widened to 16 bits.
fn oki_expand_nibble(c: &mut dialogic::Channel, nibble: u8) -> i16 {
    let step = i32::from(OKI_STEP_SIZE[c.step_index.clamp(0, 48) as usize]);
    let delta = i32::from(nibble & 7);
    let diff = ((2 * delta + 1) * step) >> 3;
    let predictor = if nibble & 8 != 0 { c.predictor - diff } else { c.predictor + diff };
    c.predictor = predictor.clamp(-2048, 2047);
    c.step_index = (c.step_index + OKI_INDEX_ADJUST[usize::from(nibble & 7)]).clamp(0, 48);
    (c.predictor * 16) as i16
}

/// Decodes an OKI packet as FFmpeg's `adpcm_ima_oki`: two nibbles a byte,
/// in `order`, alternating over the channels.
pub fn decode_oki_packet(packet: &[u8], state: &mut [dialogic::Channel], order: dialogic::NibbleOrder) -> Vec<i16> {
    let channels = state.len();
    let mut out = Vec::with_capacity(packet.len() * 2);
    if channels == 0 {
        return out;
    }
    let mut ch = 0;
    for &byte in packet {
        let (first, second) = match order {
            dialogic::NibbleOrder::HiFirst => (byte >> 4, byte & 0x0F),
            dialogic::NibbleOrder::LoFirst => (byte & 0x0F, byte >> 4),
        };
        for nibble in [first, second] {
            out.push(oki_expand_nibble(&mut state[ch], nibble));
            ch = (ch + 1) % channels;
        }
    }
    out
}

// ───────────────────────── Creative / Sound Blaster ─────────────────────────

/// FFmpeg's ADPCM table of step multipliers (`ff_adpcm_AdaptationTable`).
const ADAPTATION: [i32; 16] = [230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230];

/// The Creative and Sound Blaster codecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Ct,
    Sbpro2,
    Sbpro3,
    Sbpro4,
}

/// One channel's state (the fields of FFmpeg's `ADPCMChannelStatus` these
/// codecs use).
#[derive(Clone, Copy, Debug, Default)]
struct CreativeChannel {
    predictor: i32,
    step: i32,
    /// Sound Blaster: 0 until the first packet, whose first byte per channel
    /// is a raw sample (only channel 0's is read).
    step_index: i32,
}

/// `adpcm_ct_expand_nibble`.
fn ct_expand_nibble(c: &mut CreativeChannel, nibble: u8) -> i16 {
    let delta = i32::from(nibble & 7);
    let diff = ((2 * delta + 1) * c.step) >> 3;
    c.predictor = ((c.predictor * 254) >> 8) + if nibble & 8 != 0 { -diff } else { diff };
    c.predictor = c.predictor.clamp(-32768, 32767);
    c.step = ((ADAPTATION[usize::from(nibble & 7)] * c.step) >> 8).clamp(511, 32767);
    c.predictor as i16
}

/// `adpcm_sbpro_expand_nibble`.
fn sbpro_expand_nibble(c: &mut CreativeChannel, nibble: u8, size: u32, shift: u32) -> i16 {
    let nibble = i32::from(nibble);
    let sign = nibble & (1 << (size - 1));
    let delta = nibble & ((1 << (size - 1)) - 1);
    let diff = delta << (7 + c.step + shift as i32);
    c.predictor = (c.predictor + if sign != 0 { -diff } else { diff }).clamp(-16384, 16256);
    if delta >= 2 * size as i32 - 3 && c.step < 3 {
        c.step += 1;
    } else if delta == 0 && c.step > 0 {
        c.step -= 1;
    }
    c.predictor as i16
}

/// Creative ADPCM and Sound Blaster Pro ADPCM, decoded as FFmpeg does:
/// interleaved 16-bit output.
pub struct CreativeDecoder {
    codec_id: CodecId,
    kind: Kind,
    channels: usize,
    sample_rate: Option<u32>,
    state: [CreativeChannel; 2],
    pending: Option<(Option<i64>, u32, Vec<u8>)>,
    eof: bool,
}

/// The decoder for `params` (`adpcm_ct` or `adpcm_sbpro_*`), mono or
/// stereo.
pub fn make_creative_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let kind = match params.codec_id.as_str() {
        CODEC_ID_CT => Kind::Ct,
        CODEC_ID_SBPRO_2 => Kind::Sbpro2,
        CODEC_ID_SBPRO_3 => Kind::Sbpro3,
        CODEC_ID_SBPRO_4 => Kind::Sbpro4,
        other => return Err(Error::unsupported(format!("adpcm: {other} is not a Creative codec"))),
    };
    let channels = usize::from(params.channels.unwrap_or(1));
    if !(1..=2).contains(&channels) {
        return Err(Error::unsupported(format!("{}: {channels} channels (1 or 2)", params.codec_id.as_str())));
    }
    let mut decoder = CreativeDecoder {
        codec_id: params.codec_id.clone(),
        kind,
        channels,
        sample_rate: params.sample_rate,
        state: Default::default(),
        pending: None,
        eof: false,
    };
    decoder.start();
    Ok(Box::new(decoder))
}

impl CreativeDecoder {
    /// `adpcm_flush`: everything zero, Creative ADPCM's step 511.
    fn start(&mut self) {
        self.state = Default::default();
        if self.kind == Kind::Ct {
            for c in &mut self.state {
                c.step = 511;
            }
        }
    }

    /// One packet's samples per channel and the interleaved samples, as
    /// FFmpeg's `get_nb_samples` counts and its decode cases write them
    /// (slots a case does not write stay 0); `None` when FFmpeg refuses the
    /// packet.
    fn decode(&mut self, data: &[u8]) -> Option<(usize, Vec<i16>)> {
        let ch = self.channels;
        let st = usize::from(ch == 2);
        let raw = self.kind != Kind::Ct && self.state[0].step_index == 0;
        let mut payload = data;
        let mut nb_samples = 0;
        if raw {
            if payload.len() < ch {
                return None;
            }
            nb_samples += 1;
            payload = &payload[ch..];
        }
        let per_byte = match self.kind {
            Kind::Ct | Kind::Sbpro4 => 2,
            Kind::Sbpro3 => 3,
            Kind::Sbpro2 => 4,
        };
        nb_samples += payload.len() * per_byte / ch;
        let mut out = vec![0i16; nb_samples * ch];
        let mut at = 0;
        let mut put = |s: i16| {
            out[at] = s;
            at += 1;
        };
        if raw {
            // The first byte (per channel) is a raw 8-bit sample.
            for &byte in &data[..ch] {
                put((128 * (i32::from(byte) - 0x80)) as i16);
            }
            self.state[0].step_index = 1;
        }
        let left = nb_samples - usize::from(raw);
        let bytes = match self.kind {
            Kind::Ct | Kind::Sbpro4 => left >> (1 - st),
            Kind::Sbpro3 => (left << st) / 3,
            Kind::Sbpro2 => left >> (2 - st),
        };
        for &byte in &payload[..bytes] {
            let s = &mut self.state;
            match self.kind {
                Kind::Ct => {
                    put(ct_expand_nibble(&mut s[0], byte >> 4));
                    put(ct_expand_nibble(&mut s[st], byte & 0x0F));
                }
                Kind::Sbpro4 => {
                    put(sbpro_expand_nibble(&mut s[0], byte >> 4, 4, 0));
                    put(sbpro_expand_nibble(&mut s[st], byte & 0x0F, 4, 0));
                }
                Kind::Sbpro3 => {
                    put(sbpro_expand_nibble(&mut s[0], byte >> 5, 3, 0));
                    put(sbpro_expand_nibble(&mut s[0], (byte >> 2) & 0x07, 3, 0));
                    put(sbpro_expand_nibble(&mut s[0], byte & 0x03, 2, 0));
                }
                Kind::Sbpro2 => {
                    put(sbpro_expand_nibble(&mut s[0], byte >> 6, 2, 2));
                    put(sbpro_expand_nibble(&mut s[st], (byte >> 4) & 0x03, 2, 2));
                    put(sbpro_expand_nibble(&mut s[0], (byte >> 2) & 0x03, 2, 2));
                    put(sbpro_expand_nibble(&mut s[st], byte & 0x03, 2, 2));
                }
            }
        }
        Some((nb_samples, out))
    }
}

impl Decoder for CreativeDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    /// Interleaved 16-bit samples, whatever the container declares (a VOC
    /// file says 8 bits for its 8-bit-coded ADPCM).
    fn output_audio_format(&self) -> Option<AudioFormat> {
        Some(AudioFormat { sample_format: SampleFormat::S16, sample_rate: self.sample_rate?, channels: self.channels as u16 })
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other("adpcm decoder: call receive_frame before sending another packet"));
        }
        let (samples, pcm) = self
            .decode(&packet.data)
            .ok_or_else(|| Error::invalid(format!("{}: packet shorter than its raw first sample", self.codec_id.as_str())))?;
        let data = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        self.pending = Some((packet.pts, samples as u32, data));
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.pending.take() {
            Some((pts, samples, data)) => Ok(Frame::Audio(AudioFrame { samples, pts, data: vec![data] })),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.pending = None;
        self.eof = false;
        self.start();
        Ok(())
    }
}
