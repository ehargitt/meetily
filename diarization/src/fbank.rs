//! Kaldi-compatible log-mel filterbank, a port of `torchaudio.compliance.kaldi.fbank` with the
//! settings WeSpeaker's recipe uses at inference: 25 ms / 10 ms frames, 80 mel bins from 20 Hz to
//! Nyquist, Hamming window, power spectrum, `snip_edges = true`, no dither, input scaled to the
//! 16-bit range, then per-utterance mean normalisation (CMN).
//!
//! The ResNet34-LM ONNX export takes `(1, T, 80)` features rather than raw audio, so the features
//! are computed here. Sherpa-onnx's feature settings (Povey window, no CMN) measurably worsen
//! speaker confusion with this model.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub(crate) const NUM_BINS: usize = 80;
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const LOW_FREQ: f32 = 20.0;
const PREEMPH: f32 = 0.97;
/// WeSpeaker models are trained on 16-bit integer sample values.
const INPUT_SCALE: f32 = 32768.0;

pub(crate) struct Fbank {
    padded: usize,
    window: Vec<f32>,
    /// Sparse triangular mel filters: (first FFT bin, weights).
    mel: Vec<(usize, Vec<f32>)>,
    fft: Arc<dyn RealToComplex<f32>>,
}

fn mel_scale(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

impl Fbank {
    pub(crate) fn new() -> Self {
        let n = FRAME_LENGTH;
        let padded = n.next_power_of_two();
        let window: Vec<f32> = (0..n)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0);
                (0.54 - 0.46 * a.cos()) as f32
            })
            .collect();

        // Mel banks exactly as kaldi.get_mel_banks (vtln_warp = 1), top edge at Nyquist.
        let sample_rate = crate::SAMPLE_RATE as f32;
        let high = sample_rate * 0.5;
        let num_fft_bins = padded / 2;
        let bin_width = sample_rate / padded as f32;
        let mel_lo = mel_scale(LOW_FREQ);
        let mel_hi = mel_scale(high);
        let delta = (mel_hi - mel_lo) / (NUM_BINS as f32 + 1.0);
        let mut mel = Vec::with_capacity(NUM_BINS);
        for b in 0..NUM_BINS {
            let left = mel_lo + b as f32 * delta;
            let center = mel_lo + (b as f32 + 1.0) * delta;
            let right = mel_lo + (b as f32 + 2.0) * delta;
            let mut first = None;
            let mut weights = Vec::new();
            for k in 0..num_fft_bins {
                let m = mel_scale(bin_width * k as f32);
                let up = (m - left) / (center - left);
                let down = (right - m) / (right - center);
                let v = up.min(down).max(0.0);
                if v > 0.0 {
                    let first = *first.get_or_insert(k);
                    // A triangle has no holes; padding keeps the weights contiguous regardless.
                    weights.resize(k - first, 0.0);
                    weights.push(v);
                }
            }
            mel.push((first.unwrap_or(0), weights));
        }
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(padded);
        Self { padded, window, mel, fft }
    }

    /// Features for `wave` (16 kHz, [-1, 1]) as row-major `(frames, NUM_BINS)`, plus the frame
    /// count. Input shorter than one frame yields no frames.
    pub(crate) fn compute(&self, wave: &[f32]) -> (Vec<f32>, usize) {
        if wave.len() < FRAME_LENGTH {
            return (Vec::new(), 0);
        }
        let num_frames = 1 + (wave.len() - FRAME_LENGTH) / FRAME_SHIFT;
        let len = FRAME_LENGTH;

        let mut out = vec![0f32; num_frames * NUM_BINS];
        let mut frame = vec![0f32; self.padded];
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0f32; self.padded / 2 + 1];
        for f in 0..num_frames {
            let start = f * FRAME_SHIFT;
            for (dst, src) in frame[..len].iter_mut().zip(&wave[start..start + len]) {
                *dst = src * INPUT_SCALE;
            }
            let mean = frame[..len].iter().sum::<f32>() / len as f32;
            frame[..len].iter_mut().for_each(|x| *x -= mean);
            for j in (1..len).rev() {
                frame[j] -= PREEMPH * frame[j - 1];
            }
            frame[0] -= PREEMPH * frame[0];
            for (x, w) in frame[..len].iter_mut().zip(&self.window) {
                *x *= w;
            }
            frame[len..].iter_mut().for_each(|x| *x = 0.0);
            self.fft
                .process_with_scratch(&mut frame, &mut spec, &mut scratch)
                .expect("buffers are sized by the planner");
            for (p, c) in power.iter_mut().zip(spec.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }
            let row = &mut out[f * NUM_BINS..(f + 1) * NUM_BINS];
            for (dst, (first, weights)) in row.iter_mut().zip(&self.mel) {
                let energy: f32 = weights.iter().zip(&power[*first..]).map(|(a, b)| a * b).sum();
                *dst = energy.max(f32::EPSILON).ln();
            }
        }
        subtract_mean(&mut out, num_frames);
        (out, num_frames)
    }
}

/// Per-utterance cepstral mean normalisation over row-major `(frames, NUM_BINS)` features.
fn subtract_mean(feats: &mut [f32], num_frames: usize) {
    let mut mean = [0f64; NUM_BINS];
    for row in feats.chunks_exact(NUM_BINS) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    for m in &mut mean {
        *m /= num_frames as f64;
    }
    for row in feats.chunks_exact_mut(NUM_BINS) {
        for (v, m) in row.iter_mut().zip(&mean) {
            *v -= *m as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_match_kaldi() {
        let fb = Fbank::new();
        let (feats, nf) = fb.compute(&vec![0.1; 16000]);
        assert_eq!(nf, 98); // 1 + (16000 - 400) / 160
        assert_eq!(feats.len(), 98 * NUM_BINS);
        assert_eq!(fb.compute(&[0.1; 399]).1, 0);
        assert_eq!(fb.compute(&[0.1; 400]).1, 1);
    }

    #[test]
    fn cmn_leaves_zero_mean_per_bin() {
        let wave: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 97.0 * t).sin()
            })
            .collect();
        let (feats, nf) = Fbank::new().compute(&wave);
        assert!(nf > 0);
        for b in 0..NUM_BINS {
            let mean = feats.chunks_exact(NUM_BINS).map(|row| row[b] as f64).sum::<f64>() / nf as f64;
            assert!(mean.abs() < 1e-4, "bin {b} mean {mean}");
        }
        assert!(feats.iter().all(|x| x.is_finite()));
    }
}
