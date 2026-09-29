//! Time-aligns microphone and system audio for mixing.
//!
//! Both streams are placed on one sample timeline. A chunk is normally
//! appended where its stream's previous chunk ended, so callback-size and
//! callback-time jitter never insert silence. Its capture timestamp decides
//! the position instead when the stream (re)appears: its first chunk, the
//! first chunk after it was starved and padded, or a jump of more than
//! `RESYNC_MS` (a stall, a rebuild). A late or resumed stream therefore lands
//! at its true time — the gap before it is silence, samples older than the
//! current window are dropped — and no fixed offset survives. Slow clock drift
//! between two devices is corrected by dropping samples from whichever stream
//! runs ahead, so drift correction never inserts silence either.

use std::collections::VecDeque;

use log::{debug, info};

use crate::audio::recording_state::DeviceType;

/// Mixing window length. Windows are only mixed once both streams have one.
pub(super) const MIX_WINDOW_MS: u32 = 600;

/// A stream is starved (and padded with silence) once the other stream's
/// timeline runs this far ahead of it: it stalled, died, or does not exist.
const STARVATION_MS: u32 = 200;

/// A chunk whose capture time is further than this from where continuity puts
/// it re-anchors the stream at its capture time.
const RESYNC_MS: u32 = 100;

/// Relative clock drift between the streams tolerated before correcting it.
const DRIFT_TOLERANCE_MS: u32 = 20;

/// Smoothing of each stream's capture-time error, so timestamp jitter (callback
/// latency, resampler buffering) does not trigger drift corrections.
const LAG_SMOOTHING: f64 = 0.05;

/// One stream's buffered samples and its place on the shared timeline.
#[derive(Default)]
struct StreamTimeline {
    samples: VecDeque<f32>,
    /// Timeline index just past the last buffered sample; `None` until the
    /// stream delivers.
    end: Option<i64>,
    /// Smoothed capture index minus timeline index of this stream's samples.
    lag: f64,
    /// Place the next chunk by its capture time rather than appending it.
    reanchor: bool,
}

impl StreamTimeline {
    /// Delivering and placed by continuity; its `lag` is meaningful.
    fn is_tracking(&self) -> bool {
        self.end.is_some() && !self.reanchor
    }
}

/// Ring buffer for synchronized audio mixing.
///
/// The front of both buffers is always the timeline index `head`, so a window
/// taken from both is time-aligned. A window is mixed when both streams hold a
/// full window, or when one does and the other is starved.
pub(super) struct AudioMixerRingBuffer {
    mic: StreamTimeline,
    system: StreamTimeline,
    /// Timeline index of the next window's first sample.
    head: Option<i64>,
    sample_rate: u32,
    window_size_samples: usize,
}

impl AudioMixerRingBuffer {
    pub(super) fn new(sample_rate: u32) -> Self {
        let window_size_samples = (sample_rate as u64 * MIX_WINDOW_MS as u64 / 1000) as usize;
        info!("🔊 Ring buffer initialized: window={}ms ({} samples), aligned on capture time",
              MIX_WINDOW_MS, window_size_samples);
        Self {
            mic: StreamTimeline::default(),
            system: StreamTimeline::default(),
            head: None,
            sample_rate,
            window_size_samples,
        }
    }

    fn samples_for_ms(&self, ms: u32) -> i64 {
        self.sample_rate as i64 * ms as i64 / 1000
    }

    fn streams_mut(&mut self, device_type: DeviceType) -> (&mut StreamTimeline, &StreamTimeline) {
        match device_type {
            DeviceType::Microphone => (&mut self.mic, &self.system),
            DeviceType::System => (&mut self.system, &self.mic),
        }
    }

    /// Queue a chunk whose last sample was captured at `capture_secs`
    /// (seconds of active recording time).
    pub(super) fn add_samples(&mut self, device_type: DeviceType, samples: &[f32], capture_secs: f64) {
        if samples.is_empty() {
            return;
        }
        let n = samples.len() as i64;
        let capture_start = (capture_secs * self.sample_rate as f64).round() as i64 - n;
        let head = *self.head.get_or_insert(capture_start);
        let resync = self.samples_for_ms(RESYNC_MS) as f64;
        let drift_tolerance = self.samples_for_ms(DRIFT_TOLERANCE_MS) as f64;

        let (stream, other) = self.streams_mut(device_type);
        // Where the other stream's samples of the same capture time sit.
        let reference_lag = if other.is_tracking() { other.lag } else { stream.lag };

        let continuous_end = stream.end.filter(|_| !stream.reanchor);
        let position = match continuous_end {
            Some(prev_end) if ((capture_start - prev_end) as f64 - stream.lag).abs() <= resync => {
                stream.lag += ((capture_start - prev_end) as f64 - stream.lag) * LAG_SMOOTHING;
                // Drift: this stream's samples were captured earlier than the other's
                // at the same timeline index. Drop the surplus instead of padding the other.
                let surplus = if other.is_tracking() { other.lag - stream.lag } else { 0.0 };
                if surplus > drift_tolerance {
                    let dropped = (surplus.round() as i64).min(n);
                    stream.lag += dropped as f64;
                    debug!("{:?} mix drift correction: dropped {} samples", device_type, dropped);
                    prev_end - dropped
                } else {
                    prev_end
                }
            }
            _ => {
                if let Some(prev_end) = stream.end {
                    debug!("{:?} mix re-anchored on capture time ({} samples from continuity)",
                           device_type, capture_start - prev_end);
                }
                stream.lag = reference_lag;
                stream.reanchor = false;
                capture_start - reference_lag.round() as i64
            }
        };

        // Place the chunk: pad a gap before it, drop what overlaps buffered or
        // already-mixed audio.
        let end = stream.end.unwrap_or(head).max(head);
        if position > end {
            stream.samples.extend(std::iter::repeat(0.0).take((position - end) as usize));
        }
        let skip = (end - position).clamp(0, n) as usize;
        stream.samples.extend(samples[skip..].iter().copied());
        stream.end = Some(end.max(position + n));
    }

    fn timeline_end(&self, stream: &StreamTimeline) -> i64 {
        stream.end.unwrap_or(self.head.unwrap_or(0))
    }

    fn is_starved(&self, stream: &StreamTimeline) -> bool {
        let latest = self.timeline_end(&self.mic).max(self.timeline_end(&self.system));
        latest - self.timeline_end(stream) > self.samples_for_ms(STARVATION_MS)
    }

    /// The next mixable (mic, system) window, if any.
    pub(super) fn extract_window(&mut self) -> Option<(Vec<f32>, Vec<f32>)> {
        let window = self.window_size_samples;
        let mic_full = self.mic.samples.len() >= window;
        let system_full = self.system.samples.len() >= window;
        let ready = (mic_full && system_full)
            || (mic_full && self.is_starved(&self.system))
            || (system_full && self.is_starved(&self.mic));
        if !ready {
            return None;
        }
        Some(self.take(window))
    }

    /// Everything still buffered (less than a window per stream), with the
    /// shorter stream padded to the longer. Used when recording stops.
    pub(super) fn drain_remaining(&mut self) -> Option<(Vec<f32>, Vec<f32>)> {
        let len = self.mic.samples.len().max(self.system.samples.len());
        if len == 0 {
            return None;
        }
        Some(self.take(len))
    }

    /// Take `len` samples from both streams, padding a short one with silence.
    fn take(&mut self, len: usize) -> (Vec<f32>, Vec<f32>) {
        let head = self.head.unwrap_or(0) + len as i64;
        self.head = Some(head);
        (Self::take_padded(&mut self.mic, len, head), Self::take_padded(&mut self.system, len, head))
    }

    fn take_padded(stream: &mut StreamTimeline, len: usize, new_head: i64) -> Vec<f32> {
        let available = stream.samples.len().min(len);
        let mut window: Vec<f32> = stream.samples.drain(..available).collect();
        if available < len {
            window.resize(len, 0.0);
            // Padded past its data: place its next chunk by capture time.
            if let Some(end) = stream.end.as_mut() {
                *end = (*end).max(new_head);
                stream.reanchor = true;
            }
        }
        window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const MIC_LEVEL: f32 = 0.25;
    const SYS_LEVEL: f32 = 0.5;

    /// Deterministic jitter (no RNG dependency needed).
    struct Lcg(u64);
    impl Lcg {
        fn next_in(&mut self, lo: usize, hi: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (self.0 >> 33) as usize % (hi - lo + 1)
        }
    }

    /// A simulated capture stream. Each sample's value is its true capture
    /// time as a 48 kHz sample index plus one (0 is reserved for padding), so
    /// the mixed windows show exactly how the streams were aligned.
    struct SimStream {
        device_type: DeviceType,
        /// Device clock relative to wall time (1.005 = 0.5% fast).
        clock: f64,
        start_ms: u64,
        /// Wall-time ranges in which the stream captures nothing.
        stalls: Vec<(u64, u64)>,
        next_sample: u64,
        pending: Vec<f32>,
        next_len: usize,
        len_range: (usize, usize),
        /// Maximum callback latency added to the reported capture time.
        latency_ms: usize,
        jitter: Lcg,
    }

    impl SimStream {
        fn new(device_type: DeviceType, seed: u64) -> Self {
            Self {
                device_type,
                clock: 1.0,
                start_ms: 0,
                stalls: Vec::new(),
                next_sample: 0,
                pending: Vec::new(),
                next_len: 480,
                len_range: (480, 480),
                latency_ms: 0,
                jitter: Lcg(seed),
            }
        }

        fn sizes(mut self, lo: usize, hi: usize) -> Self {
            self.len_range = (lo, hi);
            self.next_len = self.jitter.next_in(lo, hi);
            self
        }

        /// Deliver every buffer the device completed by `now_ms`.
        fn pump(&mut self, now_ms: u64, ring: &mut AudioMixerRingBuffer) {
            if now_ms < self.start_ms {
                return;
            }
            let wall_rate = RATE as f64 / 1000.0;
            let device_rate = wall_rate * self.clock;
            let due = ((now_ms - self.start_ms) as f64 * device_rate) as u64;
            while self.next_sample < due {
                let wall_ms = self.start_ms as f64 + self.next_sample as f64 / device_rate;
                self.next_sample += 1;
                if self.stalls.iter().any(|&(a, b)| wall_ms >= a as f64 && wall_ms < b as f64) {
                    continue;
                }
                self.pending.push((wall_ms * wall_rate).round() as f32 + 1.0);
            }
            while self.pending.len() >= self.next_len {
                let chunk: Vec<f32> = self.pending.drain(..self.next_len).collect();
                let latency = if self.latency_ms > 0 { self.jitter.next_in(0, self.latency_ms) } else { 0 };
                let capture_secs = (now_ms + latency as u64) as f64 / 1000.0;
                ring.add_samples(self.device_type, &chunk, capture_secs);
                self.next_len = self.jitter.next_in(self.len_range.0, self.len_range.1);
            }
        }
    }

    /// Per mixed sample: the (mic, system) capture-time values.
    fn run(mut mic: SimStream, mut sys: SimStream, seconds: u64) -> Vec<(f32, f32)> {
        let mut ring = AudioMixerRingBuffer::new(RATE);
        let mut mixed = Vec::new();
        for ms in 0..seconds * 1_000 {
            if ms % 2 == 0 {
                mic.pump(ms, &mut ring);
                sys.pump(ms, &mut ring);
            } else {
                sys.pump(ms, &mut ring);
                mic.pump(ms, &mut ring);
            }
            while let Some((m, s)) = ring.extract_window() {
                mixed.extend(m.into_iter().zip(s));
            }
            let most = ring.window_size_samples * 2;
            assert!(ring.mic.samples.len() <= most && ring.system.samples.len() <= most, "buffers grew at {ms} ms");
        }
        mixed
    }

    /// Largest |mic − system| capture-time offset, in ms, where both carry audio.
    fn max_offset_ms(mixed: &[(f32, f32)]) -> f64 {
        mixed
            .iter()
            .filter(|(m, s)| *m != 0.0 && *s != 0.0)
            .map(|(m, s)| (*m as f64 - *s as f64).abs() * 1000.0 / RATE as f64)
            .fold(0.0, f64::max)
    }

    /// Mic capture times (ms) of mixed samples where `pick` found padding.
    fn padded_at_ms(mixed: &[(f32, f32)], pick: impl Fn(&(f32, f32)) -> (f32, f32)) -> Vec<f64> {
        mixed
            .iter()
            .map(pick)
            .filter(|(padded, reference)| *padded == 0.0 && *reference != 0.0)
            .map(|(_, reference)| reference as f64 * 1000.0 / RATE as f64)
            .collect()
    }

    fn system_padding(mixed: &[(f32, f32)]) -> Vec<f64> {
        padded_at_ms(mixed, |&(m, s)| (s, m))
    }

    fn mic_padding(mixed: &[(f32, f32)]) -> Vec<f64> {
        padded_at_ms(mixed, |&(m, s)| (m, s))
    }

    /// Padding after the first window. When the second stream first appears,
    /// its callback latency can leave a few milliseconds before its anchor.
    fn steady_state_padding(mixed: &[(f32, f32)]) -> usize {
        let after_first_window = |t: &f64| *t > MIX_WINDOW_MS as f64;
        mic_padding(mixed).iter().filter(|t| after_first_window(t)).count()
            + system_padding(mixed).iter().filter(|t| after_first_window(t)).count()
    }

    #[test]
    fn late_system_start_is_aligned_with_silence_only_before_it() {
        let mic = SimStream::new(DeviceType::Microphone, 1);
        let mut sys = SimStream::new(DeviceType::System, 2);
        sys.start_ms = 250;
        let mixed = run(mic, sys, 30);

        assert!(max_offset_ms(&mixed) < 0.1, "offset {} ms", max_offset_ms(&mixed));
        assert!(mic_padding(&mixed).is_empty());
        let padded = system_padding(&mixed);
        assert!(!padded.is_empty() && padded.iter().all(|&t| t <= 250.1), "system padded outside its absence");
    }

    #[test]
    fn stalled_and_resumed_system_realigns() {
        let mic = SimStream::new(DeviceType::Microphone, 3);
        let mut sys = SimStream::new(DeviceType::System, 4);
        sys.stalls = vec![(10_000, 11_300)];
        let mixed = run(mic, sys, 30);

        assert!(max_offset_ms(&mixed) < 0.1, "offset {} ms after the stall", max_offset_ms(&mixed));
        assert!(mic_padding(&mixed).is_empty());
        let padded = system_padding(&mixed);
        // At most one resumed chunk (<= 10 ms) can fall behind an already-mixed window.
        assert!(padded.iter().all(|&t| (10_000.0..11_311.0).contains(&t)), "system padded outside the stall");
        assert!(padded.len() as f64 >= 1.2 * RATE as f64, "the stall itself is silence");
    }

    #[test]
    fn jittered_callbacks_insert_no_silence_and_stay_aligned() {
        // Mic: 10-30 ms buffers. System: 5-85 ms bursts. Both report capture
        // time up to 10 ms late.
        let mut mic = SimStream::new(DeviceType::Microphone, 7).sizes(480, 1_440);
        let mut sys = SimStream::new(DeviceType::System, 11).sizes(256, 4_096);
        mic.latency_ms = 10;
        sys.latency_ms = 10;
        let mixed = run(mic, sys, 120);

        assert!(mixed.len() >= 199 * 28_800, "only {} samples mixed", mixed.len());
        assert_eq!(steady_state_padding(&mixed), 0, "silence inserted");
        assert!(max_offset_ms(&mixed) <= 11.0, "offset {} ms", max_offset_ms(&mixed));
    }

    #[test]
    fn clock_drift_is_corrected_without_silence() {
        for (mic_clock, sys_clock) in [(1.005, 1.0), (1.0, 0.995)] {
            let mut mic = SimStream::new(DeviceType::Microphone, 5).sizes(480, 1_440);
            let mut sys = SimStream::new(DeviceType::System, 6).sizes(256, 2_048);
            mic.clock = mic_clock;
            sys.clock = sys_clock;
            let mixed = run(mic, sys, 120);

            assert_eq!(steady_state_padding(&mixed), 0, "drift {mic_clock}/{sys_clock}: silence inserted");
            // 0.5% drift is 600 ms over the run; alignment stays within tolerance.
            assert!(max_offset_ms(&mixed) <= 25.0, "drift {mic_clock}/{sys_clock}: offset {} ms", max_offset_ms(&mixed));
        }
    }

    #[test]
    fn single_stream_session_mixes_with_silence() {
        let mut ring = AudioMixerRingBuffer::new(RATE);
        let ten_ms = RATE as usize / 100;
        let mut windows = 0;
        for i in 1..=600 {
            ring.add_samples(DeviceType::Microphone, &vec![MIC_LEVEL; ten_ms], i as f64 / 100.0);
            while let Some((mic, sys)) = ring.extract_window() {
                assert!(mic.iter().all(|&s| s == MIC_LEVEL) && sys.iter().all(|&s| s == 0.0));
                windows += 1;
            }
        }
        assert_eq!(windows, 10, "6 s of mic-only audio mixes as 10 windows");
    }

    #[test]
    fn short_gaps_are_jitter_not_starvation() {
        let mut ring = AudioMixerRingBuffer::new(RATE);
        let window = ring.window_size_samples;
        // Both start at 0; system has delivered to 500 ms, mic to 650 ms.
        ring.add_samples(DeviceType::System, &vec![SYS_LEVEL; window - RATE as usize / 10], 0.5);
        ring.add_samples(DeviceType::Microphone, &vec![MIC_LEVEL; window + RATE as usize / 20], 0.65);
        assert!(ring.extract_window().is_none(), "a 150 ms lead is not starvation");
    }

    #[test]
    fn drain_remaining_returns_the_partial_tail_padded_to_the_longer_stream() {
        let mut ring = AudioMixerRingBuffer::new(RATE);
        ring.add_samples(DeviceType::Microphone, &vec![MIC_LEVEL; 1_000], 1_000.0 / RATE as f64);
        ring.add_samples(DeviceType::System, &vec![SYS_LEVEL; 400], 1_000.0 / RATE as f64);
        assert!(ring.extract_window().is_none());

        let (mic, sys) = ring.drain_remaining().expect("tail");
        assert_eq!(mic, vec![MIC_LEVEL; 1_000]);
        // System's 400 samples were captured at the end of the same interval.
        assert!(sys[..600].iter().all(|&s| s == 0.0));
        assert_eq!(&sys[600..], &vec![SYS_LEVEL; 400][..]);
        assert!(ring.drain_remaining().is_none(), "drained once");
    }
}
