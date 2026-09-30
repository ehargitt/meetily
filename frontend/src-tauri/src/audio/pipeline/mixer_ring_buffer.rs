//! Time-aligns microphone and system audio for mixing.
//!
//! Both streams are placed on one sample timeline and a chunk is normally
//! appended where its stream's previous chunk ended, so callback-size and
//! callback-time jitter never insert silence.
//!
//! Chunks carry the time their callback ran, which is their capture time plus
//! a latency that is never negative (scheduling, resampler buffering, bursty
//! delivery). Each stream's timing is therefore estimated from its least-late
//! recent chunks: `lag` is the minimum of (arrival-derived start − timeline
//! position) over the last `LAG_WINDOW_MS`. From that:
//! - A stream appears (first chunk, or first chunk after it was starved and
//!   padded) at its capture time, aligned with the other stream's `lag`; the
//!   gap before it is silence and anything older than the mixed audio drops.
//! - A chunk arriving more than `RESYNC_MS` later than continuity predicts is
//!   a gap only if that persists for `GAP_CONFIRM_MS` of arrival time. A
//!   backlog delivered after a capture-thread stall catches up with continuity
//!   within the burst and is kept whole; a real gap (lost samples) gets its
//!   silence inserted where it happened.
//! - Clock drift and residual misalignment are corrected in steps of at most
//!   `CORRECTION_STEP_MS` per chunk: samples are dropped from the stream that
//!   runs ahead, or the incoming chunk's first sample is repeated in the one
//!   that lags, so no audible cut and no silence is introduced. A correction
//!   starts only after the misalignment has stayed beyond the deadband for
//!   `CORRECTION_PERSIST_MS` and runs until it is within the inner band, so
//!   an estimate that wobbles (bursty delivery) causes no back-and-forth steps.

use std::collections::VecDeque;

use log::{debug, info};

use crate::audio::recording_state::DeviceType;

/// Mixing window length. Windows are only mixed once both streams have one.
pub(super) const MIX_WINDOW_MS: u32 = 600;

/// A stream is starved (and padded with silence) once the other stream's
/// timeline runs this far ahead of it: it stalled, died, or does not exist.
/// Longer than a capture-thread hiccup, whose backlog should not be lost.
const STARVATION_MS: u32 = 500;

/// A chunk this much later than continuity predicts may start a gap.
const RESYNC_MS: u32 = 100;

/// A suspected gap must persist over this much arrival time to be real.
const GAP_CONFIRM_MS: u32 = 150;

/// Span of arrivals over which each stream's least latency is taken. Short so
/// the estimate follows clock drift closely; its wobble is left to the
/// correction hysteresis below.
const LAG_WINDOW_MS: u32 = 750;

/// Misalignment between the streams tolerated before correcting it.
const ALIGNMENT_DEADBAND_MS: u32 = 5;

/// Once correcting, continue until the misalignment is within this.
const ALIGNMENT_INNER_BAND_MS: u32 = 1;

/// The misalignment must stay beyond the deadband this long before a
/// correction starts, so an estimate that wobbles (bursty delivery whose
/// least-late chunk drifts in phase) does not trigger back-and-forth steps.
const CORRECTION_PERSIST_MS: u32 = 2_000;

/// Most audio removed or held in one correction step (one per chunk).
const CORRECTION_STEP_MS: u32 = 1;

/// A forward jump in arrival time that has not yet been confirmed as a gap.
struct PendingGap {
    /// Timeline index at which the jump happened.
    at: i64,
    /// Arrival time (sample index) at which it was first seen.
    since: i64,
}

/// One stream's buffered samples and its place on the shared timeline.
#[derive(Default)]
struct StreamTimeline {
    samples: VecDeque<f32>,
    /// Timeline index just past the last buffered sample; `None` until the
    /// stream delivers.
    end: Option<i64>,
    /// Place the next chunk by its capture time rather than appending it.
    reanchor: bool,
    /// Recent (arrival end, error + corrections) pairs, kept as a monotonic
    /// deque so the front is the window minimum.
    errors: VecDeque<(i64, i64)>,
    /// Net samples held (+) or dropped (−) by corrections, so errors recorded
    /// before a correction stay comparable with those after it.
    corrections: i64,
    pending_gap: Option<PendingGap>,
    /// Side (+1 hold, −1 drop) and arrival time since the misalignment has
    /// been beyond the deadband, while not yet correcting.
    beyond_deadband: Option<(i64, i64)>,
    /// Correction in progress: +1 holding, −1 dropping, 0 none.
    correcting: i64,
}

impl StreamTimeline {
    /// Delivering and placed by continuity; its `lag` is meaningful.
    fn is_tracking(&self) -> bool {
        self.end.is_some() && !self.reanchor && !self.errors.is_empty()
    }

    fn record_error(&mut self, arrival_end: i64, error: i64, window: i64) {
        let normalized = error + self.corrections;
        while self.errors.back().is_some_and(|&(_, e)| e >= normalized) {
            self.errors.pop_back();
        }
        self.errors.push_back((arrival_end, normalized));
        while self.errors.front().is_some_and(|&(t, _)| arrival_end - t > window) {
            self.errors.pop_front();
        }
    }

    /// Least recent (arrival-derived start − timeline position), in samples.
    fn lag(&self) -> i64 {
        self.errors.front().map_or(0, |&(_, e)| e) - self.corrections
    }

    /// Start over: the stream's placement was just set from its capture time.
    fn reset_timing(&mut self) {
        self.reanchor = false;
        self.errors.clear();
        self.corrections = 0;
        self.pending_gap = None;
        self.beyond_deadband = None;
        self.correcting = 0;
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

    /// Queue a chunk whose callback ran at `arrival_secs` (seconds of active
    /// recording time, at or after the capture of its last sample).
    pub(super) fn add_samples(&mut self, device_type: DeviceType, samples: &[f32], arrival_secs: f64) {
        if samples.is_empty() {
            return;
        }
        let n = samples.len() as i64;
        let arrival_end = (arrival_secs * self.sample_rate as f64).round() as i64;
        let estimated_start = arrival_end - n;
        let head = *self.head.get_or_insert(estimated_start);
        let resync = self.samples_for_ms(RESYNC_MS);
        let confirm = self.samples_for_ms(GAP_CONFIRM_MS);
        let lag_window = self.samples_for_ms(LAG_WINDOW_MS);
        let deadband = self.samples_for_ms(ALIGNMENT_DEADBAND_MS);
        let step = self.samples_for_ms(CORRECTION_STEP_MS).max(1);
        let inner_band = self.samples_for_ms(ALIGNMENT_INNER_BAND_MS);
        let persist = self.samples_for_ms(CORRECTION_PERSIST_MS);

        let (stream, other) = self.streams_mut(device_type);
        let other_lag = other.is_tracking().then(|| other.lag());

        let prev_end = match stream.end {
            Some(end) if !stream.reanchor => end,
            _ => {
                // (Re)appearing: place by capture time, aligned with the other stream.
                let reference = other_lag.unwrap_or(0);
                stream.reset_timing();
                let position = estimated_start - reference;
                Self::place(stream, head, position, samples);
                stream.record_error(arrival_end, reference, lag_window);
                return;
            }
        };

        let error = estimated_start - prev_end;
        stream.record_error(arrival_end, error, lag_window);
        let excess = error - stream.lag();

        if excess > resync {
            let pending = stream.pending_gap.get_or_insert(PendingGap { at: prev_end, since: arrival_end });
            if arrival_end - pending.since < confirm {
                // Possibly a backlog catching up; keep continuity for now.
                Self::place(stream, head, prev_end, samples);
                return;
            }
            // The chunks keep arriving late: samples were lost. Put the
            // silence where they went missing.
            let gap_at = pending.at;
            let since = pending.since;
            stream.pending_gap = None;
            debug!("{:?} mix gap of {} samples confirmed", device_type, excess);
            Self::insert_silence(stream, head, gap_at, excess);
            stream.errors.retain(|&(t, _)| t < since);
            stream.record_error(arrival_end, error - excess, lag_window);
            Self::place(stream, head, prev_end + excess, samples);
            return;
        }
        stream.pending_gap = None;

        // Align with the other stream in small steps, with hysteresis: start
        // once the misalignment has stayed beyond the deadband (same side) for
        // `CORRECTION_PERSIST_MS`, stop once it is back inside the inner band.
        let misalignment = other_lag.map_or(0, |other_lag| stream.lag() - other_lag);
        let side = if misalignment > deadband {
            1
        } else if misalignment < -deadband {
            -1
        } else {
            0
        };
        if stream.correcting != 0 {
            if misalignment.abs() <= inner_band || misalignment.signum() != stream.correcting {
                stream.correcting = 0;
            }
        } else if side == 0 {
            stream.beyond_deadband = None;
        } else {
            let (since_side, since) = *stream.beyond_deadband.get_or_insert((side, arrival_end));
            if since_side != side {
                stream.beyond_deadband = Some((side, arrival_end));
            } else if arrival_end - since >= persist {
                stream.correcting = side;
                stream.beyond_deadband = None;
            }
        }
        if stream.correcting > 0 {
            // Placed earlier than captured relative to the other stream: hold.
            let held = misalignment.min(step);
            stream.samples.extend(std::iter::repeat(samples[0]).take(held as usize));
            stream.end = Some(prev_end + held);
            stream.corrections += held;
            Self::place(stream, head, prev_end + held, samples);
        } else if stream.correcting < 0 {
            // Placed later than captured: drop from the front of this chunk.
            let dropped = (-misalignment).min(step).min(n);
            stream.corrections -= dropped;
            Self::place(stream, head, prev_end - dropped, samples);
        } else {
            Self::place(stream, head, prev_end, samples);
        }
    }

    /// Put `samples` at timeline `position`: pad a gap before it, drop what
    /// overlaps buffered or already-mixed audio.
    fn place(stream: &mut StreamTimeline, head: i64, position: i64, samples: &[f32]) {
        let n = samples.len() as i64;
        let end = stream.end.unwrap_or(head).max(head);
        if position > end {
            stream.samples.extend(std::iter::repeat(0.0).take((position - end) as usize));
        }
        let skip = (end - position).clamp(0, n) as usize;
        stream.samples.extend(samples[skip..].iter().copied());
        stream.end = Some(end.max(position + n));
    }

    /// Insert `len` samples of silence at timeline index `at` (or at the front,
    /// if audio from there on was already mixed).
    fn insert_silence(stream: &mut StreamTimeline, head: i64, at: i64, len: i64) {
        let index = (at - head).clamp(0, stream.samples.len() as i64) as usize;
        let tail = stream.samples.split_off(index);
        stream.samples.extend(std::iter::repeat(0.0).take(len as usize));
        stream.samples.extend(tail);
        stream.end = stream.end.map(|end| end.max(head) + len);
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
    #[derive(Clone)]
    struct Lcg(u64);
    impl Lcg {
        fn next_in(&mut self, lo: usize, hi: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (self.0 >> 33) as usize % (hi - lo + 1)
        }
    }

    /// A simulated capture stream. Each sample's value is its device sample
    /// index plus one (0 is reserved for padding), so the mixed output shows
    /// when each sample was captured and what was dropped or held.
    #[derive(Clone)]
    struct SimStream {
        device_type: DeviceType,
        /// Device clock relative to wall time (1.005 = 0.5% fast).
        clock: f64,
        start_ms: u64,
        /// Wall-time ranges in which the device captures nothing (samples lost).
        capture_stalls: Vec<(u64, u64)>,
        /// Wall-time ranges in which captured audio is held back, then delivered at once.
        delivery_stalls: Vec<(u64, u64)>,
        /// Deliver only every this many ms (bursty graph quantum); 0 = as captured.
        burst_ms: u64,
        next_sample: u64,
        pending: Vec<f32>,
        next_len: usize,
        len_range: (usize, usize),
        /// Maximum callback latency added to the reported time.
        latency_ms: usize,
        jitter: Lcg,
    }

    impl SimStream {
        fn new(device_type: DeviceType, seed: u64) -> Self {
            Self {
                device_type,
                clock: 1.0,
                start_ms: 0,
                capture_stalls: Vec::new(),
                delivery_stalls: Vec::new(),
                burst_ms: 0,
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

        fn device_rate_per_ms(&self) -> f64 {
            RATE as f64 / 1000.0 * self.clock
        }

        /// Wall time (ms) at which the sample with this value was captured.
        fn wall_ms(&self, value: f32) -> f64 {
            self.start_ms as f64 + (value as f64 - 1.0) / self.device_rate_per_ms()
        }

        /// Deliver every buffer the device completed by `now_ms`.
        fn pump(&mut self, now_ms: u64, ring: &mut AudioMixerRingBuffer) {
            if now_ms < self.start_ms {
                return;
            }
            let due = ((now_ms - self.start_ms) as f64 * self.device_rate_per_ms()) as u64;
            while self.next_sample < due {
                let wall_ms = self.start_ms as f64 + self.next_sample as f64 / self.device_rate_per_ms();
                self.next_sample += 1;
                if self.capture_stalls.iter().any(|&(a, b)| wall_ms >= a as f64 && wall_ms < b as f64) {
                    continue;
                }
                self.pending.push(self.next_sample as f32);
            }
            let held_back = self.delivery_stalls.iter().any(|&(a, b)| now_ms >= a && now_ms < b)
                || (self.burst_ms > 0 && now_ms % self.burst_ms != 0);
            if held_back {
                return;
            }
            while self.pending.len() >= self.next_len {
                let chunk: Vec<f32> = self.pending.drain(..self.next_len).collect();
                let latency = if self.latency_ms > 0 { self.jitter.next_in(0, self.latency_ms) } else { 0 };
                ring.add_samples(self.device_type, &chunk, (now_ms + latency as u64) as f64 / 1000.0);
                self.next_len = self.jitter.next_in(self.len_range.0, self.len_range.1);
            }
        }
    }

    /// Mixed output as (mic, system) sample values, per output sample.
    struct Mixed {
        pairs: Vec<(f32, f32)>,
        mic: SimStream,
        sys: SimStream,
    }

    impl Mixed {
        /// Offsets (mic − system capture time, ms) with the mic capture time,
        /// where both carry audio.
        fn offsets(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
            self.pairs.iter().filter(|(m, s)| *m != 0.0 && *s != 0.0).map(|&(m, s)| {
                let mic_ms = self.mic.wall_ms(m);
                (mic_ms, mic_ms - self.sys.wall_ms(s))
            })
        }

        fn max_offset_ms(&self, after_ms: f64) -> f64 {
            self.offsets().filter(|(t, _)| *t >= after_ms).map(|(_, o)| o.abs()).fold(0.0, f64::max)
        }

        /// Wall times (ms, from the other stream) at which `pick` was padded.
        fn padding(&self, system: bool) -> Vec<f64> {
            self.pairs
                .iter()
                .filter_map(|&(m, s)| match system {
                    true if s == 0.0 && m != 0.0 => Some(self.mic.wall_ms(m)),
                    false if m == 0.0 && s != 0.0 => Some(self.sys.wall_ms(s)),
                    _ => None,
                })
                .collect()
        }

        /// Padding after the first window. When the second stream first
        /// appears, its callback latency can leave a few ms before its anchor.
        fn steady_state_padding(&self) -> usize {
            let late = |t: &f64| *t > MIX_WINDOW_MS as f64;
            self.padding(true).iter().filter(|t| late(t)).count() + self.padding(false).iter().filter(|t| late(t)).count()
        }

        /// Largest run of samples removed at once and of samples held, per stream.
        fn largest_corrections(&self, system: bool) -> (u64, u64) {
            let values: Vec<f32> = self
                .pairs
                .iter()
                .map(|&(m, s)| if system { s } else { m })
                .filter(|v| *v != 0.0)
                .collect();
            let (mut dropped, mut held, mut run) = (0u64, 0u64, 0u64);
            for pair in values.windows(2) {
                let step = (pair[1] - pair[0]) as i64;
                if step == 0 {
                    run += 1;
                    held = held.max(run);
                } else {
                    run = 0;
                    dropped = dropped.max((step - 1).max(0) as u64);
                }
            }
            (dropped, held)
        }

        /// Correction steps (a drop or a hold) in this stream's output whose
        /// capture time is at or after `after_ms`.
        fn correction_events(&self, system: bool, after_ms: f64) -> usize {
            let stream = if system { &self.sys } else { &self.mic };
            let values: Vec<f32> = self
                .pairs
                .iter()
                .map(|&(m, s)| if system { s } else { m })
                .filter(|v| *v != 0.0)
                .collect();
            let mut events = 0;
            let mut in_hold = false;
            for pair in values.windows(2) {
                let step = pair[1] - pair[0];
                let late = stream.wall_ms(pair[1]) >= after_ms;
                if step == 0.0 {
                    if !in_hold && late {
                        events += 1;
                    }
                    in_hold = true;
                } else {
                    in_hold = false;
                    if step > 1.0 && late {
                        events += 1;
                    }
                }
            }
            events
        }

        /// Samples of this stream's captured audio missing from the output.
        fn lost_samples(&self, system: bool) -> u64 {
            let stream = if system { &self.sys } else { &self.mic };
            let delivered: std::collections::HashSet<u32> = self
                .pairs
                .iter()
                .map(|&(m, s)| if system { s } else { m })
                .filter(|v| *v != 0.0)
                .map(|v| v as u32)
                .collect();
            let last = delivered.iter().copied().max().unwrap_or(0);
            (1..=last).filter(|v| !delivered.contains(v)).count() as u64 - stream.capture_stall_samples(last)
        }
    }

    impl SimStream {
        fn capture_stall_samples(&self, up_to: u32) -> u64 {
            (1..=up_to)
                .filter(|&v| {
                    let wall = self.wall_ms(v as f32);
                    self.capture_stalls.iter().any(|&(a, b)| wall >= a as f64 && wall < b as f64)
                })
                .count() as u64
        }
    }

    fn run(mut mic: SimStream, mut sys: SimStream, seconds: u64) -> Mixed {
        let mut ring = AudioMixerRingBuffer::new(RATE);
        let mut pairs = Vec::new();
        let (mic_template, sys_template) = (mic.clone(), sys.clone());
        for ms in 0..seconds * 1_000 {
            if ms % 2 == 0 {
                mic.pump(ms, &mut ring);
                sys.pump(ms, &mut ring);
            } else {
                sys.pump(ms, &mut ring);
                mic.pump(ms, &mut ring);
            }
            while let Some((m, s)) = ring.extract_window() {
                pairs.extend(m.into_iter().zip(s));
            }
            let most = ring.window_size_samples * 2;
            assert!(ring.mic.samples.len() <= most && ring.system.samples.len() <= most, "buffers grew at {ms} ms");
        }
        Mixed { pairs, mic: mic_template, sys: sys_template }
    }

    #[test]
    fn late_system_start_is_aligned_with_silence_only_before_it() {
        let mic = SimStream::new(DeviceType::Microphone, 1);
        let mut sys = SimStream::new(DeviceType::System, 2);
        sys.start_ms = 250;
        let mixed = run(mic, sys, 30);

        assert!(mixed.max_offset_ms(0.0) < 0.1, "offset {} ms", mixed.max_offset_ms(0.0));
        assert!(mixed.padding(false).is_empty());
        let padded = mixed.padding(true);
        assert!(!padded.is_empty() && padded.iter().all(|&t| t <= 250.1), "system padded outside its absence");
    }

    #[test]
    fn stalled_and_resumed_system_realigns() {
        let mic = SimStream::new(DeviceType::Microphone, 3);
        let mut sys = SimStream::new(DeviceType::System, 4);
        sys.capture_stalls = vec![(10_000, 11_300)];
        let mixed = run(mic, sys, 30);

        assert!(mixed.max_offset_ms(0.0) < 0.1, "offset {} ms after the stall", mixed.max_offset_ms(0.0));
        assert!(mixed.padding(false).is_empty());
        let padded = mixed.padding(true);
        // At most one resumed chunk (<= 10 ms) can fall behind an already-mixed window.
        assert!(padded.iter().all(|&t| (10_000.0..11_311.0).contains(&t)), "system padded outside the stall");
        assert!(padded.len() as f64 >= 1.2 * RATE as f64, "the stall itself is silence");
    }

    #[test]
    fn a_short_capture_gap_is_silence_exactly_where_audio_was_lost() {
        let mic = SimStream::new(DeviceType::Microphone, 8);
        let mut sys = SimStream::new(DeviceType::System, 9);
        sys.capture_stalls = vec![(10_000, 10_150)];
        let mixed = run(mic, sys, 20);

        let padded = mixed.padding(true);
        assert!(padded.iter().all(|&t| (9_990.0..10_160.0).contains(&t)), "silence outside the gap");
        assert!(padded.len() >= 140 * 48, "the 150 ms gap is silence");
        assert!(mixed.max_offset_ms(10_500.0) < 1.0, "offset {} ms after the gap", mixed.max_offset_ms(10_500.0));
    }

    #[test]
    fn a_delayed_backlog_loses_no_audio_and_stays_aligned() {
        let mic = SimStream::new(DeviceType::Microphone, 10).sizes(480, 1_024);
        let mut sys = SimStream::new(DeviceType::System, 11).sizes(480, 1_024);
        // The capture thread is blocked for 300 ms, then delivers what ALSA buffered.
        sys.delivery_stalls = vec![(10_000, 10_300)];
        let mixed = run(mic, sys, 20);

        assert_eq!(mixed.lost_samples(true), 0, "backlog audio lost");
        assert_eq!(mixed.steady_state_padding(), 0, "silence inserted");
        assert!(mixed.max_offset_ms(0.0) < 1.0, "offset {} ms", mixed.max_offset_ms(0.0));
    }

    #[test]
    fn bursty_delivery_inserts_no_silence_and_settles() {
        for burst_ms in [64, 85, 170, 256] {
            let mic = SimStream::new(DeviceType::Microphone, 12).sizes(480, 1_440);
            let mut sys = SimStream::new(DeviceType::System, 13).sizes(1_024, 1_024);
            sys.burst_ms = burst_ms;
            let mixed = run(mic, sys, 90);

            assert_eq!(mixed.steady_state_padding(), 0, "{burst_ms} ms bursts: silence inserted");
            // Arrival times cannot show where in a burst's period the audio
            // was captured: the least-late chunk can be up to one period
            // (1024 samples, 21.3 ms) late, and that bias stays.
            assert!(mixed.max_offset_ms(20_000.0) <= 21.4,
                    "{burst_ms} ms bursts: offset {} ms", mixed.max_offset_ms(20_000.0));
            let (dropped, held) = mixed.largest_corrections(true);
            assert!(dropped <= 48 && held <= 48, "{burst_ms} ms bursts: correction of {dropped}/{held} samples at once");
            // The latency of the least-late chunk wanders with the burst phase;
            // that must not keep nudging the streams back and forth.
            for system in [false, true] {
                assert_eq!(mixed.correction_events(system, 20_000.0), 0,
                           "{burst_ms} ms bursts: still correcting after settling (system={system})");
            }
        }
    }

    #[test]
    fn jittered_callbacks_insert_no_silence_and_stay_aligned() {
        // Mic: 10-30 ms buffers. System: 5-85 ms bursts. Both report their
        // callback up to 10 ms after capture.
        let mut mic = SimStream::new(DeviceType::Microphone, 7).sizes(480, 1_440);
        let mut sys = SimStream::new(DeviceType::System, 11).sizes(256, 4_096);
        mic.latency_ms = 10;
        sys.latency_ms = 10;
        let mixed = run(mic, sys, 120);

        assert!(mixed.pairs.len() >= 199 * 28_800, "only {} samples mixed", mixed.pairs.len());
        assert_eq!(mixed.steady_state_padding(), 0, "silence inserted");
        assert!(mixed.max_offset_ms(5_000.0) <= 7.0, "offset {} ms", mixed.max_offset_ms(5_000.0));
    }

    #[test]
    fn clock_drift_is_corrected_in_small_steps_without_silence() {
        // 0.5% is ~25x a real device's clock error; 200 ppm is realistic.
        for (mic_clock, sys_clock, max_offset_ms) in [
            (1.005, 1.0, 20.0),
            (1.0, 0.995, 20.0),
            (0.995, 1.0, 20.0),
            (1.0, 1.005, 20.0),
            (1.0002, 1.0, 8.0),
            (1.0, 0.9998, 8.0),
        ] {
            let mut mic = SimStream::new(DeviceType::Microphone, 5).sizes(480, 1_440);
            let mut sys = SimStream::new(DeviceType::System, 6).sizes(256, 2_048);
            mic.clock = mic_clock;
            sys.clock = sys_clock;
            let mixed = run(mic, sys, 120);

            assert_eq!(mixed.steady_state_padding(), 0, "drift {mic_clock}/{sys_clock}: silence inserted");
            assert!(mixed.max_offset_ms(2_000.0) <= max_offset_ms,
                    "drift {mic_clock}/{sys_clock}: offset {} ms", mixed.max_offset_ms(2_000.0));
            for system in [false, true] {
                let (dropped, held) = mixed.largest_corrections(system);
                // One step is 1 ms; allow two steps landing back to back.
                assert!(dropped <= 96 && held <= 96,
                        "drift {mic_clock}/{sys_clock}: {dropped} samples dropped / {held} held at once");
            }
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
        // Both start at 0; system has delivered to 500 ms, mic to 950 ms.
        ring.add_samples(DeviceType::System, &vec![SYS_LEVEL; window - RATE as usize / 10], 0.5);
        ring.add_samples(DeviceType::Microphone, &vec![MIC_LEVEL; window + RATE as usize * 35 / 100], 0.95);
        assert!(ring.extract_window().is_none(), "a 450 ms lead is not starvation");
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
