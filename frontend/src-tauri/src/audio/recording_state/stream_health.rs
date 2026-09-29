//! Per-stream capture health for one recording session.
//!
//! A capture stream can fail in two ways that must not end the recording:
//! it can report errors (cpal's ALSA worker calls the error callback in a
//! tight loop on `POLLERR`), or it can silently stop delivering callbacks
//! (cpal 0.15.3 re-prepares an ALSA capture PCM after an overrun but never
//! restarts it, RustAudio/cpal#730). Both end up here: persistent errors or a
//! stall mark the stream dead, and the session supervisor rebuilds it.
//!
//! Every decision takes `now` explicitly so the policy is unit-testable.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{AudioError, DeviceType};

/// Errors are counted over this sliding window.
pub const ERROR_WINDOW: Duration = Duration::from_secs(2);
/// More errors than this inside `ERROR_WINDOW` means the stream is dead.
pub const ERROR_BURST_LIMIT: usize = 5;
/// A running, unpaused stream with no callback for this long is dead.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(3);
/// `audio-stream-degraded` is emitted at most once per stream per interval.
pub const DEGRADED_EVENT_INTERVAL: Duration = Duration::from_secs(10);
/// Stream-error log lines are summarised at most once per interval.
const ERROR_LOG_INTERVAL: Duration = Duration::from_secs(5);
/// A stream rebuilt this many times inside `REBUILD_WINDOW` is given up on,
/// so a device that dies straight after every rebuild cannot loop forever.
pub const MAX_REBUILDS_PER_WINDOW: usize = 5;
pub const REBUILD_WINDOW: Duration = Duration::from_secs(300);

/// Lifecycle of one capture stream within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamStatus {
    /// The session has no stream of this type (none selected or it never started).
    Absent,
    /// A stream is installed and expected to deliver audio.
    Running,
    /// The stream errored or stalled; a rebuild is pending or in progress.
    Dead,
    /// Rebuilding failed; the session continues without this stream.
    Failed,
}

/// Notification from the capture side to the session supervisor.
#[derive(Debug, Clone)]
pub struct StreamFault {
    pub device_type: DeviceType,
    pub reason: String,
}

/// Health changes the frontend is told about. The command layer maps each
/// variant onto its event name and payload.
#[derive(Debug, Clone)]
pub enum StreamHealthEvent {
    /// `audio-stream-degraded`: a stream errored or stalled and is being rebuilt.
    Degraded { device_type: DeviceType, device_name: String, reason: String },
    /// `audio-stream-recovered`: a rebuilt stream is delivering audio again.
    Recovered { device_type: DeviceType, device_name: String },
    /// `system-audio-unavailable`: the system stream could not be started.
    SystemAudioUnavailable { device_name: Option<String>, reason: String },
    /// `recording-error`: no stream can capture audio any more.
    CaptureFailed { message: String },
}

/// What the caller of [`StreamHealth::record_error`] should do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorOutcome {
    /// This error tipped the stream into `Dead`; hand it to the supervisor.
    pub became_dead: bool,
    /// The stream was already dead: cpal is spinning on a broken PCM.
    pub already_dead: bool,
    /// `Some(n)`: log this error now, noting `n` suppressed since the last line.
    pub log_now: Option<u64>,
}

#[derive(Debug)]
struct HealthInner {
    status: StreamStatus,
    running_since_ms: u64,
    awaiting_first_callback: bool,
    errors: VecDeque<Instant>,
    last_degraded_event: Option<Instant>,
    last_error_log: Option<Instant>,
    suppressed_error_logs: u64,
    rebuilds: VecDeque<Instant>,
}

/// Health of one capture stream. The data callback only touches an atomic.
#[derive(Debug)]
pub struct StreamHealth {
    epoch: Instant,
    /// Milliseconds since `epoch` of the latest data callback, plus one (0 = none).
    last_callback_ms: AtomicU64,
    /// Mirrors `status == Dead` so cpal's error spin can bail without locking.
    dead: AtomicBool,
    inner: Mutex<HealthInner>,
}

impl StreamHealth {
    pub fn new(epoch: Instant) -> Self {
        Self {
            epoch,
            last_callback_ms: AtomicU64::new(0),
            dead: AtomicBool::new(false),
            inner: Mutex::new(HealthInner {
                status: StreamStatus::Absent,
                running_since_ms: 0,
                awaiting_first_callback: false,
                errors: VecDeque::new(),
                last_degraded_event: None,
                last_error_log: None,
                suppressed_error_logs: 0,
                rebuilds: VecDeque::new(),
            }),
        }
    }

    fn millis(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.epoch).as_millis() as u64 + 1
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HealthInner> {
        // A panic while holding this lock cannot leave the plain-data state
        // inconsistent, so recover it rather than poisoning capture for good.
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn status(&self) -> StreamStatus {
        self.lock().status
    }

    /// Record a data callback. Called from the audio thread on every buffer.
    pub fn on_callback(&self, now: Instant) {
        self.last_callback_ms.store(self.millis(now), Ordering::Relaxed);
    }

    /// A stream was installed: at session start, or by a rebuild (`is_rebuild`),
    /// in which case the first callback afterwards is reported as a recovery.
    pub fn mark_running(&self, now: Instant, is_rebuild: bool) {
        let mut inner = self.lock();
        inner.status = StreamStatus::Running;
        inner.running_since_ms = self.millis(now);
        inner.awaiting_first_callback = is_rebuild;
        inner.errors.clear();
        self.dead.store(false, Ordering::SeqCst);
    }

    pub fn mark_absent(&self) {
        self.set_terminal(StreamStatus::Absent);
    }

    pub fn mark_failed(&self) {
        self.set_terminal(StreamStatus::Failed);
    }

    fn set_terminal(&self, status: StreamStatus) {
        let mut inner = self.lock();
        inner.status = status;
        inner.awaiting_first_callback = false;
        // Any stream still attached keeps its callbacks cheap.
        self.dead.store(true, Ordering::SeqCst);
    }

    /// Transition `Running -> Dead`. Returns false if the stream was not running.
    fn mark_dead(&self, inner: &mut HealthInner) -> bool {
        if inner.status != StreamStatus::Running {
            return false;
        }
        inner.status = StreamStatus::Dead;
        inner.awaiting_first_callback = false;
        self.dead.store(true, Ordering::SeqCst);
        true
    }

    /// Record a stream error. A disconnect, or more than `ERROR_BURST_LIMIT`
    /// errors inside `ERROR_WINDOW`, marks the stream dead.
    pub fn record_error(&self, error: &AudioError, now: Instant) -> ErrorOutcome {
        if self.dead.load(Ordering::SeqCst) {
            return ErrorOutcome { became_dead: false, already_dead: true, log_now: None };
        }

        let mut inner = self.lock();
        inner.errors.push_back(now);
        while inner
            .errors
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) > ERROR_WINDOW)
        {
            inner.errors.pop_front();
        }

        let persistent = matches!(error, AudioError::DeviceDisconnected)
            || inner.errors.len() > ERROR_BURST_LIMIT;
        let became_dead = persistent && self.mark_dead(&mut inner);

        let log_due = became_dead
            || inner
                .last_error_log
                .map_or(true, |t| now.saturating_duration_since(t) >= ERROR_LOG_INTERVAL);
        let log_now = if log_due {
            inner.last_error_log = Some(now);
            Some(std::mem::take(&mut inner.suppressed_error_logs))
        } else {
            inner.suppressed_error_logs += 1;
            None
        };

        ErrorOutcome { became_dead, already_dead: false, log_now }
    }

    /// Mark a running stream dead if it has delivered nothing for `STALL_TIMEOUT`.
    /// `active` is false while the session is paused, when no check is made.
    pub fn check_stall(&self, now: Instant, active: bool) -> bool {
        if !active {
            return false;
        }
        let mut inner = self.lock();
        if inner.status != StreamStatus::Running {
            return false;
        }
        let last_activity = self
            .last_callback_ms
            .load(Ordering::Relaxed)
            .max(inner.running_since_ms);
        let silent_for = self.millis(now).saturating_sub(last_activity);
        silent_for > STALL_TIMEOUT.as_millis() as u64 && self.mark_dead(&mut inner)
    }

    /// True once, for a rebuilt stream that has delivered its first callback.
    pub fn take_recovered(&self) -> bool {
        let mut inner = self.lock();
        let delivered = self.last_callback_ms.load(Ordering::Relaxed) >= inner.running_since_ms;
        if inner.status == StreamStatus::Running && inner.awaiting_first_callback && delivered {
            inner.awaiting_first_callback = false;
            return true;
        }
        false
    }

    /// Rate limit for `audio-stream-degraded`.
    pub fn should_emit_degraded(&self, now: Instant) -> bool {
        let mut inner = self.lock();
        let due = inner
            .last_degraded_event
            .map_or(true, |t| now.saturating_duration_since(t) >= DEGRADED_EVENT_INTERVAL);
        if due {
            inner.last_degraded_event = Some(now);
        }
        due
    }

    /// Account for one rebuild. False when the stream has used its rebuild
    /// budget, in which case the caller gives up on it.
    pub fn begin_rebuild(&self, now: Instant) -> bool {
        let mut inner = self.lock();
        while inner
            .rebuilds
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) > REBUILD_WINDOW)
        {
            inner.rebuilds.pop_front();
        }
        if inner.rebuilds.len() >= MAX_REBUILDS_PER_WINDOW {
            return false;
        }
        inner.rebuilds.push_back(now);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running(epoch: Instant) -> StreamHealth {
        let health = StreamHealth::new(epoch);
        health.mark_running(epoch, false);
        health
    }

    fn at(epoch: Instant, ms: u64) -> Instant {
        epoch + Duration::from_millis(ms)
    }

    #[test]
    fn sparse_errors_never_kill_the_stream() {
        let epoch = Instant::now();
        let health = running(epoch);
        // One error every 500 ms for a minute: at most 5 inside any 2 s window.
        for i in 0..120 {
            let outcome = health.record_error(&AudioError::StreamFailed, at(epoch, i * 500));
            assert!(!outcome.became_dead, "error {i} killed a stream with a sparse error rate");
        }
        assert_eq!(health.status(), StreamStatus::Running);
    }

    #[test]
    fn error_burst_marks_dead_once_then_short_circuits() {
        let epoch = Instant::now();
        let health = running(epoch);
        let outcomes: Vec<ErrorOutcome> = (0..10)
            .map(|i| health.record_error(&AudioError::StreamFailed, at(epoch, i)))
            .collect();

        let deaths = outcomes.iter().filter(|o| o.became_dead).count();
        assert_eq!(deaths, 1, "exactly one error hands the stream to the supervisor");
        assert!(outcomes[ERROR_BURST_LIMIT].became_dead, "the error past the limit kills it");
        assert!(outcomes[ERROR_BURST_LIMIT + 1..].iter().all(|o| o.already_dead));
        assert_eq!(health.status(), StreamStatus::Dead);
    }

    #[test]
    fn errors_outside_the_window_are_forgotten() {
        let epoch = Instant::now();
        let health = running(epoch);
        for i in 0..ERROR_BURST_LIMIT as u64 {
            health.record_error(&AudioError::StreamFailed, at(epoch, i));
        }
        // The earlier burst has aged out of the 2 s window.
        let outcome = health.record_error(&AudioError::StreamFailed, at(epoch, 2_500));
        assert!(!outcome.became_dead);
    }

    #[test]
    fn disconnect_is_immediately_dead() {
        let epoch = Instant::now();
        let health = running(epoch);
        assert!(health.record_error(&AudioError::DeviceDisconnected, epoch).became_dead);
    }

    #[test]
    fn error_logging_is_rate_limited_and_counts_suppressed_lines() {
        let epoch = Instant::now();
        let health = running(epoch);
        // Spaced so the burst limit is never reached.
        assert_eq!(health.record_error(&AudioError::StreamFailed, epoch).log_now, Some(0));
        assert_eq!(health.record_error(&AudioError::StreamFailed, at(epoch, 1_000)).log_now, None);
        assert_eq!(health.record_error(&AudioError::StreamFailed, at(epoch, 3_000)).log_now, None);
        assert_eq!(health.record_error(&AudioError::StreamFailed, at(epoch, 5_000)).log_now, Some(2));
    }

    #[test]
    fn stall_is_detected_after_three_silent_seconds() {
        let epoch = Instant::now();
        let health = running(epoch);
        health.on_callback(at(epoch, 1_000));
        assert!(!health.check_stall(at(epoch, 3_900), true));
        assert!(health.check_stall(at(epoch, 4_100), true));
        assert_eq!(health.status(), StreamStatus::Dead);
        assert!(!health.check_stall(at(epoch, 9_000), true), "a dead stream is reported once");
    }

    #[test]
    fn stall_check_skips_paused_and_absent_streams() {
        let epoch = Instant::now();
        let health = running(epoch);
        assert!(!health.check_stall(at(epoch, 60_000), false), "paused");
        let absent = StreamHealth::new(epoch);
        assert!(!absent.check_stall(at(epoch, 60_000), true), "absent");
    }

    #[test]
    fn a_stream_that_never_delivers_stalls_from_install_time() {
        let epoch = Instant::now();
        let health = StreamHealth::new(epoch);
        health.mark_running(at(epoch, 10_000), true);
        assert!(!health.check_stall(at(epoch, 12_000), true), "grace period after install");
        assert!(health.check_stall(at(epoch, 13_500), true));
    }

    #[test]
    fn recovery_is_reported_once_after_the_first_callback() {
        let epoch = Instant::now();
        let health = running(epoch);
        health.record_error(&AudioError::DeviceDisconnected, epoch);
        health.mark_running(at(epoch, 1_000), true);
        assert!(!health.take_recovered(), "no audio since the rebuild yet");
        assert!(!health.record_error(&AudioError::StreamFailed, at(epoch, 1_100)).already_dead);
        health.on_callback(at(epoch, 1_200));
        assert!(health.take_recovered());
        assert!(!health.take_recovered());
    }

    #[test]
    fn degraded_events_are_rate_limited_to_one_per_ten_seconds() {
        let epoch = Instant::now();
        let health = running(epoch);
        assert!(health.should_emit_degraded(epoch));
        assert!(!health.should_emit_degraded(at(epoch, 9_000)));
        assert!(health.should_emit_degraded(at(epoch, 10_000)));
    }

    #[test]
    fn rebuild_budget_is_bounded_per_window() {
        let epoch = Instant::now();
        let health = running(epoch);
        for i in 0..MAX_REBUILDS_PER_WINDOW as u64 {
            assert!(health.begin_rebuild(at(epoch, i * 1_000)));
        }
        assert!(!health.begin_rebuild(at(epoch, 10_000)), "budget exhausted");
        assert!(health.begin_rebuild(at(epoch, 301_000)), "old rebuilds age out");
    }
}
