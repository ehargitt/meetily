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
/// A stream whose rebuild failed is retried for the rest of the session, so a
/// device that comes back later is picked up again: first after this long,
/// then at doubling intervals up to `FAILED_STREAM_RETRY_MAX` (each retry can
/// walk cpal's device enumeration).
pub const FAILED_STREAM_RETRY_INTERVAL: Duration = Duration::from_secs(30);
pub const FAILED_STREAM_RETRY_MAX: Duration = Duration::from_secs(300);
/// A rebuilt stream that dies before it has delivered audio for this long
/// did not really recover.
pub const HEALTHY_AFTER_REBUILD: Duration = Duration::from_secs(30);
/// After this many rebuilt streams in a row died early, rebuilding on the same
/// device is treated as failed (a wedged device that opens but never works).
pub const MAX_SHORT_LIVED_REBUILDS: u32 = 3;

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
    /// The device itself went away, so rebuilding on it is pointless.
    pub disconnected: bool,
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
    /// `audio-stream-degraded` was emitted for the current outage, so its
    /// recovery is announced too.
    degraded_announced: bool,
    errors: VecDeque<Instant>,
    last_degraded_event: Option<Instant>,
    last_error_log: Option<Instant>,
    suppressed_error_logs: u64,
    /// When the stream was last marked `Failed` (or last retried).
    failed_at: Option<Instant>,
    /// Wait before the next retry of a `Failed` stream.
    retry_interval: Duration,
    /// The running stream was installed by a rebuild or hot-swap.
    installed_by_rebuild: bool,
    /// Consecutive rebuilt streams that died before running healthily.
    short_lived_rebuilds: u32,
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
                degraded_announced: false,
                errors: VecDeque::new(),
                last_degraded_event: None,
                last_error_log: None,
                suppressed_error_logs: 0,
                failed_at: None,
                retry_interval: FAILED_STREAM_RETRY_INTERVAL,
                installed_by_rebuild: false,
                short_lived_rebuilds: 0,
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

    /// A stream was installed: at session start, or by a rebuild or hot-swap
    /// (`is_rebuild`), in which case its first callback ends the outage and is
    /// reported as a recovery if the outage was announced.
    pub fn mark_running(&self, now: Instant, is_rebuild: bool) {
        let mut inner = self.lock();
        inner.status = StreamStatus::Running;
        inner.running_since_ms = self.millis(now);
        inner.awaiting_first_callback = is_rebuild;
        inner.installed_by_rebuild = is_rebuild;
        if !is_rebuild {
            inner.degraded_announced = false;
        }
        inner.errors.clear();
        inner.failed_at = None;
        self.dead.store(false, Ordering::SeqCst);
    }

    pub fn mark_absent(&self) {
        let mut inner = self.lock();
        inner.status = StreamStatus::Absent;
        inner.awaiting_first_callback = false;
        self.dead.store(true, Ordering::SeqCst);
    }

    /// Rebuilding failed. The stream stays `Failed` until a later retry
    /// (see `retry_due`) or a hot-swap installs a new stream. Failing again
    /// after a retry that did not work, or after a rebuilt stream died early
    /// again, doubles the wait.
    pub fn mark_failed(&self, now: Instant) {
        let mut inner = self.lock();
        let escalate = inner.status == StreamStatus::Failed
            || inner.short_lived_rebuilds >= MAX_SHORT_LIVED_REBUILDS;
        inner.retry_interval = if escalate {
            (inner.retry_interval * 2).min(FAILED_STREAM_RETRY_MAX)
        } else {
            FAILED_STREAM_RETRY_INTERVAL
        };
        inner.status = StreamStatus::Failed;
        inner.awaiting_first_callback = false;
        inner.failed_at = Some(now);
        // Any stream still attached keeps its callbacks cheap.
        self.dead.store(true, Ordering::SeqCst);
    }

    /// True (once per interval) when a `Failed` stream should be retried.
    pub fn retry_due(&self, now: Instant) -> bool {
        let mut inner = self.lock();
        let interval = inner.retry_interval;
        let due = inner.status == StreamStatus::Failed
            && inner
                .failed_at
                .map_or(true, |t| now.saturating_duration_since(t) >= interval);
        if due {
            inner.failed_at = Some(now);
        }
        due
    }

    /// The user was told about this outage (degraded, unavailable or
    /// exhausted), so the stream coming back is announced too.
    pub fn announce_outage(&self) {
        self.lock().degraded_announced = true;
    }

    /// Several rebuilt streams in a row died before running healthily:
    /// rebuilding on the same device again is not going to work.
    pub fn rebuilds_keep_dying(&self) -> bool {
        self.lock().short_lived_rebuilds >= MAX_SHORT_LIVED_REBUILDS
    }

    /// Transition `Running -> Dead`. Returns false if the stream was not running.
    fn mark_dead(&self, inner: &mut HealthInner, now: Instant) -> bool {
        if inner.status != StreamStatus::Running {
            return false;
        }
        let lived_ms = self.millis(now).saturating_sub(inner.running_since_ms);
        let delivered = self.last_callback_ms.load(Ordering::Relaxed) >= inner.running_since_ms;
        if !inner.installed_by_rebuild
            || (delivered && lived_ms >= HEALTHY_AFTER_REBUILD.as_millis() as u64)
        {
            inner.short_lived_rebuilds = 0;
            inner.retry_interval = FAILED_STREAM_RETRY_INTERVAL;
        } else {
            inner.short_lived_rebuilds += 1;
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
        let became_dead = persistent && self.mark_dead(&mut inner, now);

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
        silent_for > STALL_TIMEOUT.as_millis() as u64 && self.mark_dead(&mut inner, now)
    }

    /// True once, for a rebuilt stream that has delivered its first callback
    /// after an outage that was announced with `audio-stream-degraded`. A
    /// hot-swap nobody was warned about ends silently.
    pub fn take_recovered(&self) -> bool {
        let mut inner = self.lock();
        let delivered = self.last_callback_ms.load(Ordering::Relaxed) >= inner.running_since_ms;
        if inner.status == StreamStatus::Running && inner.awaiting_first_callback && delivered {
            inner.awaiting_first_callback = false;
            return std::mem::take(&mut inner.degraded_announced);
        }
        false
    }

    /// Rate limit for `audio-stream-degraded`. A true result means the caller
    /// announces this outage, and its recovery will be announced too.
    pub fn should_emit_degraded(&self, now: Instant) -> bool {
        let mut inner = self.lock();
        let due = inner
            .last_degraded_event
            .map_or(true, |t| now.saturating_duration_since(t) >= DEGRADED_EVENT_INTERVAL);
        if due {
            inner.last_degraded_event = Some(now);
            inner.degraded_announced = true;
        }
        due
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
        assert!(health.should_emit_degraded(epoch));
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
    fn an_unannounced_swap_does_not_report_a_recovery() {
        let epoch = Instant::now();
        let health = running(epoch);
        // Device-monitor hot-swap: no degraded event preceded it.
        health.mark_running(at(epoch, 1_000), true);
        health.on_callback(at(epoch, 1_100));
        assert!(!health.take_recovered());
    }

    #[test]
    fn failed_streams_are_retried_with_a_growing_interval() {
        let epoch = Instant::now();
        let health = running(epoch);
        assert!(!health.retry_due(at(epoch, 60_000)), "only failed streams are retried");
        health.mark_failed(at(epoch, 0));
        assert!(health.record_error(&AudioError::StreamFailed, at(epoch, 500)).already_dead);
        assert!(!health.retry_due(at(epoch, 29_000)));
        assert!(health.retry_due(at(epoch, 30_000)));
        assert!(!health.retry_due(at(epoch, 31_000)), "one retry per interval");

        // Each failed retry doubles the wait: 60 s, 120 s, 240 s, then 300 s.
        let mut t = 30_000;
        for wait in [60_000, 120_000, 240_000, 300_000, 300_000] {
            health.mark_failed(at(epoch, t));
            assert!(!health.retry_due(at(epoch, t + wait - 1_000)), "retried before {wait} ms");
            assert!(health.retry_due(at(epoch, t + wait)), "not retried after {wait} ms");
            t += wait;
        }

        // A stream that comes back starts the schedule over.
        health.mark_running(at(epoch, t), true);
        assert!(!health.retry_due(at(epoch, t + 1_000_000)));
        health.record_error(&AudioError::DeviceDisconnected, at(epoch, t));
        health.mark_failed(at(epoch, t));
        assert!(health.retry_due(at(epoch, t + 30_000)));
    }

    fn rebuild_then_kill(health: &StreamHealth, installed_ms: u64, delivered_ms: Option<u64>, died_ms: u64, epoch: Instant) {
        health.mark_running(at(epoch, installed_ms), true);
        if let Some(ms) = delivered_ms {
            health.on_callback(at(epoch, ms));
        }
        assert!(health.record_error(&AudioError::DeviceDisconnected, at(epoch, died_ms)).became_dead);
    }

    #[test]
    fn rebuilds_that_die_early_are_counted_until_one_runs_healthily() {
        let epoch = Instant::now();
        let health = running(epoch);
        health.record_error(&AudioError::DeviceDisconnected, epoch);
        assert!(!health.rebuilds_keep_dying(), "the original stream dying is not a failed rebuild");

        rebuild_then_kill(&health, 1_000, None, 2_000, epoch); // never delivered
        rebuild_then_kill(&health, 3_000, Some(3_100), 4_000, epoch); // delivered briefly
        assert!(!health.rebuilds_keep_dying());
        rebuild_then_kill(&health, 5_000, None, 50_000, epoch); // alive but silent
        assert!(health.rebuilds_keep_dying(), "three rebuilds in a row died early");

        // Giving up now, and after each retried stream that dies early again,
        // waits longer before the next retry.
        health.mark_failed(at(epoch, 50_000));
        assert!(health.retry_due(at(epoch, 110_000)), "60 s, not 30 s, after a failed streak");
        rebuild_then_kill(&health, 110_000, None, 111_000, epoch);
        health.mark_failed(at(epoch, 111_000));
        assert!(!health.retry_due(at(epoch, 230_000)));
        assert!(health.retry_due(at(epoch, 231_000)), "120 s");

        // One that runs healthily for 30 s clears the streak and the backoff.
        rebuild_then_kill(&health, 240_000, Some(240_100), 271_000, epoch);
        assert!(!health.rebuilds_keep_dying());
        health.mark_failed(at(epoch, 271_000));
        assert!(health.retry_due(at(epoch, 301_000)));
    }
}
