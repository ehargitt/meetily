//! Decisions of the per-session stream supervisor: when to rebuild a dead
//! stream, on which device, and what to tell the user when that fails.
//!
//! Everything that touches the running app (swapping a stream in the global
//! recording manager, Tauri events, sleeping) goes through
//! [`SupervisorHost`], so the policy is tested without audio hardware.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{error, info, warn};
use serde::Serialize;

use crate::audio::devices::{AudioDevice, DeviceType as DeviceKind};
use crate::audio::recording_state::{
    DeviceType, RecordingState, StreamFault, StreamHealthEvent, StreamStatus,
};

/// How often the supervisor checks for stalls, recoveries and failed streams.
pub(super) const WATCHDOG_INTERVAL: Duration = Duration::from_secs(1);

/// Delays before each attempt to rebuild a dead stream on its own device.
const REBUILD_BACKOFF_MS: [u64; 4] = [250, 1_000, 2_000, 4_000];

/// Delay before moving a disconnected microphone to the default input, so
/// the OS can finish switching its default away from the dead device.
const DISCONNECT_FALLBACK_DELAY_MS: u64 = 150;

/// What the supervisor needs from the running app.
pub(super) trait SupervisorHost: Send + Sync {
    /// The session is still the live recording (not stopping, not replaced).
    fn session_live(&self) -> bool;
    /// Replace the session's stream of `device_type` with one on `device`.
    fn swap_stream(
        &self,
        device_type: DeviceType,
        device: Arc<AudioDevice>,
    ) -> impl Future<Output = Result<(), String>> + Send;
    /// Name of the system default input device, if one exists.
    fn default_input_name(&self) -> Option<String>;
    /// `mic-device-switched`: the microphone moved to another device.
    fn emit_mic_switched(&self, device_name: &str);
    /// `mic-recovery-exhausted`: the microphone could not be restored.
    fn emit_mic_recovery_exhausted(&self, device_name: &str);
    /// Guard preventing overlapping rebuilds/swaps of one stream type.
    fn rebuild_flag(&self, device_type: DeviceType) -> &'static AtomicBool;
    fn sleep(&self, duration: Duration) -> impl Future<Output = ()> + Send;
}

/// Why a rebuild runs.
#[derive(Debug, Clone)]
pub(super) struct RebuildRequest {
    pub device_type: DeviceType,
    pub reason: String,
    /// The device went away: move a microphone to the default input at once.
    pub disconnected: bool,
}

impl From<StreamFault> for RebuildRequest {
    fn from(fault: StreamFault) -> Self {
        Self { device_type: fault.device_type, reason: fault.reason, disconnected: fault.disconnected }
    }
}

/// Clears a rebuild/swap flag on every exit path, including panics.
pub(super) struct FlagReset(&'static AtomicBool);

impl Drop for FlagReset {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// A claimed rebuild: holds the per-type guard for as long as it runs.
pub(super) struct RebuildTicket {
    _guard: FlagReset,
    /// Quiet periodic retry of a stream that already failed.
    retry_of_failed: bool,
}

fn session_device(session: &RecordingState, device_type: DeviceType) -> Option<Arc<AudioDevice>> {
    match device_type {
        DeviceType::Microphone => session.get_microphone_device(),
        DeviceType::System => session.get_system_device(),
    }
}

/// Claim a rebuild of `request.device_type` if one is due: the stream is dead,
/// or it failed and its periodic retry is due, and no rebuild or mic swap of
/// that type is running. Announces a new outage with `audio-stream-degraded`.
pub(super) fn claim_rebuild<H: SupervisorHost>(
    host: &H,
    session: &RecordingState,
    request: &RebuildRequest,
    now: Instant,
) -> Option<RebuildTicket> {
    let health = session.stream_health(request.device_type);
    let retry_of_failed = match health.status() {
        StreamStatus::Dead => false,
        StreamStatus::Failed if health.retry_due(now) => true,
        _ => return None,
    };
    let flag = host.rebuild_flag(request.device_type);
    if flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        return None;
    }
    let ticket = RebuildTicket { _guard: FlagReset(flag), retry_of_failed };

    if !retry_of_failed && health.should_emit_degraded(now) {
        session.emit_health_event(StreamHealthEvent::Degraded {
            device_type: request.device_type,
            device_name: session_device(session, request.device_type)
                .map(|d| d.name.clone())
                .unwrap_or_default(),
            reason: request.reason.clone(),
        });
    }
    Some(ticket)
}

/// One device to try, after `delay`.
struct Attempt {
    device: Arc<AudioDevice>,
    delay: Duration,
    is_fallback: bool,
}

/// The devices a rebuild tries, in order. A microphone whose device went away
/// moves straight to the default input; otherwise the stream's own device is
/// retried with backoff first. A periodic retry of a failed stream makes one
/// attempt per device.
fn rebuild_attempts(
    device_type: DeviceType,
    device: &Arc<AudioDevice>,
    default_input: Option<String>,
    disconnected: bool,
    retry_of_failed: bool,
) -> Vec<Attempt> {
    let fallback = default_input
        .filter(|name| device_type == DeviceType::Microphone && *name != device.name)
        .map(|name| Arc::new(AudioDevice::new(name, DeviceKind::Input)));

    let own_device = |delay_ms: u64| Attempt {
        device: device.clone(),
        delay: Duration::from_millis(delay_ms),
        is_fallback: false,
    };
    let mut attempts: Vec<Attempt> = match (&fallback, disconnected, retry_of_failed) {
        (Some(_), true, _) => Vec::new(),
        (_, _, true) => vec![own_device(0)],
        _ => REBUILD_BACKOFF_MS.iter().map(|&ms| own_device(ms)).collect(),
    };
    if let Some(fallback) = fallback {
        let delay = if disconnected && !retry_of_failed { DISCONNECT_FALLBACK_DELAY_MS } else { 0 };
        attempts.push(Attempt { device: fallback, delay: Duration::from_millis(delay), is_fallback: true });
    }
    attempts
}

/// Rebuild a claimed stream. Gives up on it (see [`fail_stream`]) when every
/// attempt fails; a periodic retry of an already-failed stream fails quietly.
pub(super) async fn rebuild_stream<H: SupervisorHost>(
    host: &H,
    session: &Arc<RecordingState>,
    request: RebuildRequest,
    ticket: RebuildTicket,
) {
    let device_type = request.device_type;
    let Some(device) = session_device(session, device_type) else {
        if !ticket.retry_of_failed {
            fail_stream(host, session, device_type, None, "no device recorded for the session".to_string());
        }
        return;
    };

    info!("[STREAM_REBUILD] Rebuilding {:?} stream on '{}': {}", device_type, device.name, request.reason);
    let mut last_error = request.reason;
    let attempts = rebuild_attempts(
        device_type,
        &device,
        host.default_input_name(),
        request.disconnected,
        ticket.retry_of_failed,
    );
    for attempt in attempts {
        host.sleep(attempt.delay).await;
        if !host.session_live() {
            return;
        }
        match host.swap_stream(device_type, attempt.device.clone()).await {
            Ok(()) => {
                info!("[STREAM_REBUILD] {:?} stream now on '{}'", device_type, attempt.device.name);
                if attempt.is_fallback {
                    host.emit_mic_switched(&attempt.device.name);
                }
                return;
            }
            Err(e) => {
                warn!("[STREAM_REBUILD] {:?} on '{}' failed: {}", device_type, attempt.device.name, e);
                last_error = e;
            }
        }
    }

    if !host.session_live() {
        return;
    }
    if ticket.retry_of_failed {
        // Still gone; the user was already told. Try again next interval.
        session.stream_health(device_type).mark_failed(Instant::now());
        return;
    }
    fail_stream(host, session, device_type, Some(device.name.clone()), last_error);
}

/// Give up on a stream until its next periodic retry: tell the user, and
/// report a capture failure if no stream is left. The session keeps its
/// recording state so the normal stop/save runs.
pub(super) fn fail_stream<H: SupervisorHost>(
    host: &H,
    session: &RecordingState,
    device_type: DeviceType,
    device_name: Option<String>,
    reason: String,
) {
    error!("[STREAM_REBUILD] Giving up on {:?} stream {:?}: {}", device_type, device_name, reason);
    session.stream_health(device_type).mark_failed(Instant::now());
    match device_type {
        DeviceType::System => {
            session.emit_health_event(StreamHealthEvent::SystemAudioUnavailable { device_name, reason });
        }
        DeviceType::Microphone => host.emit_mic_recovery_exhausted(&device_name.unwrap_or_default()),
    }
    if session.all_streams_down() {
        session.report_capture_failed(
            "Audio capture stopped: no microphone or system audio stream could be restarted. \
             The recording up to this point is being saved."
                .to_string(),
        );
    }
}

/// Watchdog tick: mark stalled streams dead and announce recoveries. Returns
/// the rebuild requests to consider (every stream; `claim_rebuild` decides).
pub(super) fn watchdog_tick(session: &RecordingState, now: Instant) -> Vec<RebuildRequest> {
    for device_type in session.detect_stalls(now) {
        warn!("[STREAM_WATCHDOG] {:?} stream delivered no audio for 3 s", device_type);
    }
    [DeviceType::Microphone, DeviceType::System]
        .into_iter()
        .map(|device_type| {
            if session.stream_health(device_type).take_recovered() {
                let device_name = session_device(session, device_type).map(|d| d.name.clone()).unwrap_or_default();
                info!("[STREAM_WATCHDOG] {:?} stream on '{}' is delivering audio again", device_type, device_name);
                session.emit_health_event(StreamHealthEvent::Recovered { device_type, device_name });
            }
            RebuildRequest {
                device_type,
                reason: "no audio received for 3 seconds".to_string(),
                disconnected: false,
            }
        })
        .collect()
}

fn device_type_label(device_type: DeviceType) -> &'static str {
    match device_type {
        DeviceType::Microphone => "microphone",
        DeviceType::System => "system",
    }
}

#[derive(Debug, Clone, Serialize)]
struct StreamDegradedPayload {
    device_type: &'static str,
    device_name: String,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct StreamRecoveredPayload {
    device_type: &'static str,
    device_name: String,
}

#[derive(Debug, Clone, Serialize)]
struct SystemAudioUnavailablePayload {
    device_name: Option<String>,
    reason: String,
}

/// The frontend event name and payload for a stream health change.
pub(super) fn frontend_event(event: StreamHealthEvent) -> (&'static str, serde_json::Value) {
    let payload = match &event {
        StreamHealthEvent::Degraded { device_type, device_name, reason } => {
            serde_json::to_value(StreamDegradedPayload {
                device_type: device_type_label(*device_type),
                device_name: device_name.clone(),
                reason: reason.clone(),
            })
        }
        StreamHealthEvent::Recovered { device_type, device_name } => {
            serde_json::to_value(StreamRecoveredPayload {
                device_type: device_type_label(*device_type),
                device_name: device_name.clone(),
            })
        }
        StreamHealthEvent::SystemAudioUnavailable { device_name, reason } => {
            serde_json::to_value(SystemAudioUnavailablePayload {
                device_name: device_name.clone(),
                reason: reason.clone(),
            })
        }
        StreamHealthEvent::CaptureFailed { message } => Ok(serde_json::Value::String(message.clone())),
    }
    .unwrap_or(serde_json::Value::Null);
    let name = match event {
        StreamHealthEvent::Degraded { .. } => "audio-stream-degraded",
        StreamHealthEvent::Recovered { .. } => "audio-stream-recovered",
        StreamHealthEvent::SystemAudioUnavailable { .. } => "system-audio-unavailable",
        StreamHealthEvent::CaptureFailed { .. } => "recording-error",
    };
    (name, payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::recording_state::AudioError;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// Stands in for the app: scripted swap results, recorded side effects.
    struct FakeHost {
        session: Arc<RecordingState>,
        live: AtomicBool,
        swap_results: Mutex<VecDeque<Result<(), String>>>,
        swaps: Mutex<Vec<(DeviceType, String)>>,
        default_input: Option<String>,
        app_events: Mutex<Vec<String>>,
        flags: &'static [AtomicBool; 2],
    }

    impl FakeHost {
        fn new(session: Arc<RecordingState>, results: Vec<Result<(), String>>) -> Self {
            Self {
                session,
                live: AtomicBool::new(true),
                swap_results: Mutex::new(results.into()),
                swaps: Mutex::new(Vec::new()),
                default_input: None,
                app_events: Mutex::new(Vec::new()),
                flags: Box::leak(Box::new([AtomicBool::new(false), AtomicBool::new(false)])),
            }
        }

        fn swapped_to(&self) -> Vec<String> {
            self.swaps.lock().unwrap().iter().map(|(_, name)| name.clone()).collect()
        }
    }

    impl SupervisorHost for FakeHost {
        fn session_live(&self) -> bool {
            self.live.load(Ordering::SeqCst)
        }

        fn swap_stream(
            &self,
            device_type: DeviceType,
            device: Arc<AudioDevice>,
        ) -> impl Future<Output = Result<(), String>> + Send {
            self.swaps.lock().unwrap().push((device_type, device.name.clone()));
            let result = self.swap_results.lock().unwrap().pop_front().unwrap_or(Err("no device".into()));
            if result.is_ok() {
                // What RecordingManager::install_rebuilt_stream does.
                self.session.stream_health(device_type).mark_running(Instant::now(), true);
                match device_type {
                    DeviceType::Microphone => self.session.set_microphone_device(device),
                    DeviceType::System => self.session.set_system_device(device),
                }
            }
            std::future::ready(result)
        }

        fn default_input_name(&self) -> Option<String> {
            self.default_input.clone()
        }

        fn emit_mic_switched(&self, device_name: &str) {
            self.app_events.lock().unwrap().push(format!("mic-device-switched:{device_name}"));
        }

        fn emit_mic_recovery_exhausted(&self, device_name: &str) {
            self.app_events.lock().unwrap().push(format!("mic-recovery-exhausted:{device_name}"));
        }

        fn rebuild_flag(&self, device_type: DeviceType) -> &'static AtomicBool {
            &self.flags[device_type as usize]
        }

        fn sleep(&self, _duration: Duration) -> impl Future<Output = ()> + Send {
            std::future::ready(())
        }
    }

    /// A live session with a mic ("USB mic") and, optionally, system audio.
    fn session(with_system: bool) -> (Arc<RecordingState>, Arc<Mutex<Vec<StreamHealthEvent>>>) {
        let state = RecordingState::new();
        state.start_recording().unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        state.set_health_callback(move |event| sink.lock().unwrap().push(event));
        state.set_microphone_device(Arc::new(AudioDevice::new("USB mic".into(), DeviceKind::Input)));
        state.stream_health(DeviceType::Microphone).mark_running(Instant::now(), false);
        if with_system {
            state.set_system_device(Arc::new(AudioDevice::new("monitor".into(), DeviceKind::Output)));
            state.stream_health(DeviceType::System).mark_running(Instant::now(), false);
        }
        (state, events)
    }

    fn kill(state: &RecordingState, device_type: DeviceType) {
        state
            .stream_health(device_type)
            .record_error(&AudioError::DeviceDisconnected, Instant::now());
    }

    fn request(device_type: DeviceType, disconnected: bool) -> RebuildRequest {
        RebuildRequest { device_type, reason: "test fault".into(), disconnected }
    }

    fn event_names(events: &Mutex<Vec<StreamHealthEvent>>) -> Vec<&'static str> {
        events.lock().unwrap().iter().cloned().map(|e| frontend_event(e).0).collect()
    }

    async fn run(host: &FakeHost, req: RebuildRequest) {
        let ticket = claim_rebuild(host, &host.session, &req, Instant::now()).expect("rebuild claimed");
        rebuild_stream(host, &host.session, req, ticket).await;
    }

    #[test]
    fn only_dead_streams_are_claimed_and_only_once_at_a_time() {
        let (state, events) = session(true);
        let host = FakeHost::new(state.clone(), vec![]);
        let req = request(DeviceType::System, false);
        assert!(claim_rebuild(&host, &state, &req, Instant::now()).is_none(), "running stream");

        kill(&state, DeviceType::System);
        let ticket = claim_rebuild(&host, &state, &req, Instant::now()).expect("dead stream is claimed");
        assert!(claim_rebuild(&host, &state, &req, Instant::now()).is_none(), "already rebuilding");
        assert_eq!(event_names(&events), vec!["audio-stream-degraded"]);
        drop(ticket);
        assert!(claim_rebuild(&host, &state, &req, Instant::now()).is_some(), "guard released");
    }

    #[tokio::test]
    async fn a_rebuild_retries_the_same_device_until_it_works() {
        let (state, events) = session(true);
        let host = FakeHost::new(state.clone(), vec![Err("busy".into()), Ok(())]);
        kill(&state, DeviceType::System);
        run(&host, request(DeviceType::System, false)).await;

        assert_eq!(host.swapped_to(), vec!["monitor", "monitor"]);
        assert_eq!(state.stream_health(DeviceType::System).status(), StreamStatus::Running);
        state.note_stream_callback(DeviceType::System);
        watchdog_tick(&state, Instant::now());
        assert_eq!(event_names(&events), vec!["audio-stream-degraded", "audio-stream-recovered"]);
    }

    #[tokio::test]
    async fn a_failed_system_stream_is_reported_and_recording_continues() {
        let (state, events) = session(true);
        let host = FakeHost::new(state.clone(), vec![]);
        kill(&state, DeviceType::System);
        run(&host, request(DeviceType::System, false)).await;

        assert_eq!(host.swapped_to().len(), REBUILD_BACKOFF_MS.len());
        assert_eq!(state.stream_health(DeviceType::System).status(), StreamStatus::Failed);
        assert_eq!(event_names(&events), vec!["audio-stream-degraded", "system-audio-unavailable"]);
        assert!(state.is_recording());
    }

    #[tokio::test]
    async fn losing_the_last_stream_reports_a_capture_failure_once() {
        let (state, events) = session(false);
        let host = FakeHost::new(state.clone(), vec![]);
        kill(&state, DeviceType::Microphone);
        run(&host, request(DeviceType::Microphone, false)).await;

        assert_eq!(event_names(&events), vec!["audio-stream-degraded", "recording-error"]);
        assert_eq!(*host.app_events.lock().unwrap(), vec!["mic-recovery-exhausted:USB mic"]);
        assert!(state.is_recording(), "the frontend's normal stop/save ends the session");
    }

    #[tokio::test]
    async fn a_mic_that_cannot_be_rebuilt_falls_back_to_the_default_input() {
        let (state, _) = session(true);
        let mut results: Vec<Result<(), String>> = vec![Err("gone".into()); REBUILD_BACKOFF_MS.len()];
        results.push(Ok(()));
        let mut host = FakeHost::new(state.clone(), results);
        host.default_input = Some("default".into());
        kill(&state, DeviceType::Microphone);
        run(&host, request(DeviceType::Microphone, false)).await;

        assert_eq!(host.swapped_to().last().map(String::as_str), Some("default"));
        assert_eq!(*host.app_events.lock().unwrap(), vec!["mic-device-switched:default"]);
        assert_eq!(state.get_microphone_device().unwrap().name, "default");
    }

    #[tokio::test]
    async fn a_disconnected_mic_moves_to_the_default_input_without_same_device_retries() {
        let (state, _) = session(true);
        let mut host = FakeHost::new(state.clone(), vec![Ok(())]);
        host.default_input = Some("default".into());
        kill(&state, DeviceType::Microphone);
        run(&host, request(DeviceType::Microphone, true)).await;

        assert_eq!(host.swapped_to(), vec!["default"]);
    }

    #[tokio::test]
    async fn a_disconnected_mic_that_is_the_default_is_still_retried() {
        let (state, _) = session(true);
        state.set_microphone_device(Arc::new(AudioDevice::new("default".into(), DeviceKind::Input)));
        let mut host = FakeHost::new(state.clone(), vec![Err("x".into()), Ok(())]);
        host.default_input = Some("default".into());
        kill(&state, DeviceType::Microphone);
        run(&host, request(DeviceType::Microphone, true)).await;

        assert_eq!(host.swapped_to(), vec!["default", "default"]);
    }

    #[tokio::test]
    async fn a_failed_stream_is_retried_quietly_and_recovers_later() {
        let (state, events) = session(true);
        let host = FakeHost::new(state.clone(), vec![]);
        kill(&state, DeviceType::System);
        run(&host, request(DeviceType::System, false)).await;
        assert_eq!(state.stream_health(DeviceType::System).status(), StreamStatus::Failed);

        let req = request(DeviceType::System, false);
        assert!(claim_rebuild(&host, &state, &req, Instant::now()).is_none(), "not due yet");

        // First retry: still gone, no new events.
        let later = Instant::now() + Duration::from_secs(31);
        let ticket = claim_rebuild(&host, &state, &req, later).expect("retry due");
        rebuild_stream(&host, &state, req.clone(), ticket).await;
        assert_eq!(event_names(&events), vec!["audio-stream-degraded", "system-audio-unavailable"]);

        // Second retry: the device is back.
        host.swap_results.lock().unwrap().push_back(Ok(()));
        let ticket = claim_rebuild(&host, &state, &req, later + Duration::from_secs(31)).expect("retry due");
        rebuild_stream(&host, &state, req, ticket).await;
        state.note_stream_callback(DeviceType::System);
        watchdog_tick(&state, Instant::now());
        assert_eq!(
            event_names(&events),
            vec!["audio-stream-degraded", "system-audio-unavailable", "audio-stream-recovered"]
        );
    }

    #[tokio::test]
    async fn a_stopped_session_is_left_alone() {
        let (state, events) = session(true);
        let host = FakeHost::new(state.clone(), vec![]);
        kill(&state, DeviceType::System);
        let req = request(DeviceType::System, false);
        let ticket = claim_rebuild(&host, &state, &req, Instant::now()).unwrap();
        host.live.store(false, Ordering::SeqCst);
        rebuild_stream(&host, &state, req, ticket).await;

        assert!(host.swapped_to().is_empty());
        assert_eq!(event_names(&events), vec!["audio-stream-degraded"], "no failure reported after stop");
    }

    #[test]
    fn a_hot_swap_without_an_outage_is_not_announced() {
        let (state, events) = session(true);
        let host = FakeHost::new(state.clone(), vec![Ok(())]);
        // The device-monitor fallback swaps a mic that never reported a fault.
        let swap = host.swap_stream(DeviceType::Microphone, Arc::new(AudioDevice::new("default".into(), DeviceKind::Input)));
        drop(swap);
        state.note_stream_callback(DeviceType::Microphone);
        watchdog_tick(&state, Instant::now());
        assert!(events.lock().unwrap().is_empty());
    }

    #[test]
    fn health_events_map_to_the_frontend_contract() {
        let cases = [
            (
                StreamHealthEvent::Degraded { device_type: DeviceType::Microphone, device_name: "m".into(), reason: "r".into() },
                "audio-stream-degraded",
                json!({ "device_type": "microphone", "device_name": "m", "reason": "r" }),
            ),
            (
                StreamHealthEvent::Recovered { device_type: DeviceType::System, device_name: "s".into() },
                "audio-stream-recovered",
                json!({ "device_type": "system", "device_name": "s" }),
            ),
            (
                StreamHealthEvent::SystemAudioUnavailable { device_name: None, reason: "r".into() },
                "system-audio-unavailable",
                json!({ "device_name": null, "reason": "r" }),
            ),
            (StreamHealthEvent::CaptureFailed { message: "gone".into() }, "recording-error", json!("gone")),
        ];
        for (event, name, payload) in cases {
            assert_eq!(frontend_event(event), (name, payload));
        }
    }
}
