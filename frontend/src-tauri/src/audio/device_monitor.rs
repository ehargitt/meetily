// Audio device monitoring for disconnect/reconnect detection
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use anyhow::Result;
use log::{debug, info, warn, error};

#[cfg(target_os = "macos")]
use cidre::{core_audio as ca, os};

use super::devices::AudioDevice;
use super::devices::discovery::list_audio_devices_blocking;

/// Poll interval while every monitored device is present.
const POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Poll interval while a monitored device is missing.
const MISSING_POLL_INTERVAL: Duration = Duration::from_secs(2);
/// A device enumeration still running after this is left to finish on its own.
const ENUMERATION_TIMEOUT: Duration = Duration::from_secs(10);
/// How long `stop_monitoring` waits for the monitor loop before aborting it.
const STOP_TIMEOUT: Duration = Duration::from_secs(2);

type Enumeration = JoinHandle<Result<Vec<AudioDevice>>>;

/// Device monitoring events
#[derive(Debug, Clone)]
pub enum DeviceEvent {
    /// A device that was in use has disconnected
    DeviceDisconnected {
        device_name: String,
        device_type: DeviceMonitorType,
    },
    /// A previously disconnected device has reconnected
    DeviceReconnected {
        device_name: String,
        device_type: DeviceMonitorType,
    },
    /// Device list has changed (new device added or removed)
    DeviceListChanged,
}

/// Type of device being monitored
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceMonitorType {
    Microphone,
    SystemAudio,
}

/// Monitor state for a single device
#[derive(Debug, Clone)]
struct MonitoredDevice {
    name: String,
    device_type: DeviceMonitorType,
    consecutive_missing: u32,
    is_bluetooth: bool,
}

impl MonitoredDevice {
    fn new(name: String, device_type: DeviceMonitorType) -> Self {
        // Heuristic: check if device name contains bluetooth-related keywords
        let is_bluetooth = name.to_lowercase().contains("airpods")
            || name.to_lowercase().contains("bluetooth")
            || name.to_lowercase().contains("wireless");

        Self {
            name,
            device_type,
            consecutive_missing: 0,
            is_bluetooth,
        }
    }

    /// Get appropriate disconnect threshold based on device type
    fn disconnect_threshold(&self) -> u32 {
        // Bluetooth devices get more grace period (they can briefly disconnect)
        if self.is_bluetooth {
            3 // 3 polling cycles (6-15 seconds)
        } else {
            2 // 2 polling cycles (4-10 seconds)
        }
    }

    /// Get appropriate reconnect check interval
    #[allow(dead_code)]
    fn reconnect_interval(&self) -> Duration {
        if self.is_bluetooth {
            Duration::from_secs(5) // Check every 5s for Bluetooth
        } else {
            Duration::from_secs(3) // Check every 3s for wired devices
        }
    }
}

//---------- Core Audio callback-driven device detection (macOS)----------

/// Callback data shared between the Core Audio listener and the monitor loop.
#[cfg(target_os = "macos")]
struct ListenerCallbackData {
    /// Wakes the monitor loop for immediate device check
    notify: Arc<tokio::sync::Notify>,
}

/// RAII guard that unregisters the Core Audio property listener on drop.
#[cfg(target_os = "macos")]
struct DeviceChangeListenerGuard {
    data_ptr: *mut ListenerCallbackData,
}

// SAFETY: data_ptr points to a Box<ListenerCallbackData> whose fields are Send + Sync.
// The pointer is created in register_device_change_listeners() and only
// used for cleanup in Drop — always on the same tokio task.
#[cfg(target_os = "macos")]
unsafe impl Send for DeviceChangeListenerGuard {}

#[cfg(target_os = "macos")]
impl Drop for DeviceChangeListenerGuard {
    fn drop(&mut self) {
        let ptr = self.data_ptr as *mut ();
        let _ = ca::System::OBJ.remove_prop_listener(
            &ca::PropSelector::HW_DEVICES.global_addr(),
            device_list_changed_callback,
            ptr,
        );
        // Deliberately leak the Box'd callback data (~100 bytes) instead of freeing it.
        // AudioObjectRemovePropertyListener does not fence in-flight callbacks: a
        // callback already dispatched on Core Audio's internal queue can still
        // dereference this pointer after removal returns, so freeing here would be a
        // use-after-free. The leak is bounded — one allocation per recording session,
        // reclaimed at app exit. Same tradeoff as the Core Audio listeners in
        // system_detector.rs.
        // unsafe { let _ = Box::from_raw(self.data_ptr); }
        info!("Core Audio device change listener removed (1 listener, callback data intentionally leaked)");
    }
}

/// Core Audio callback: hardware device list changed (add/remove).
#[cfg(target_os = "macos")]
extern "C-unwind" fn device_list_changed_callback(
    _obj_id: ca::Obj,
    _number_addresses: u32,
    _addresses: *const ca::PropAddr,
    client_data: *mut (),
) -> os::Status {
    let data = unsafe { &*(client_data as *const ListenerCallbackData) };
    data.notify.notify_one();
    debug!("Core Audio device list changed — waking monitor loop");
    os::Status::NO_ERR
}

/// Register the Core Audio property listener for device list changes.
/// Returns an RAII guard that unregisters the listener on drop.
#[cfg(target_os = "macos")]
fn register_device_change_listeners(
    notify: Arc<tokio::sync::Notify>,
) -> Option<DeviceChangeListenerGuard> {
    let data_ptr = Box::into_raw(Box::new(ListenerCallbackData { notify }));
    let ptr = data_ptr as *mut ();

    // Register HW_DEVICES listener (required — bail if fails)
    match ca::System::OBJ.add_prop_listener(
        &ca::PropSelector::HW_DEVICES.global_addr(),
        device_list_changed_callback,
        ptr,
    ) {
        Ok(()) => info!("Registered Core Audio HW_DEVICES listener"),
        Err(e) => {
            error!("Failed to register HW_DEVICES listener: {:?}", e);
            unsafe { let _ = Box::from_raw(data_ptr); }
            return None;
        }
    }

    Some(DeviceChangeListenerGuard { data_ptr })
}

/// Audio device monitor that detects disconnects and reconnects
pub struct AudioDeviceMonitor {
    monitor_handle: Option<JoinHandle<()>>,
    event_sender: mpsc::UnboundedSender<DeviceEvent>,
    stop_signal: Arc<tokio::sync::Notify>,
    /// Wakes the monitor loop instantly when Core Audio reports a device
    /// change (macOS); never signaled on other platforms.
    device_change_notify: Arc<tokio::sync::Notify>,
    /// Mailbox for hot-swap device updates. When a mic hot-swap completes,
    /// the new device names are written here. The monitor loop reads it on
    /// its next poll cycle and updates its tracked device list.
    /// Format: (new_mic_name, optional_new_system_name)
    device_update_mailbox: Arc<std::sync::Mutex<Option<(String, Option<String>)>>>,
}

impl AudioDeviceMonitor {
    /// Create a new device monitor
    pub fn new() -> (Self, mpsc::UnboundedReceiver<DeviceEvent>) {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let stop_signal = Arc::new(tokio::sync::Notify::new());
        let device_change_notify = Arc::new(tokio::sync::Notify::new());

        (
            Self {
                monitor_handle: None,
                event_sender,
                stop_signal,
                device_change_notify,
                device_update_mailbox: Arc::new(std::sync::Mutex::new(None)),
            },
            event_receiver,
        )
    }

    /// Notify the monitor that the mic has been hot-swapped to a new device.
    /// Optionally also update the system audio tracked name (since BT devices
    /// like AirPods are often both mic AND speaker — when they disconnect,
    /// both entries go stale). The monitor loop picks this up on its next
    /// poll cycle.
    pub fn notify_mic_swapped(&self, new_mic_name: String, new_system_name: Option<String>) {
        if let Ok(mut mailbox) = self.device_update_mailbox.lock() {
            *mailbox = Some((new_mic_name, new_system_name));
        }
    }

    /// Start monitoring specified devices
    pub fn start_monitoring(
        &mut self,
        microphone: Option<Arc<AudioDevice>>,
        system_audio: Option<Arc<AudioDevice>>,
    ) -> Result<()> {
        if self.monitor_handle.is_some() {
            warn!("Device monitor already running");
            return Ok(());
        }

        let mut monitored_devices = Vec::new();

        if let Some(mic) = microphone {
            monitored_devices.push(MonitoredDevice::new(
                mic.name.clone(),
                DeviceMonitorType::Microphone,
            ));
            info!("🔍 Monitoring microphone: '{}' (Bluetooth: {})",
                  mic.name, monitored_devices.last().unwrap().is_bluetooth);
        }

        if let Some(sys) = system_audio {
            monitored_devices.push(MonitoredDevice::new(
                sys.name.clone(),
                DeviceMonitorType::SystemAudio,
            ));
            info!("🔍 Monitoring system audio: '{}' (Bluetooth: {})",
                  sys.name, monitored_devices.last().unwrap().is_bluetooth);
        }

        if monitored_devices.is_empty() {
            return Err(anyhow::anyhow!("No devices to monitor"));
        }

        let event_sender = self.event_sender.clone();
        let stop_signal = self.stop_signal.clone();
        let device_change_notify = self.device_change_notify.clone();
        let device_update_mailbox = self.device_update_mailbox.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(monitored_devices, event_sender, stop_signal, device_change_notify, device_update_mailbox).await;
        });

        self.monitor_handle = Some(handle);
        info!("✅ Device monitor started");
        Ok(())
    }

    /// Stop monitoring. Waits at most `STOP_TIMEOUT` for the monitor loop (it
    /// may be waiting on a slow device enumeration), then aborts it.
    pub async fn stop_monitoring(&mut self) {
        self.stop_monitoring_within(STOP_TIMEOUT).await;
    }

    async fn stop_monitoring_within(&mut self, timeout: Duration) {
        info!("Stopping device monitor");
        self.stop_signal.notify_one();

        if let Some(mut handle) = self.monitor_handle.take() {
            if tokio::time::timeout(timeout, &mut handle).await.is_err() {
                warn!("Device monitor did not stop within {:?}; aborting it", timeout);
                handle.abort();
            }
        }

        info!("Device monitor stopped");
    }

    /// The current device list, enumerated on a blocking thread. `None` when
    /// the enumeration failed or has not finished within `timeout`; an
    /// unfinished one is kept in `pending` and awaited again next time rather
    /// than starting another, so a hung sound server holds one thread, not one
    /// per poll.
    async fn next_device_list<F>(
        pending: &mut Option<Enumeration>,
        enumerate: F,
        timeout: Duration,
    ) -> Option<Vec<AudioDevice>>
    where
        F: FnOnce() -> Result<Vec<AudioDevice>> + Send + 'static,
    {
        let mut enumeration = pending.take().unwrap_or_else(|| tokio::task::spawn_blocking(enumerate));
        match tokio::time::timeout(timeout, &mut enumeration).await {
            Ok(Ok(Ok(devices))) => Some(devices),
            Ok(Ok(Err(e))) => {
                error!("Failed to list audio devices: {}", e);
                None
            }
            Ok(Err(e)) => {
                error!("Audio device enumeration failed: {}", e);
                None
            }
            Err(_) => {
                warn!("Audio device enumeration still running after {:?}; skipping this check", timeout);
                *pending = Some(enumeration);
                None
            }
        }
    }

    /// Poll faster while a monitored device is missing.
    fn poll_interval(monitored_devices: &[MonitoredDevice]) -> Duration {
        if monitored_devices.iter().any(|d| d.consecutive_missing > 0) {
            MISSING_POLL_INTERVAL
        } else {
            POLL_INTERVAL
        }
    }

    /// Main monitoring loop
    async fn monitor_loop(
        mut monitored_devices: Vec<MonitoredDevice>,
        event_sender: mpsc::UnboundedSender<DeviceEvent>,
        stop_signal: Arc<tokio::sync::Notify>,
        device_change_notify: Arc<tokio::sync::Notify>,
        device_update_mailbox: Arc<std::sync::Mutex<Option<(String, Option<String>)>>>,
    ) {
        let mut last_device_list = Vec::new();
        let mut check_interval = Self::poll_interval(&monitored_devices);
        let mut pending_enumeration: Option<Enumeration> = None;

        #[cfg(target_os = "macos")]
        let _listener_guard = register_device_change_listeners(device_change_notify.clone());

        loop {
            // Check for stop signal with timeout
            tokio::select! {
                _ = stop_signal.notified() => {
                    info!("Device monitor received stop signal");
                    break;
                }
                _ = device_change_notify.notified() => {
                    debug!("Device monitor woken by Core Audio callback — checking devices immediately");
                    // Fall through to poll/diff below
                }
                _ = tokio::time::sleep(check_interval) => {
                    // Continue with monitoring check
                }
            }

            // Get current device list
            let Some(current_devices) =
                Self::next_device_list(&mut pending_enumeration, list_audio_devices_blocking, ENUMERATION_TIMEOUT).await
            else {
                continue;
            };

            // Check for hot-swap device update from the mailbox.
            // If a hot-swap completed since our last cycle, update our tracked
            // devices so we stop polling for the dead ones and start watching
            // the fallback devices instead. Rebuilding via MonitoredDevice::new
            // re-derives is_bluetooth from the new name and resets
            // consecutive_missing.
            if let Ok(mut mailbox) = device_update_mailbox.lock() {
                if let Some((new_mic_name, new_system_name)) = mailbox.take() {
                    for dev in &mut monitored_devices {
                        match dev.device_type {
                            DeviceMonitorType::Microphone => {
                                info!("[DEVICE_MONITOR] Updated tracked mic: '{}' → '{}' (hot-swap)", dev.name, new_mic_name);
                                *dev = MonitoredDevice::new(new_mic_name.clone(), dev.device_type.clone());
                            }
                            DeviceMonitorType::SystemAudio => {
                                if let Some(ref sys_name) = new_system_name {
                                    info!("[DEVICE_MONITOR] Updated tracked system audio: '{}' → '{}' (hot-swap)", dev.name, sys_name);
                                    *dev = MonitoredDevice::new(sys_name.clone(), dev.device_type.clone());
                                }
                            }
                        }
                    }
                }
            }

            // Check if device list changed
            if current_devices.len() != last_device_list.len() {
                debug!("Device list changed: {} -> {} devices",
                       last_device_list.len(), current_devices.len());
                let _ = event_sender.send(DeviceEvent::DeviceListChanged);
            }
            last_device_list = current_devices.clone();

            // Check each monitored device
            for monitored in &mut monitored_devices {
                let device_found = current_devices.iter().any(|d| d.name == monitored.name);

                if device_found {
                    // Device is present
                    if monitored.consecutive_missing > 0 {
                        // Device has reconnected!
                        info!("✅ Device '{}' reconnected after {} missing checks",
                              monitored.name, monitored.consecutive_missing);

                        let _ = event_sender.send(DeviceEvent::DeviceReconnected {
                            device_name: monitored.name.clone(),
                            device_type: monitored.device_type.clone(),
                        });

                        monitored.consecutive_missing = 0;
                    }
                } else {
                    // Device is missing
                    monitored.consecutive_missing += 1;

                    debug!("⚠️ Device '{}' missing for {} checks (threshold: {})",
                          monitored.name, monitored.consecutive_missing,
                          monitored.disconnect_threshold());

                    // Re-fire every `threshold` cycles while the device is still missing. A
                    // successful hot-swap retargets us via the mailbox (resets the counter); a
                    // FAILED swap leaves the dead device tracked, so this re-fire is what retries
                    // it. Total attempts are bounded in trigger_mic_fallback_to_default. (P1 #2)
                    if monitored.consecutive_missing % monitored.disconnect_threshold() == 0 {
                        warn!("❌ Device '{}' ({:?}) disconnected!",
                              monitored.name, monitored.device_type);

                        let _ = event_sender.send(DeviceEvent::DeviceDisconnected {
                            device_name: monitored.name.clone(),
                            device_type: monitored.device_type.clone(),
                        });
                    }
                }
            }

            let next_interval = Self::poll_interval(&monitored_devices);
            if next_interval != check_interval {
                debug!("Adjusting monitor interval to {:?}", next_interval);
                check_interval = next_interval;
            }
        }
    }
}

impl Default for AudioDeviceMonitor {
    fn default() -> Self {
        Self::new().0
    }
}

impl Drop for AudioDeviceMonitor {
    fn drop(&mut self) {
        // Signal stop
        self.stop_signal.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::devices::DeviceType;

    #[test]
    fn test_bluetooth_detection() {
        let airpods = MonitoredDevice::new(
            "John's AirPods Pro".to_string(),
            DeviceMonitorType::Microphone,
        );
        assert!(airpods.is_bluetooth);
        assert_eq!(airpods.disconnect_threshold(), 3);

        let builtin = MonitoredDevice::new(
            "Built-in Microphone".to_string(),
            DeviceMonitorType::Microphone,
        );
        assert!(!builtin.is_bluetooth);
        assert_eq!(builtin.disconnect_threshold(), 2);
    }

    #[tokio::test]
    async fn test_monitor_creation() {
        let (mut monitor, _receiver) = AudioDeviceMonitor::new();
        assert!(monitor.monitor_handle.is_none());

        // Stop should be safe even if not started
        monitor.stop_monitoring().await;
    }

    #[test]
    fn polling_speeds_up_while_a_device_is_missing() {
        let mut devices = vec![
            MonitoredDevice::new("USB mic".to_string(), DeviceMonitorType::Microphone),
            MonitoredDevice::new("monitor".to_string(), DeviceMonitorType::SystemAudio),
        ];
        assert_eq!(AudioDeviceMonitor::poll_interval(&devices), POLL_INTERVAL);
        devices[1].consecutive_missing = 1;
        assert_eq!(AudioDeviceMonitor::poll_interval(&devices), MISSING_POLL_INTERVAL);
    }

    #[tokio::test]
    async fn a_monitor_loop_that_does_not_stop_is_aborted() {
        struct SetOnDrop(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let guard = SetOnDrop(dropped.clone());

        let (mut monitor, _receiver) = AudioDeviceMonitor::new();
        // Stands in for a loop stuck awaiting a hung device enumeration.
        monitor.monitor_handle = Some(tokio::spawn(async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        }));

        tokio::time::timeout(Duration::from_secs(1), monitor.stop_monitoring_within(Duration::from_millis(50)))
            .await
            .expect("stop waited on the stuck loop");
        tokio::time::timeout(Duration::from_secs(1), async {
            while !dropped.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the stuck loop was aborted");
    }

    #[tokio::test]
    async fn a_slow_enumeration_is_awaited_again_instead_of_started_twice() {
        let mut pending = None;
        let slow = || {
            std::thread::sleep(Duration::from_millis(300));
            Ok(vec![AudioDevice::new("USB mic".to_string(), DeviceType::Input)])
        };
        let first = AudioDeviceMonitor::next_device_list(&mut pending, slow, Duration::from_millis(20)).await;
        assert!(first.is_none(), "the check is skipped while enumeration is slow");
        assert!(pending.is_some());

        let started_again = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = started_again.clone();
        let second = AudioDeviceMonitor::next_device_list(
            &mut pending,
            move || {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(Vec::new())
            },
            Duration::from_secs(5),
        )
        .await;
        let names: Vec<String> = second.expect("the first enumeration's result").into_iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["USB mic"]);
        assert!(!started_again.load(std::sync::atomic::Ordering::SeqCst), "no second enumeration thread");
        assert!(pending.is_none());
    }
}
