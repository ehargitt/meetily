use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tauri::{
    Emitter,
    menu::{MenuBuilder, MenuItemBuilder, PredefinedMenuItem},
    tray::TrayIconBuilder,
    AppHandle, Manager, Runtime,
};

use crate::audio::recording_commands::{StopSource, STOP_IN_PROGRESS_ERROR};

/// Set while a Quit is stopping and saving; a second Quit meanwhile exits immediately.
static QUIT_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Longest Quit then waits for the frontend to save the stopped meeting.
const QUIT_SAVE_TIMEOUT: Duration = Duration::from_secs(120);
/// How long after tray Start the menu is rebuilt from the real recording state,
/// so a start the frontend declined (setup incomplete, no model, or the previous
/// meeting still saving) does not leave it on "Starting".
const START_REQUEST_MENU_RESYNC: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
pub enum RecordingState {
    Stopped,
    Starting,
    Recording,
    Pausing,
    Paused,
    Resuming,
    Stopping,
}

pub fn create_tray<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    // Start with default menu, will update with actual state after initialization
    // Pass can_record=true initially, will be updated by update_tray_menu immediately
    let menu = build_menu(app, RecordingState::Stopped, true)?;

    TrayIconBuilder::with_id("main-tray")
        .menu(&menu)
        .tooltip("Meetily")
        .icon(app.default_window_icon().unwrap().clone())
        .on_menu_event(|app, event| handle_menu_event(app, event.id.as_ref()))
        .build(app)?;

    // Update tray menu with actual recording state after creation
    update_tray_menu(app);

    Ok(())
}

fn handle_menu_event<R: Runtime>(app: &AppHandle<R>, item_id: &str) {
    match item_id {
        "toggle_recording" => toggle_recording_handler(app),
        "pause_recording" => pause_recording_handler(app),
        "resume_recording" => resume_recording_handler(app),
        "stop_recording" => stop_recording_handler(app),
        "open_window" => focus_main_window(app),
        "settings" => {
            focus_main_window(app);
            // Client-side navigation: reloading the webview would cut short
            // the frontend's save of a meeting that was just stopped.
            if let Err(e) = app.emit("open-settings-from-tray", ()) {
                log::error!("Tray: Failed to request the settings page: {}", e);
            }
        }
        "check_updates" => check_updates_handler(app),
        "quit" => quit_handler(app),
        _ => {}
    }
}
fn toggle_recording_handler<R: Runtime>(app: &AppHandle<R>) {
    focus_main_window(app);
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        if crate::is_recording().await {
            // Immediately show stopping state
            set_tray_state(&app_clone, RecordingState::Stopping);

            log::info!("Tray toggle: Stopping recording...");
            stop_from_tray(&app_clone).await;
        } else {
            // Immediately show starting state
            set_tray_state(&app_clone, RecordingState::Starting);

            // The frontend starts the recording like its own Start button,
            // without a webview reload that would cut short the save of a
            // meeting stopped moments ago; it refuses a start while that runs.
            log::info!("Emitting start recording event from tray");
            if let Err(e) = app_clone.emit("request-recording-toggle", ()) {
                log::error!("Tray: Failed to request a recording start: {}", e);
            }
            tokio::time::sleep(START_REQUEST_MENU_RESYNC).await;
            update_tray_menu_async(&app_clone).await;
        }
    });
}

fn pause_recording_handler<R: Runtime>(app: &AppHandle<R>) {
    // Immediately show pausing state
    set_tray_state(app, RecordingState::Pausing);

    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::audio::recording_commands::pause_recording(app_clone.clone()).await {
            log::error!("Failed to pause recording from tray: {}", e);
            // Revert to current state on error
            update_tray_menu_async(&app_clone).await;
        } else {
            log::info!("Recording paused from tray");
            // The pause_recording function will call update_tray_menu, so no need to call it here
        }
    });
}

fn resume_recording_handler<R: Runtime>(app: &AppHandle<R>) {
    // Immediately show resuming state
    set_tray_state(app, RecordingState::Resuming);

    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::audio::recording_commands::resume_recording(app_clone.clone()).await
        {
            log::error!("Failed to resume recording from tray: {}", e);
            // Revert to current state on error
            update_tray_menu_async(&app_clone).await;
        } else {
            log::info!("Recording resumed from tray");
            // The resume_recording function will call update_tray_menu, so no need to call it here
        }
    });
}

fn stop_recording_handler<R: Runtime>(app: &AppHandle<R>) {
    // Immediately show stopping state
    set_tray_state(app, RecordingState::Stopping);

    focus_main_window(app);
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        log::info!("Tray: Stopping recording...");
        stop_from_tray(&app_clone).await;
    });
}

/// Stop the recording from the tray and hand post-processing (SQLite save,
/// navigation, analytics) to the frontend via `recording-stop-complete`.
async fn stop_from_tray<R: Runtime>(app: &AppHandle<R>) -> TrayStop {
    // Generate save path (same as RecordingControls.tsx)
    let data_dir = match app.path().app_data_dir() {
        Ok(dir) => dir,
        Err(e) => {
            log::error!("Failed to get app data dir: {}", e);
            update_tray_menu_async(app).await;
            return TrayStop::Failed;
        }
    };

    let timestamp = chrono::Local::now().format("%Y-%m-%dT%H-%M-%S").to_string();
    let save_path = data_dir.join(format!("recording-{}.wav", timestamp));

    let stop_result = crate::audio::recording_commands::stop_recording(
        app.clone(),
        crate::audio::recording_commands::RecordingArgs {
            save_path: save_path.to_string_lossy().to_string(),
        },
        StopSource::Tray,
    )
    .await;

    match stop_result {
        Ok(_) => {
            log::info!("Tray: Recording stopped successfully");

            // Trigger frontend post-processing via event (works from any page)
            if let Err(e) = app.emit("recording-stop-complete", true) {
                log::error!("Tray: Failed to emit recording-stop-complete event: {}", e);
            }
            TrayStop::Stopped
        }
        // The stop already running updates the tray and the frontend when it finishes.
        Err(e) if e == STOP_IN_PROGRESS_ERROR => {
            log::info!("Tray: a stop is already in progress; ignoring this one");
            TrayStop::AlreadyStopping
        }
        Err(e) => {
            log::error!("Tray: Failed to stop recording: {}", e);
            // Revert tray state on error
            update_tray_menu_async(app).await;
            TrayStop::Failed
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrayStop {
    Stopped,
    AlreadyStopping,
    /// The recording is still live.
    Failed,
}

/// Stop the recording for Quit, or wait for the stop already running.
/// Returns false if the recording is still live because a stop failed.
async fn stop_for_quit<R: Runtime>(app: &AppHandle<R>) -> bool {
    if stop_from_tray(app).await == TrayStop::Failed {
        return false;
    }
    loop {
        if !crate::audio::recording_commands::is_recording().await {
            return true;
        }
        // Still recording with no stop running: the stop that was running failed.
        if !crate::audio::recording_commands::is_stopping() {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// Wait until the frontend has saved a meeting since the save count was
/// `saves_before`, or `timeout` passes.
async fn wait_for_frontend_save(saves_before: u64, timeout: Duration) {
    let saved = tokio::time::timeout(timeout, async {
        while crate::database::repositories::transcript::saved_meeting_count() == saves_before {
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    })
    .await;
    if saved.is_err() {
        log::warn!(
            "Tray: the stopped meeting was not saved to the database within {}s; exiting anyway",
            timeout.as_secs()
        );
    }
}

/// Quit from the tray. During a recording, stop and save it first (the same
/// path as tray Stop), wait for the frontend to store the meeting, then exit;
/// each wait is bounded (the stop by the stop's own worst case). A meeting
/// stopped moments before Quit also gets its frontend save waited for, and a
/// background transcription still running writes what it has before exit. If
/// the stop fails the recording is still live, so the app stays open and the
/// user is told. A second Quit while Quit is stopping and saving exits at once.
fn quit_handler<R: Runtime>(app: &AppHandle<R>) {
    if QUIT_REQUESTED.swap(true, Ordering::SeqCst) {
        log::warn!("Tray: Quit clicked again; exiting now");
        app.exit(0);
        return;
    }

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        if crate::audio::recording_commands::is_recording().await {
            log::info!("Tray: Quit during a recording; stopping and saving it first");
            set_tray_state(&app, RecordingState::Stopping);
            let saves_before = crate::database::repositories::transcript::saved_meeting_count();
            let stop_bound = crate::audio::recording_commands::current_stop_time_bound();

            match tokio::time::timeout(stop_bound, stop_for_quit(&app)).await {
                Ok(true) => wait_for_frontend_save(saves_before, QUIT_SAVE_TIMEOUT).await,
                Ok(false) => {
                    log::error!(
                        "Tray: the recording could not be stopped, so Meetily stays open and keeps recording"
                    );
                    QUIT_REQUESTED.store(false, Ordering::SeqCst);
                    notify_quit_cancelled(&app).await;
                    update_tray_menu_async(&app).await;
                    return;
                }
                Err(_) => log::error!(
                    "Tray: stopping the recording took over {}s; exiting anyway",
                    stop_bound.as_secs()
                ),
            }
        } else if let Some(stop) = crate::audio::recording_commands::last_completed_stop() {
            // A meeting stopped just before Quit may still be in the frontend's save flow.
            if let Some(remaining) = QUIT_SAVE_TIMEOUT.checked_sub(stop.at.elapsed()) {
                log::info!("Tray: Quit right after a stop; waiting for the meeting to be saved");
                wait_for_frontend_save(stop.saved_meetings, remaining).await;
            }
        }
        crate::audio::recording_commands::close_lingering_drain_for_exit(&app).await;
        app.exit(0);
    });
}

/// Tell the user Quit did not exit because the recording could not be stopped.
async fn notify_quit_cancelled<R: Runtime>(app: &AppHandle<R>) {
    let Some(notifications) =
        app.try_state::<crate::notifications::commands::NotificationManagerState<tauri::Wry>>()
    else {
        return;
    };
    if let Err(e) = crate::notifications::commands::show_system_error_notification(
        &notifications,
        "Meetily is still recording: the recording could not be stopped, so Meetily stayed open. Stop the recording from the Meetily window, then quit.".to_string(),
    )
    .await
    {
        log::error!("Tray: failed to show the Quit-cancelled notification: {}", e);
    }
}

fn check_updates_handler<R: Runtime>(app: &AppHandle<R>) {
    focus_main_window(app);
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.eval(
            "window.dispatchEvent(new CustomEvent('check-updates-from-tray'))"
        );
    }
}

pub fn update_tray_menu<R: Runtime>(app: &AppHandle<R>) {
    // For sync update, spawn async task to get current state
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        // Small delay to ensure recording state has been updated
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        update_tray_menu_async(&app_clone).await;
    });
}

pub fn set_tray_state<R: Runtime>(app: &AppHandle<R>, state: RecordingState) {
    log::info!("Tray: Setting intermediate state: {:?}", state);
    // During recording state transitions, we assume recording is allowed (we're already recording)
    if let Ok(menu) = build_menu(app, state, true) {
        if let Some(tray) = app.tray_by_id("main-tray") {
            let result = tray.set_menu(Some(menu));
            log::info!("Tray: Intermediate state menu update result: {:?}", result);
        } else {
            log::warn!("Tray: Could not find tray with id 'main-tray'");
        }
    } else {
        log::error!("Tray: Failed to build menu for intermediate state");
    }
}

async fn get_current_recording_state() -> RecordingState {
    // Check if currently recording
    let is_recording = crate::audio::recording_commands::is_recording().await;
    log::info!(
        "Tray: get_current_recording_state - is_recording: {}",
        is_recording
    );

    if !is_recording {
        log::info!("Tray: Recording state is Stopped");
        return RecordingState::Stopped;
    }

    if crate::audio::recording_commands::is_stopping() {
        return RecordingState::Stopping;
    }

    // Check if paused
    let is_paused = crate::audio::recording_commands::is_recording_paused().await;
    log::info!("Tray: is_paused: {}", is_paused);

    if is_paused {
        log::info!("Tray: Recording state is Paused");
        RecordingState::Paused
    } else {
        log::info!("Tray: Recording state is Recording");
        RecordingState::Recording
    }
}

/// Check if recording is allowed based on onboarding status and transcription model availability
/// Returns true if:
/// - Onboarding is complete (user may prefer Whisper later), OR
/// - Parakeet transcription model is ready (downloaded)
async fn check_can_record<R: Runtime>(app: &AppHandle<R>) -> bool {
    // First check if onboarding is complete
    let onboarding_complete = match crate::onboarding::load_onboarding_status(app).await {
        Ok(status) => status.completed,
        Err(e) => {
            log::warn!("Tray: Failed to load onboarding status: {}, assuming complete", e);
            true // Assume complete if we can't check (safe default)
        }
    };

    // If onboarding is complete, always allow recording
    // (user may prefer Whisper or have their own transcription setup)
    if onboarding_complete {
        return true;
    }

    // During onboarding, check if Parakeet transcription model is ready
    match crate::parakeet_engine::commands::parakeet_has_available_models().await {
        Ok(has_models) => has_models,
        Err(e) => {
            log::warn!("Tray: Failed to check Parakeet models: {}, assuming not ready", e);
            false
        }
    }
}

pub async fn update_tray_menu_async<R: Runtime>(app: &AppHandle<R>) {
    log::info!("Tray: update_tray_menu_async called");
    // Get the current recording state
    let recording_state = get_current_recording_state().await;
    log::info!("Tray: Current recording state: {:?}", recording_state);

    // Determine if recording should be allowed
    // Only block recording during incomplete onboarding when no transcription model is ready
    let can_record = check_can_record(app).await;
    log::info!("Tray: can_record: {}", can_record);

    if let Ok(menu) = build_menu(app, recording_state, can_record) {
        if let Some(tray) = app.tray_by_id("main-tray") {
            let result = tray.set_menu(Some(menu));
            log::info!("Tray: Menu update result: {:?}", result);
        } else {
            log::warn!("Tray: Could not find tray with id 'main-tray'");
        }
    } else {
        log::error!("Tray: Failed to build menu");
    }
}

fn build_menu<R: Runtime>(
    app: &AppHandle<R>,
    state: RecordingState,
    can_record: bool, // True if recording is allowed (onboarding complete OR transcription model ready)
) -> tauri::Result<tauri::menu::Menu<R>> {
    let mut builder = MenuBuilder::new(app);

    // If recording is not allowed (during onboarding, no transcription model), show disabled message
    if !can_record {
        builder = builder.item(
            &MenuItemBuilder::new("⏳ Downloading transcription model...")
                .enabled(false)
                .build(app)?,
        );
    } else {
        match state {
            RecordingState::Stopped => {
                builder = builder
                    .item(&MenuItemBuilder::with_id("toggle_recording", "Start Recording").build(app)?);
            }
            RecordingState::Starting => {
                builder = builder.item(
                    &MenuItemBuilder::new("🔄 Starting Recording...")
                        .enabled(false)
                        .build(app)?,
                );
            }
            RecordingState::Recording => {
                builder = builder
                    .item(&MenuItemBuilder::with_id("pause_recording", "⏸ Pause Recording").build(app)?)
                    .item(&MenuItemBuilder::with_id("stop_recording", "⏹ Stop Recording").build(app)?);
            }
            RecordingState::Pausing => {
                builder = builder
                    .item(
                        &MenuItemBuilder::new("⏸ Pausing...")
                            .enabled(false)
                            .build(app)?,
                    )
                    .item(&MenuItemBuilder::with_id("stop_recording", "⏹ Stop Recording").build(app)?);
            }
            RecordingState::Paused => {
                builder = builder
                    .item(
                        &MenuItemBuilder::with_id("resume_recording", "▶ Resume Recording")
                            .build(app)?,
                    )
                    .item(&MenuItemBuilder::with_id("stop_recording", "⏹ Stop Recording").build(app)?);
            }
            RecordingState::Resuming => {
                builder = builder
                    .item(
                        &MenuItemBuilder::new("▶ Resuming...")
                            .enabled(false)
                            .build(app)?,
                    )
                    .item(&MenuItemBuilder::with_id("stop_recording", "⏹ Stop Recording").build(app)?);
            }
            RecordingState::Stopping => {
                builder = builder.item(
                    &MenuItemBuilder::new("⏹ Stopping...")
                        .enabled(false)
                        .build(app)?,
                );
            }
        }
    }

    builder
        .item(&PredefinedMenuItem::separator(app)?)
        .item(&MenuItemBuilder::with_id("open_window", "Open Main Window").build(app)?)
        .item(&MenuItemBuilder::with_id("settings", "Settings").build(app)?)
        .item(&MenuItemBuilder::with_id("check_updates", "Check for Updates").build(app)?)
        .item(&PredefinedMenuItem::separator(app)?)
        .item(&MenuItemBuilder::with_id("quit", "Quit").build(app)?)
        .build()
}

pub(crate) fn focus_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        if let Err(e) = window.unminimize() {
            log::error!("Failed to unminimize main window: {}", e);
        }

        if let Err(e) = window.show() {
            log::error!("Failed to show main window: {}", e);
        }

        if let Err(e) = window.set_focus() {
            log::error!("Failed to focus main window: {}", e);
        }

        if let Err(e) = window.eval("window.focus()") {
            log::error!("Failed to focus main webview: {}", e);
        }
    } else {
        log::warn!("Could not find main window");
    }
}
