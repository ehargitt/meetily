'use client';

import React, { createContext, useContext, useState, useEffect, useRef, useCallback, ReactNode, MutableRefObject } from 'react';
import { Transcript, TranscriptUpdate } from '@/types';
import { toast } from 'sonner';
import { useRecordingState } from './RecordingStateContext';
import { transcriptService } from '@/services/transcriptService';
import { recordingService } from '@/services/recordingService';
import { indexedDBService } from '@/services/indexedDBService';
import { mergeTranscripts } from '@/lib/transcriptMerge';

interface TranscriptContextType {
  transcripts: Transcript[];
  transcriptsRef: MutableRefObject<Transcript[]>
  addTranscript: (update: TranscriptUpdate) => void;
  copyTranscript: () => void;
  flushBuffer: () => void;
  transcriptContainerRef: React.RefObject<HTMLDivElement>;
  meetingTitle: string;
  setMeetingTitle: (title: string) => void;
  clearTranscripts: () => void;
  currentMeetingId: string | null;
  markMeetingAsSaved: () => Promise<void>;
}

const TranscriptContext = createContext<TranscriptContextType | undefined>(undefined);

interface TranscriptSessionFilter {
  current: number | null;  // session of the live recording, once recording-started names it
  minimum: number;         // sessions below this belong to earlier recordings
  highest: number;         // highest session id seen
}

/** Whether a transcript update belongs to the recording being shown. */
function isFromCurrentSession(sessionId: number | undefined, session: TranscriptSessionFilter): boolean {
  if (sessionId === undefined) return true;
  if (session.current !== null) return sessionId === session.current;
  return sessionId >= session.minimum;
}

/**
 * Seeds the filter after a (re)load from the last session the backend issued:
 * a recording backend is running that session, an idle one has retired them all.
 */
function seedSessionFilter(session: TranscriptSessionFilter, lastSessionId: number, backendRecording: boolean) {
  session.highest = Math.max(session.highest, lastSessionId);
  if (session.current !== null) return; // recording-started already named the session
  if (backendRecording) {
    session.current = lastSessionId;
  } else {
    session.minimum = Math.max(session.minimum, lastSessionId + 1);
  }
}

export function TranscriptProvider({ children }: { children: ReactNode }) {
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [meetingTitle, setMeetingTitle] = useState('+ New Call');
  const [currentMeetingId, setCurrentMeetingId] = useState<string | null>(null);

  // Recording state context - provides backend-synced state
  const recordingState = useRecordingState();

  // Refs for transcript management
  const transcriptsRef = useRef<Transcript[]>(transcripts);
  const isUserAtBottomRef = useRef<boolean>(true);
  const transcriptContainerRef = useRef<HTMLDivElement>(null);
  const finalFlushRef = useRef<(() => void) | null>(null);

  // Keep ref updated with current transcripts
  useEffect(() => {
    transcriptsRef.current = transcripts;
  }, [transcripts]);

  // Transcription session filter. `current` is the session of the live
  // recording (from recording-started). Clearing for a new recording retires
  // every session seen so far (`minimum`), so a previous meeting's lingering
  // drain can't append to the new one before its own session id arrives.
  // A webview reload resets it; it is then seeded
  // from the backend on mount. An older backend reports no session, and every
  // update is accepted until recording-started names one.
  const transcriptSessionRef = useRef<TranscriptSessionFilter>({ current: null, minimum: 0, highest: 0 });
  const isRecordingRef = useRef(recordingState.isRecording);
  isRecordingRef.current = recordingState.isRecording;

  // Read by the listeners below, which register once: re-subscribing when the
  // ID changes (e.g. restored after a reload) could drop transcript segments.
  const currentMeetingIdRef = useRef<string | null>(currentMeetingId);
  useEffect(() => {
    currentMeetingIdRef.current = currentMeetingId;
  }, [currentMeetingId]);

  // Smart auto-scroll: Track user scroll position
  useEffect(() => {
    const handleScroll = () => {
      const container = transcriptContainerRef.current;
      if (!container) return;

      const { scrollTop, scrollHeight, clientHeight } = container;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10; // 10px tolerance
      isUserAtBottomRef.current = isAtBottom;
    };

    const container = transcriptContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, []);

  // Auto-scroll when transcripts change (only if user is at bottom)
  useEffect(() => {
    // Only auto-scroll if user was at the bottom before new content
    if (isUserAtBottomRef.current && transcriptContainerRef.current) {
      // Wait for Framer Motion animation to complete (150ms) before scrolling
      // This ensures scrollHeight includes the full rendered height of the new transcript
      const scrollTimeout = setTimeout(() => {
        const container = transcriptContainerRef.current;
        if (container) {
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
          });
        }
      }, 150); // Match Framer Motion transition duration

      return () => clearTimeout(scrollTimeout);
    }
  }, [transcripts]);

  // Initialize IndexedDB and listen for recording-started/stopped events
  useEffect(() => {
    // `cancelled` guard prevents leaking a listener when StrictMode/HMR runs
    // cleanup before the async listen(...) registration resolves.
    let cancelled = false;
    let unlistenRecordingStarted: (() => void) | undefined;
    let unlistenRecordingStopped: (() => void) | undefined;

    const setupRecordingListeners = async () => {
      try {
        // Initialize IndexedDB
        await indexedDBService.init();

        // Listen for recording-started event
        const fnStarted = await recordingService.onRecordingStarted(async (payload) => {
          if (typeof payload?.session_id === 'number') {
            const session = transcriptSessionRef.current;
            session.current = payload.session_id;
            session.highest = Math.max(session.highest, payload.session_id);
          }
          try {
            // Generate unique meeting ID
            const meetingId = `meeting-${Date.now()}`;
            setCurrentMeetingId(meetingId);

            // Store in sessionStorage as fallback for markMeetingAsSaved
            sessionStorage.setItem('indexeddb_current_meeting_id', meetingId);
            console.log('[Recording Started] 💾 IndexedDB meeting ID stored:', meetingId);

            // Get meeting name
            const meetingName = await recordingService.getRecordingMeetingName();

            // Use a better fallback that matches the backend's naming pattern
            const effectiveTitle = meetingName || `Meeting ${new Date().toISOString().slice(0, 19).replace('T', '_').replace(/:/g, '-')}`;

            // Initialize meeting metadata in IndexedDB
            await indexedDBService.saveMeetingMetadata({
              meetingId,
              title: effectiveTitle,
              startTime: Date.now(),
              lastUpdated: Date.now(),
              transcriptCount: 0,
              savedToSQLite: false,
              folderPath: undefined // Will update shortly
            });

            // Synchronize meeting title to state (fixes tray stop title issue)
            setMeetingTitle(effectiveTitle);

            // Fetch folder path from backend and update metadata
            // This ensures folder path is persisted even if app crashes
            try {
              const { invoke } = await import('@tauri-apps/api/core');
              const folderPath = await invoke<string>('get_meeting_folder_path');
              if (folderPath) {
                const metadata = await indexedDBService.getMeetingMetadata(meetingId);
                if (metadata) {
                  metadata.folderPath = folderPath;
                  await indexedDBService.saveMeetingMetadata(metadata);
                }
              }
            } catch (error) {
              // Non-fatal - will be set on stop if recording completes normally
            }
          } catch (error) {
            console.error('Failed to initialize meeting in IndexedDB:', error);
          }
        });
        if (cancelled) { fnStarted(); return; }
        unlistenRecordingStarted = fnStarted;

        // Listen for recording-stopped event
        const fnStopped = await recordingService.onRecordingStopped(async (payload) => {
          try {
            const meetingId = currentMeetingIdRef.current;
            if (meetingId) {
              // Update folder path in IndexedDB
              const metadata = await indexedDBService.getMeetingMetadata(meetingId);

              if (metadata && payload.folder_path) {
                metadata.folderPath = payload.folder_path;
                await indexedDBService.saveMeetingMetadata(metadata);
              }
            }
          } catch (error) {
            console.error('Failed to update meeting metadata on stop:', error);
          }
        });
        if (cancelled) { fnStopped(); return; }
        unlistenRecordingStopped = fnStopped;
      } catch (error) {
        console.error('Failed to setup recording listeners:', error);
      }
    };

    setupRecordingListeners();

    return () => {
      cancelled = true;
      if (unlistenRecordingStarted) {
        unlistenRecordingStarted();
        console.log('🧹 Recording started listener cleaned up');
      }
      if (unlistenRecordingStopped) {
        unlistenRecordingStopped();
        console.log('🧹 Recording stopped listener cleaned up');
      }
    };
  }, []);

  // Main transcript buffering logic with sequence_id ordering
  useEffect(() => {
    let cancelled = false;
    let unlistenFn: (() => void) | undefined;
    let transcriptCounter = 0;
    let transcriptBuffer = new Map<number, Transcript>();
    let lastProcessedSequence = 0;
    let processingTimer: NodeJS.Timeout | undefined;

    const processBufferedTranscripts = (forceFlush = false) => {
      const sortedTranscripts: Transcript[] = [];

      // Process all available sequential transcripts
      let nextSequence = lastProcessedSequence + 1;
      while (transcriptBuffer.has(nextSequence)) {
        const bufferedTranscript = transcriptBuffer.get(nextSequence)!;
        sortedTranscripts.push(bufferedTranscript);
        transcriptBuffer.delete(nextSequence);
        lastProcessedSequence = nextSequence;
        nextSequence++;
      }

      // Add any buffered transcripts that might be out of order
      const now = Date.now();
      const staleThreshold = 100;  // 100ms safety net only (serial workers = sequential order)
      const recentThreshold = 0;    // Show immediately - no delay needed with serial processing
      const staleTranscripts: Transcript[] = [];
      const recentTranscripts: Transcript[] = [];
      const forceFlushTranscripts: Transcript[] = [];

      for (const [sequenceId, transcript] of transcriptBuffer.entries()) {
        if (forceFlush) {
          // Force flush mode: process ALL remaining transcripts regardless of timing
          forceFlushTranscripts.push(transcript);
          transcriptBuffer.delete(sequenceId);
          console.log(`Force flush: processing transcript with sequence_id ${sequenceId}`);
        } else {
          const transcriptAge = now - parseInt(transcript.id.split('-')[0]);
          if (transcriptAge > staleThreshold) {
            // Process stale transcripts (>100ms old - safety net)
            staleTranscripts.push(transcript);
            transcriptBuffer.delete(sequenceId);
          } else if (transcriptAge >= recentThreshold) {
            // Process immediately (0ms threshold with serial workers)
            recentTranscripts.push(transcript);
            transcriptBuffer.delete(sequenceId);
            console.log(`Processing transcript with sequence_id ${sequenceId}, age: ${transcriptAge}ms`);
          }
        }
      }

      // Sort both stale and recent transcripts by chunk_start_time, then by sequence_id
      const sortTranscripts = (transcripts: Transcript[]) => {
        return transcripts.sort((a, b) => {
          const chunkTimeDiff = (a.chunk_start_time || 0) - (b.chunk_start_time || 0);
          if (chunkTimeDiff !== 0) return chunkTimeDiff;
          return (a.sequence_id || 0) - (b.sequence_id || 0);
        });
      };

      const sortedStaleTranscripts = sortTranscripts(staleTranscripts);
      const sortedRecentTranscripts = sortTranscripts(recentTranscripts);
      const sortedForceFlushTranscripts = sortTranscripts(forceFlushTranscripts);

      const allNewTranscripts = [...sortedTranscripts, ...sortedRecentTranscripts, ...sortedStaleTranscripts, ...sortedForceFlushTranscripts];

      if (allNewTranscripts.length > 0) {
        setTranscripts(prev => {
          // Create a set of existing sequence_ids for deduplication
          const existingSequenceIds = new Set(prev.map(t => t.sequence_id).filter(id => id !== undefined));

          // Filter out any new transcripts that already exist
          const uniqueNewTranscripts = allNewTranscripts.filter(transcript =>
            transcript.sequence_id !== undefined && !existingSequenceIds.has(transcript.sequence_id)
          );

          // Only combine if we have unique new transcripts
          if (uniqueNewTranscripts.length === 0) {
            console.log('No unique transcripts to add - all were duplicates');
            return prev; // No new unique transcripts to add
          }

          console.log(`Adding ${uniqueNewTranscripts.length} unique transcripts out of ${allNewTranscripts.length} received`);

          // Merge with existing transcripts, maintaining chronological order
          const combined = [...prev, ...uniqueNewTranscripts];

          // Sort by chunk_start_time first, then by sequence_id
          return combined.sort((a, b) => {
            const chunkTimeDiff = (a.chunk_start_time || 0) - (b.chunk_start_time || 0);
            if (chunkTimeDiff !== 0) return chunkTimeDiff;
            return (a.sequence_id || 0) - (b.sequence_id || 0);
          });
        });

        // Log the processing summary
        const logMessage = forceFlush
          ? `Force flush processed ${allNewTranscripts.length} transcripts (${sortedTranscripts.length} sequential, ${forceFlushTranscripts.length} forced)`
          : `Processed ${allNewTranscripts.length} transcripts (${sortedTranscripts.length} sequential, ${recentTranscripts.length} recent, ${staleTranscripts.length} stale)`;
        console.log(logMessage);
      }
    };

    // Assign final flush function to ref for external access
    finalFlushRef.current = () => processBufferedTranscripts(true);

    const setupListener = async () => {
      try {
        console.log('🔥 Setting up MAIN transcript listener during component initialization...');
        const fn = await transcriptService.onTranscriptUpdate((update) => {
          const session = transcriptSessionRef.current;
          if (!isFromCurrentSession(update.session_id, session)) {
            console.log('🚫 MAIN LISTENER: Dropping update from transcription session', update.session_id);
            return;
          }
          if (update.session_id !== undefined) {
            session.highest = Math.max(session.highest, update.session_id);
          }
          const now = Date.now();
          console.log('🎯 MAIN LISTENER: Received transcript update:', {
            sequence_id: update.sequence_id,
            text: update.text.substring(0, 50) + '...',
            timestamp: update.timestamp,
            is_partial: update.is_partial,
            received_at: new Date(now).toISOString(),
            buffer_size_before: transcriptBuffer.size
          });

          // Check for duplicate sequence_id before processing
          if (transcriptBuffer.has(update.sequence_id)) {
            console.log('🚫 MAIN LISTENER: Duplicate sequence_id, skipping buffer:', update.sequence_id);
            return;
          }

          // Create transcript for buffer with NEW timestamp fields
          const newTranscript: Transcript = {
            id: `${Date.now()}-${transcriptCounter++}`,
            text: update.text,
            timestamp: update.timestamp,
            sequence_id: update.sequence_id,
            chunk_start_time: update.chunk_start_time,
            is_partial: update.is_partial,
            confidence: update.confidence,
            // NEW: Recording-relative timestamps for playback sync
            audio_start_time: update.audio_start_time,
            audio_end_time: update.audio_end_time,
            duration: update.duration,
          };

          // Add to buffer
          transcriptBuffer.set(update.sequence_id, newTranscript);
          console.log(`✅ MAIN LISTENER: Buffered transcript with sequence_id ${update.sequence_id}. Buffer size: ${transcriptBuffer.size}, Last processed: ${lastProcessedSequence}`);

          // Save to IndexedDB (non-blocking)
          const meetingId = currentMeetingIdRef.current;
          if (meetingId) {
            indexedDBService.saveTranscript(meetingId, update)
              .catch(err => console.warn('IndexedDB save failed:', err));
          }

          // Clear any existing timer and set a new one
          if (processingTimer) {
            clearTimeout(processingTimer);
          }

          // Process buffer with minimal delay for immediate UI updates (serial workers = sequential order)
          processingTimer = setTimeout(processBufferedTranscripts, 10);
        });
        if (cancelled) { fn(); return; }
        unlistenFn = fn;
        console.log('✅ MAIN transcript listener setup complete');
      } catch (error) {
        console.error('❌ Failed to setup MAIN transcript listener:', error);
        alert('Failed to setup transcript listener. Check console for details.');
      }
    };

    setupListener();
    console.log('Started enhanced listener setup');

    return () => {
      cancelled = true;
      console.log('🧹 CLEANUP: Cleaning up MAIN transcript listener...');
      if (processingTimer) {
        clearTimeout(processingTimer);
        console.log('🧹 CLEANUP: Cleared processing timer');
      }
      if (unlistenFn) {
        unlistenFn();
        console.log('🧹 CLEANUP: MAIN transcript listener cleaned up');
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    recordingService.getRecordingState()
      .then(backendState => {
        if (cancelled || typeof backendState.last_session_id !== 'number') return;
        seedSessionFilter(transcriptSessionRef.current, backendState.last_session_id, backendState.is_recording);
      })
      .catch(error => console.warn('[TranscriptContext] Could not read the last transcription session:', error));
    return () => {
      cancelled = true;
    };
  }, []);

  // currentMeetingId is only set on recording-started, which a webview reload
  // mid-recording misses; without it new segments stop reaching IndexedDB
  // (crash recovery). Restore it from sessionStorage, which survives reloads.
  useEffect(() => {
    if (!recordingState.isRecording || currentMeetingId) return;
    const storedMeetingId = sessionStorage.getItem('indexeddb_current_meeting_id');
    if (storedMeetingId) {
      console.log('[Reload Sync] Restored IndexedDB meeting ID:', storedMeetingId);
      setCurrentMeetingId(storedMeetingId);
    }
  }, [recordingState.isRecording, currentMeetingId]);

  // Sync transcript history and meeting name from backend on reload
  // This fixes the issue where reloading during active recording causes state desync.
  // Merged rather than gated on an empty list: segments that arrive live before
  // the history do not stand in for it.
  useEffect(() => {
    const syncFromBackend = async () => {
      if (recordingState.isRecording) {
        try {
          console.log('[Reload Sync] Recording active after reload, syncing transcript history...');

          // Fetch transcript history from backend
          const history = await transcriptService.getTranscriptHistory();
          console.log(`[Reload Sync] Retrieved ${history.length} transcript segments from backend`);

          // Convert backend format to frontend Transcript format
          const formattedTranscripts: Transcript[] = history.map((segment: any) => ({
            id: segment.id,
            text: segment.text,
            timestamp: segment.display_time, // Use display_time for UI
            sequence_id: segment.sequence_id,
            chunk_start_time: segment.audio_start_time,
            is_partial: false, // History segments are always final
            confidence: segment.confidence,
            audio_start_time: segment.audio_start_time,
            audio_end_time: segment.audio_end_time,
            duration: segment.duration,
          }));

          // Segments emitted after the backend read its history are only in `prev`.
          setTranscripts(prev => mergeTranscripts(prev, formattedTranscripts));
          console.log('[Reload Sync] ✅ Transcript history synced successfully');

          // Fetch meeting name from backend
          const meetingName = await recordingService.getRecordingMeetingName();
          if (meetingName) {
            console.log('[Reload Sync] Retrieved meeting name:', meetingName);
            setMeetingTitle(meetingName);
            console.log('[Reload Sync] ✅ Meeting title synced successfully');
          }
        } catch (error) {
          console.error('[Reload Sync] Failed to sync from backend:', error);
        }
      }
    };

    syncFromBackend();
  }, [recordingState.isRecording]); // Run when recording state changes

  // Manual transcript update handler (for RecordingControls component)
  const addTranscript = useCallback((update: TranscriptUpdate) => {
    console.log('🎯 addTranscript called with:', {
      sequence_id: update.sequence_id,
      text: update.text.substring(0, 50) + '...',
      timestamp: update.timestamp,
      is_partial: update.is_partial
    });

    const newTranscript: Transcript = {
      id: update.sequence_id ? update.sequence_id.toString() : Date.now().toString(),
      text: update.text,
      timestamp: update.timestamp,
      sequence_id: update.sequence_id || 0,
      chunk_start_time: update.chunk_start_time,
      is_partial: update.is_partial,
      confidence: update.confidence,
      audio_start_time: update.audio_start_time,
      audio_end_time: update.audio_end_time,
      duration: update.duration,
    };

    setTranscripts(prev => {
      console.log('📊 Current transcripts count before update:', prev.length);

      // Check if this transcript already exists
      const exists = prev.some(
        t => t.text === update.text && t.timestamp === update.timestamp
      );
      if (exists) {
        console.log('🚫 Duplicate transcript detected, skipping:', update.text.substring(0, 30) + '...');
        return prev;
      }

      // Add new transcript and sort by sequence_id to maintain order
      const updated = [...prev, newTranscript];
      const sorted = updated.sort((a, b) => (a.sequence_id || 0) - (b.sequence_id || 0));

      console.log('✅ Added new transcript. New count:', sorted.length);
      console.log('📝 Latest transcript:', {
        id: newTranscript.id,
        text: newTranscript.text.substring(0, 30) + '...',
        sequence_id: newTranscript.sequence_id
      });

      return sorted;
    });
  }, []);

  // Copy transcript to clipboard with recording-relative timestamps
  const copyTranscript = useCallback(() => {
    // Format timestamps as recording-relative [MM:SS] instead of wall-clock time
    const formatTime = (seconds: number | undefined): string => {
      if (seconds === undefined) return '[--:--]';
      const totalSecs = Math.floor(seconds);
      const mins = Math.floor(totalSecs / 60);
      const secs = totalSecs % 60;
      return `[${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}]`;
    };

    const fullTranscript = transcripts
      .map(t => `${formatTime(t.audio_start_time)} ${t.text}`)
      .join('\n');
    navigator.clipboard.writeText(fullTranscript);

    toast.success("Transcript copied to clipboard");
  }, [transcripts]);

  // Force flush buffer (for final transcript processing)
  const flushBuffer = useCallback(() => {
    if (finalFlushRef.current) {
      console.log('🔄 Flushing transcript buffer...');
      finalFlushRef.current();
    }
  }, []);

  // Clear transcripts (used when starting new recording)
  const clearTranscripts = useCallback(() => {
    setTranscripts([]);
    // Only a clear between recordings retires sessions; the live one must keep
    // its segments.
    if (!isRecordingRef.current) {
      const session = transcriptSessionRef.current;
      session.minimum = session.highest + 1;
      session.current = null;
    }
    // Don't clear currentMeetingId here - it will be set by recording-started event
  }, []);

  // Mark current meeting as saved in IndexedDB
  const markMeetingAsSaved = useCallback(async () => {
    // Try context state first, fallback to sessionStorage
    const meetingId = currentMeetingId || sessionStorage.getItem('indexeddb_current_meeting_id');

    if (!meetingId) {
      console.error('[IndexedDB] ❌ Cannot mark meeting as saved: No meeting ID available!');
      console.error('[IndexedDB] currentMeetingId:', currentMeetingId);
      console.error('[IndexedDB] sessionStorage:', sessionStorage.getItem('indexeddb_current_meeting_id'));
      return;
    }

    try {
      await indexedDBService.markMeetingSaved(meetingId);

      // Clear both sources
      setCurrentMeetingId(null);
      sessionStorage.removeItem('indexeddb_current_meeting_id');
    } catch (error) {
      console.error('[IndexedDB] ❌ Failed to mark meeting as saved:', error);
    }
  }, [currentMeetingId]);

  const value: TranscriptContextType = {
    transcripts,
    transcriptsRef,
    addTranscript,
    copyTranscript,
    flushBuffer,
    transcriptContainerRef,
    meetingTitle,
    setMeetingTitle,
    clearTranscripts,
    currentMeetingId,
    markMeetingAsSaved,
  };

  return (
    <TranscriptContext.Provider value={value}>
      {children}
    </TranscriptContext.Provider>
  );
}

export function useTranscripts() {
  const context = useContext(TranscriptContext);
  if (context === undefined) {
    throw new Error('useTranscripts must be used within a TranscriptProvider');
  }
  return context;
}
