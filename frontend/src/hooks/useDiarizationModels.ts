import { useState, useCallback, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import {
  DiarizationModelsDownloadError,
  DiarizationModelsDownloadProgress,
  DiarizationModelsStatus,
} from '@/types';

export interface UseDiarizationModelsReturn {
  /** Null until the first status read finishes (or when it fails). */
  status: DiarizationModelsStatus | null;
  /** True from the download request until Rust reports the download has ended. */
  isDownloading: boolean;
  /** A cancel was requested and the download has not stopped yet. */
  isCancelling: boolean;
  downloadProgress: DiarizationModelsDownloadProgress | null;
  downloadError: string | null;
  refresh: () => Promise<DiarizationModelsStatus | null>;
  download: () => Promise<void>;
  cancelDownload: () => Promise<void>;
}

/**
 * Install state of the speaker identification models, with download, progress and cancel.
 * Rust owns the download, so it can outlive this hook: the status read on mount and on window
 * focus picks up a download that another screen started.
 */
export function useDiarizationModels(): UseDiarizationModelsReturn {
  const [status, setStatus] = useState<DiarizationModelsStatus | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<DiarizationModelsDownloadProgress | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  // The failure that follows a user cancel is expected and is not shown as an error.
  const cancelRequestedRef = useRef(false);
  // This hook's own download request is still waiting for Rust to answer.
  const downloadRequestPendingRef = useRef(false);

  const finishDownload = useCallback((error: string | null) => {
    if (!mountedRef.current) return;
    setIsDownloading(false);
    setIsCancelling(false);
    setDownloadProgress(null);
    setDownloadError(cancelRequestedRef.current ? null : error);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const next = await invoke<DiarizationModelsStatus>('get_diarization_models_status');
      if (!mountedRef.current) return next;
      setStatus(next);
      if (next.download_in_progress) {
        setIsDownloading(true);
      } else if (!downloadRequestPendingRef.current) {
        // Covers a download whose end event this screen missed while it was closed.
        setIsDownloading(false);
        setIsCancelling(false);
        setDownloadProgress(null);
      }
      return next;
    } catch (err) {
      console.error('Failed to read speaker identification model status:', err);
      return null;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refresh();

    const unlisteners: UnlistenFn[] = [];
    let cleanedUp = false;
    const register = async <T>(event: string, handler: (payload: T) => void) => {
      const unlisten = await listen<T>(event, ({ payload }) => handler(payload));
      if (cleanedUp) unlisten();
      else unlisteners.push(unlisten);
    };

    void register<DiarizationModelsDownloadProgress>('diarization-models-download-progress', payload => {
      if (!mountedRef.current) return;
      setIsDownloading(true);
      setDownloadProgress(payload);
    });
    void register<unknown>('diarization-models-download-complete', () => {
      finishDownload(null);
      void refresh();
    });
    void register<DiarizationModelsDownloadError>('diarization-models-download-error', payload => {
      finishDownload(payload.cancelled ? null : payload.error);
      void refresh();
    });

    const onFocus = () => { void refresh(); };
    window.addEventListener('focus', onFocus);

    return () => {
      mountedRef.current = false;
      cleanedUp = true;
      unlisteners.forEach(unlisten => unlisten());
      window.removeEventListener('focus', onFocus);
    };
  }, [refresh, finishDownload]);

  const download = useCallback(async () => {
    cancelRequestedRef.current = false;
    downloadRequestPendingRef.current = true;
    setIsDownloading(true);
    setIsCancelling(false);
    setDownloadError(null);
    setDownloadProgress(null);
    try {
      // Resolves once the download has finished, failed or been cancelled.
      await invoke('download_diarization_models');
      downloadRequestPendingRef.current = false;
      finishDownload(null);
    } catch (err) {
      downloadRequestPendingRef.current = false;
      console.error('Failed to download speaker identification models:', err);
      finishDownload(String(err));
    }
    void refresh();
  }, [refresh, finishDownload]);

  // The download keeps running until Rust sees the cancel, so the retry button only returns
  // once Rust reports the end; retrying earlier would be refused as "already in progress".
  const cancelDownload = useCallback(async () => {
    cancelRequestedRef.current = true;
    setIsCancelling(true);
    try {
      await invoke('cancel_diarization_models_download');
    } catch (err) {
      console.error('Failed to cancel model download:', err);
      if (mountedRef.current) setIsCancelling(false);
    }
    void refresh();
  }, [refresh]);

  return {
    status, isDownloading, isCancelling, downloadProgress, downloadError, refresh, download, cancelDownload,
  };
}
