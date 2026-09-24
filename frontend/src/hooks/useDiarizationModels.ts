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
  isDownloading: boolean;
  downloadProgress: DiarizationModelsDownloadProgress | null;
  downloadError: string | null;
  refresh: () => Promise<DiarizationModelsStatus | null>;
  download: () => Promise<void>;
  cancelDownload: () => Promise<void>;
}

/** Install state of the speaker identification models, with download, progress and cancel. */
export function useDiarizationModels(): UseDiarizationModelsReturn {
  const [status, setStatus] = useState<DiarizationModelsStatus | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<DiarizationModelsDownloadProgress | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  // The failure that follows a user cancel is expected and is not shown as an error.
  const cancelRequestedRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const next = await invoke<DiarizationModelsStatus>('get_diarization_models_status');
      if (mountedRef.current) setStatus(next);
      return next;
    } catch (err) {
      console.error('Failed to read speaker identification model status:', err);
      return null;
    }
  }, []);

  const finishDownload = useCallback((error: string | null) => {
    if (!mountedRef.current) return;
    setIsDownloading(false);
    setDownloadProgress(null);
    setDownloadError(cancelRequestedRef.current ? null : error);
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

    return () => {
      mountedRef.current = false;
      cleanedUp = true;
      unlisteners.forEach(unlisten => unlisten());
    };
  }, [refresh, finishDownload]);

  const download = useCallback(async () => {
    cancelRequestedRef.current = false;
    setIsDownloading(true);
    setDownloadError(null);
    setDownloadProgress(null);
    try {
      await invoke('download_diarization_models');
      // The command may return once the download has started; completion then arrives as an event.
      const next = await refresh();
      if (next?.installed) finishDownload(null);
    } catch (err) {
      console.error('Failed to download speaker identification models:', err);
      finishDownload(String(err));
    }
  }, [refresh, finishDownload]);

  const cancelDownload = useCallback(async () => {
    cancelRequestedRef.current = true;
    try {
      await invoke('cancel_diarization_models_download');
    } catch (err) {
      console.error('Failed to cancel model download:', err);
    }
    finishDownload(null);
    void refresh();
  }, [refresh, finishDownload]);

  return { status, isDownloading, downloadProgress, downloadError, refresh, download, cancelDownload };
}
