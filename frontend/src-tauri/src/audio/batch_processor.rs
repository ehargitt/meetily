use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::sleep;

/// Most recent batch results kept; older ones are dropped so a long recording
/// does not grow memory without bound.
const MAX_RETAINED_RESULTS: usize = 100;

/// Smart batching processor for reducing operation frequency
/// Collects operations and executes them in batches to reduce overhead
pub struct BatchProcessor<T, R> {
    #[allow(dead_code)]
    batch_size: usize,
    #[allow(dead_code)]
    timeout: Duration,
    #[allow(dead_code)]
    processor: Arc<dyn Fn(Vec<T>) -> R + Send + Sync>,
    sender: mpsc::UnboundedSender<T>,
    results: Arc<RwLock<VecDeque<R>>>,
}

async fn retain_result<R>(results: &RwLock<VecDeque<R>>, result: R) {
    let mut results = results.write().await;
    if results.len() == MAX_RETAINED_RESULTS {
        results.pop_front();
    }
    results.push_back(result);
}

impl<T, R> BatchProcessor<T, R>
where
    T: Send + 'static,
    R: Send + Sync + Clone + 'static,
{
    /// Create a new batch processor
    pub fn new<F>(
        batch_size: usize,
        timeout: Duration,
        processor: F,
    ) -> Self
    where
        F: Fn(Vec<T>) -> R + Send + Sync + 'static,
    {
        let (sender, mut receiver) = mpsc::unbounded_channel::<T>();
        let processor = Arc::new(processor);
        let results = Arc::new(RwLock::new(VecDeque::with_capacity(MAX_RETAINED_RESULTS)));

        let processor_clone = Arc::clone(&processor);
        let results_clone = Arc::clone(&results);

        // Spawn background task to process batches
        tokio::spawn(async move {
            let mut batch = Vec::with_capacity(batch_size);
            let mut last_process = Instant::now();

            loop {
                tokio::select! {
                    // Receive new items
                    item = receiver.recv() => {
                        match item {
                            Some(item) => {
                                batch.push(item);

                                // Process batch if full
                                if batch.len() >= batch_size {
                                    let result = processor_clone(std::mem::take(&mut batch));
                                    retain_result(&results_clone, result).await;
                                    last_process = Instant::now();
                                }
                            }
                            None => break, // Channel closed
                        }
                    }

                    // Timeout to process partial batches
                    _ = sleep(timeout) => {
                        if !batch.is_empty() && last_process.elapsed() >= timeout {
                            let result = processor_clone(std::mem::take(&mut batch));
                            retain_result(&results_clone, result).await;
                            last_process = Instant::now();
                        }
                    }
                }
            }

            // Process any remaining items on shutdown
            if !batch.is_empty() {
                let result = processor_clone(batch);
                retain_result(&results_clone, result).await;
            }
        });

        Self {
            batch_size,
            timeout,
            processor,
            sender,
            results,
        }
    }

    /// Add an item to be processed in a batch
    pub fn add(&self, item: T) -> Result<(), mpsc::error::SendError<T>> {
        self.sender.send(item)
    }

    /// Get the most recent processed results, oldest first
    pub async fn get_results(&self) -> Vec<R> {
        self.results.read().await.iter().cloned().collect()
    }

    /// Clear processed results
    pub async fn clear_results(&self) {
        self.results.write().await.clear();
    }
}

/// Specialized batch processor for audio metrics collection
pub struct AudioMetricsBatcher {
    processor: BatchProcessor<AudioMetric, AudioMetricsSummary>,
}

#[derive(Debug, Clone)]
pub struct AudioMetric {
    pub timestamp: Instant,
    pub chunk_id: u64,
    pub sample_count: usize,
    pub duration_ms: f64,
    pub average_level: f32,
}

#[derive(Debug, Clone)]
pub struct AudioMetricsSummary {
    pub total_chunks: usize,
    pub total_samples: usize,
    pub total_duration_ms: f64,
    pub average_level: f32,
    pub timespan: Duration,
    pub chunks_per_second: f64,
}

impl AudioMetricsBatcher {
    /// Create a new audio metrics batcher
    pub fn new() -> Self {
        let processor = BatchProcessor::new(
            50, // Batch size: process every 50 chunks
            Duration::from_secs(5), // Timeout: process every 5 seconds
            |metrics: Vec<AudioMetric>| {
                if metrics.is_empty() {
                    return AudioMetricsSummary {
                        total_chunks: 0,
                        total_samples: 0,
                        total_duration_ms: 0.0,
                        average_level: 0.0,
                        timespan: Duration::from_secs(0),
                        chunks_per_second: 0.0,
                    };
                }

                let total_chunks = metrics.len();
                let total_samples: usize = metrics.iter().map(|m| m.sample_count).sum();
                let total_duration_ms: f64 = metrics.iter().map(|m| m.duration_ms).sum();
                let average_level: f32 = metrics.iter().map(|m| m.average_level).sum::<f32>() / total_chunks as f32;

                let first_timestamp = metrics.first().unwrap().timestamp;
                let last_timestamp = metrics.last().unwrap().timestamp;
                let timespan = last_timestamp.duration_since(first_timestamp);

                let chunks_per_second = if timespan.as_secs_f64() > 0.0 {
                    total_chunks as f64 / timespan.as_secs_f64()
                } else {
                    0.0
                };

                AudioMetricsSummary {
                    total_chunks,
                    total_samples,
                    total_duration_ms,
                    average_level,
                    timespan,
                    chunks_per_second,
                }
            },
        );

        Self { processor }
    }

    /// Add an audio metric to be batched
    pub fn add_metric(&self, metric: AudioMetric) -> Result<(), mpsc::error::SendError<AudioMetric>> {
        self.processor.add(metric)
    }

    /// Get the most recent summarized audio metrics, oldest first
    pub async fn get_summaries(&self) -> Vec<AudioMetricsSummary> {
        self.processor.get_results().await
    }

    /// Clear cached summaries
    pub async fn clear_summaries(&self) {
        self.processor.clear_results().await
    }
}

impl Default for AudioMetricsBatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro for batched audio metrics logging
#[macro_export]
macro_rules! batch_audio_metric {
    ($batcher:expr, $chunk_id:expr, $sample_count:expr, $duration_ms:expr, $level:expr) => {
        if let Some(batcher) = $batcher {
            let metric = $crate::audio::batch_processor::AudioMetric {
                timestamp: std::time::Instant::now(),
                chunk_id: $chunk_id,
                sample_count: $sample_count,
                duration_ms: $duration_ms,
                average_level: $level,
            };
            let _ = batcher.add_metric(metric);
        }
    };
}
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn only_the_most_recent_results_are_kept() {
        let batches = MAX_RETAINED_RESULTS + 25;
        let processor = BatchProcessor::new(1, Duration::from_secs(60), |items: Vec<usize>| items[0]);
        for item in 0..batches {
            processor.add(item).unwrap();
        }

        let results = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let results = processor.get_results().await;
                if results.last() == Some(&(batches - 1)) {
                    return results;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("every batch is processed");

        assert_eq!(results.len(), MAX_RETAINED_RESULTS);
        assert_eq!(results.first(), Some(&(batches - MAX_RETAINED_RESULTS)));
    }
}
