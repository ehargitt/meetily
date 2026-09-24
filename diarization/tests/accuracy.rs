//! End-to-end accuracy on a 60 s AMI excerpt (see `fixtures/README.md`). Needs the real models:
//!
//! ```text
//! MEETILY_DIARIZATION_MODELS_DIR=/path/to/models cargo test -p meetily-diarization -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::time::Instant;

use meetily_diarization::{DiarizationConfig, DiarizationEngine, DiarizationError, SpeakerTurn, Stage};

const FIXTURE: &str = "ami_es2004a_795s_60s";
/// `diar-proto run --step 48000 --workers 8` on the same excerpt: 4 speakers, DER 20.49% with no
/// collar (miss 15.64%, false alarm 4.34%, confusion 0.51%).
const PROTOTYPE_DER: f64 = 0.2049;

fn fixture(ext: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(format!("{FIXTURE}.{ext}"))
}

fn models_dir() -> PathBuf {
    std::env::var_os("MEETILY_DIARIZATION_MODELS_DIR")
        .map(PathBuf::from)
        .expect("set MEETILY_DIARIZATION_MODELS_DIR to a directory holding both diarization models")
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open fixture wav");
    let spec = reader.spec();
    assert_eq!((spec.sample_rate, spec.channels, spec.bits_per_sample), (16_000, 1, 16));
    reader.samples::<i16>().map(|s| s.expect("wav sample") as f32 / 32768.0).collect()
}

/// Reference turns `(start, end, speaker index)` from an RTTM file.
fn read_rttm(path: &Path) -> (Vec<(f64, f64, usize)>, usize) {
    let text = std::fs::read_to_string(path).expect("read rttm");
    let mut names: Vec<String> = Vec::new();
    let mut turns = Vec::new();
    for line in text.lines().filter(|l| l.starts_with("SPEAKER")) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let start: f64 = f[3].parse().expect("rttm onset");
        let dur: f64 = f[4].parse().expect("rttm duration");
        let id = names.iter().position(|n| n == f[7]).unwrap_or_else(|| {
            names.push(f[7].to_string());
            names.len() - 1
        });
        turns.push((start, start + dur, id));
    }
    (turns, names.len())
}

/// Hypothesis→reference mapping (`None` = unmapped) maximising total overlap, by brute force.
fn best_mapping(
    overlap: &[Vec<f64>],
    h: usize,
    used: &mut Vec<bool>,
    cur: &mut Vec<Option<usize>>,
    best: &mut (f64, Vec<Option<usize>>),
) {
    if h == overlap.len() {
        let score: f64 = cur.iter().enumerate().filter_map(|(h, r)| r.map(|r| overlap[h][r])).sum();
        if score > best.0 {
            *best = (score, cur.clone());
        }
        return;
    }
    cur.push(None);
    best_mapping(overlap, h + 1, used, cur, best);
    cur.pop();
    for r in 0..used.len() {
        if !used[r] {
            used[r] = true;
            cur.push(Some(r));
            best_mapping(overlap, h + 1, used, cur, best);
            cur.pop();
            used[r] = false;
        }
    }
}

/// Diarization error rate on 10 ms frames with no collar (same scoring as the prototype's
/// `eval::der`): (missed + false alarm + confusion) / reference speech.
fn der(reference: &[(f64, f64, usize)], num_ref: usize, hyp: &[SpeakerTurn], total_secs: f64) -> f64 {
    let frames = (total_secs * 100.0).ceil() as usize + 1;
    let to_frame = |t: f64| ((t * 100.0).round().max(0.0) as usize).min(frames);
    let num_hyp = hyp.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
    let mut r = vec![vec![false; num_ref]; frames];
    let mut h = vec![vec![false; num_hyp]; frames];
    for &(s, e, id) in reference {
        (to_frame(s)..to_frame(e)).for_each(|f| r[f][id] = true);
    }
    for t in hyp {
        (to_frame(t.start)..to_frame(t.end)).for_each(|f| h[f][t.speaker] = true);
    }
    let mut overlap = vec![vec![0f64; num_ref]; num_hyp];
    for f in 0..frames {
        for (hs, row) in overlap.iter_mut().enumerate() {
            for (rs, o) in row.iter_mut().enumerate() {
                if h[f][hs] && r[f][rs] {
                    *o += 1.0;
                }
            }
        }
    }
    let mut best = (f64::MIN, Vec::new());
    best_mapping(&overlap, 0, &mut vec![false; num_ref], &mut Vec::new(), &mut best);
    let mapping = best.1;

    let (mut errors, mut speech) = (0f64, 0f64);
    for f in 0..frames {
        let n_ref = r[f].iter().filter(|&&x| x).count() as f64;
        let n_hyp = h[f].iter().filter(|&&x| x).count() as f64;
        let correct = (0..num_hyp).filter(|&hs| h[f][hs] && mapping[hs].is_some_and(|rs| r[f][rs])).count() as f64;
        speech += n_ref;
        errors += (n_ref - n_hyp).max(0.0) + (n_hyp - n_ref).max(0.0) + (n_ref.min(n_hyp) - correct);
    }
    errors / speech.max(1.0)
}

#[test]
#[ignore = "needs MEETILY_DIARIZATION_MODELS_DIR"]
fn ami_excerpt_matches_prototype_accuracy() {
    let audio = read_wav(&fixture("wav"));
    let (reference, num_ref) = read_rttm(&fixture("rttm"));
    let expected: Vec<SpeakerTurn> =
        serde_json::from_str(&std::fs::read_to_string(fixture("expected_turns.json")).expect("read expected turns"))
            .expect("parse expected turns");

    let started = Instant::now();
    let engine = DiarizationEngine::load(&models_dir(), 8).expect("load models");
    let loaded = started.elapsed();
    let reports: Mutex<Vec<(Stage, f32)>> = Mutex::new(Vec::new());
    let out = engine
        .diarize(&audio, &DiarizationConfig::fast(), &AtomicBool::new(false), &|stage, frac| {
            reports.lock().unwrap().push((stage, frac))
        })
        .expect("diarize");
    let diarized = started.elapsed() - loaded;
    let reports = reports.into_inner().unwrap();
    for stage in [Stage::Segmenting, Stage::Embedding, Stage::Clustering] {
        let last = reports.iter().filter(|r| r.0 == stage).map(|r| r.1).fold(f32::MIN, f32::max);
        assert_eq!(last, 1.0, "{stage:?} progress never completed");
    }

    let duration = audio.len() as f64 / 16_000.0;
    let der = der(&reference, num_ref, &out.turns, duration);
    println!(
        "speakers={} (reference {num_ref}) turns={} DER={:.2}% (prototype {:.2}%) load={:.2}s diarize={:.2}s",
        out.centroids.len(),
        out.turns.len(),
        der * 100.0,
        PROTOTYPE_DER * 100.0,
        loaded.as_secs_f64(),
        diarized.as_secs_f64()
    );

    assert_eq!(out.centroids.len(), num_ref);
    assert_eq!(out.speech_secs.len(), num_ref);
    assert!(der <= PROTOTYPE_DER + 0.02, "DER {der:.4} exceeds prototype {PROTOTYPE_DER} + 2 points");
    assert_eq!(out.turns.len(), expected.len(), "turn count differs from the prototype");
    for (got, want) in out.turns.iter().zip(&expected) {
        assert_eq!(got.speaker, want.speaker, "{got:?} vs {want:?}");
        assert!((got.start - want.start).abs() < 0.02 && (got.end - want.end).abs() < 0.02, "{got:?} vs {want:?}");
    }
    for c in &out.centroids {
        let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4 && c.len() == meetily_diarization::EMBEDDING_DIM);
    }
}

#[test]
#[ignore = "needs MEETILY_DIARIZATION_MODELS_DIR"]
fn cancel_and_short_audio() {
    let engine = DiarizationEngine::load(&models_dir(), 2).expect("load models");
    let audio = read_wav(&fixture("wav"));
    let err = engine.diarize(&audio, &DiarizationConfig::fast(), &AtomicBool::new(true), &|_, _| {}).unwrap_err();
    assert!(matches!(err, DiarizationError::Cancelled), "{err:?}");

    let out = engine
        .diarize(&audio[..7_999], &DiarizationConfig::fast(), &AtomicBool::new(false), &|_, _| {})
        .expect("short audio");
    assert!(out.turns.is_empty() && out.centroids.is_empty());

    let silence = vec![0f32; 16_000 * 5];
    let out =
        engine.diarize(&silence, &DiarizationConfig::fast(), &AtomicBool::new(false), &|_, _| {}).expect("silence");
    assert!(out.turns.is_empty() && out.centroids.is_empty());
}

#[test]
fn missing_models_are_reported() {
    let dir = tempfile::tempdir().expect("temp dir");
    match DiarizationEngine::load(dir.path(), 1) {
        Err(DiarizationError::ModelsMissing(names)) => assert_eq!(names.len(), 2),
        other => panic!("expected ModelsMissing, got {:?}", other.err()),
    }
}
