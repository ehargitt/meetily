//! Linking local speakers into global speakers.
//!
//! Average-linkage agglomerative clustering on cosine distance, computed with the
//! nearest-neighbour-chain algorithm (O(n²) time, n(n-1)/2 f32 memory). Average linkage is
//! reducible, so NN-chain yields the same dendrogram as the naive algorithm, and cutting at a
//! threshold equals applying every merge at or below it (scipy `fcluster(criterion="distance")`).
//! Clusters that are too small are then folded into the nearest large one (pyannote 3.1
//! `min_cluster_size`), and each window's local speakers are mapped one-to-one onto the result.

use std::collections::HashMap;

use crate::segmentation::NUM_LOCAL;
use crate::voiceprint::{cosine, l2_normalize};

/// Clean frames (~1 s) an embedding needs to take part in clustering; shorter ones are only
/// assigned to the resulting clusters.
const LONG_EMBED_FRAMES: usize = 60;

/// An embedded local speaker.
pub(crate) struct Embedded {
    pub chunk: usize,
    pub local: usize,
    pub clean_frames: usize,
    pub emb: Vec<f32>,
}

pub(crate) struct ClusterParams {
    pub threshold: f32,
    pub num_speakers: Option<usize>,
    pub min_cluster_size: usize,
    pub min_cluster_frac: f32,
}

/// pyannote folds clusters under 12 members at a 1 s step; scaling by the step keeps the same
/// floor in seconds of speech (a fixed 12 at 3 s collapsed a 9-voice test to 5 speakers).
pub(crate) fn min_cluster_size(step_secs: f32) -> usize {
    ((12.0 / step_secs).round() as usize).max(2)
}

/// Condensed upper-triangular distance matrix.
struct Condensed {
    n: usize,
    d: Vec<f32>,
}

impl Condensed {
    fn idx(&self, i: usize, j: usize) -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        self.n * i - i * (i + 1) / 2 + (j - i - 1)
    }
    fn get(&self, i: usize, j: usize) -> f32 {
        self.d[self.idx(i, j)]
    }
    fn set(&mut self, i: usize, j: usize, v: f32) {
        let k = self.idx(i, j);
        self.d[k] = v;
    }
}

/// Average-linkage merges `(a, b, height)` in discovery order; `a`/`b` are cluster
/// representatives (original indices).
fn nn_chain(mut dm: Condensed) -> Vec<(usize, usize, f32)> {
    let n = dm.n;
    let mut size = vec![1usize; n];
    let mut active = vec![true; n];
    let mut merges = Vec::with_capacity(n.saturating_sub(1));
    let mut chain: Vec<usize> = Vec::new();
    let mut remaining = n;
    while remaining > 1 {
        if chain.is_empty() {
            chain.push(active.iter().position(|&a| a).expect("remaining > 1"));
        }
        loop {
            let a = *chain.last().expect("chain is non-empty");
            let prev = chain.len().checked_sub(2).map(|i| chain[i]);
            // Preferring the previous chain element on ties guarantees termination.
            let mut best = prev.unwrap_or(usize::MAX);
            let mut best_d = prev.map(|p| dm.get(a, p)).unwrap_or(f32::INFINITY);
            for (k, &is_active) in active.iter().enumerate() {
                if k == a || !is_active {
                    continue;
                }
                let d = dm.get(a, k);
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            if Some(best) == prev {
                chain.truncate(chain.len() - 2);
                let (x, y) = (a.min(best), a.max(best));
                merges.push((x, y, best_d));
                // Lance-Williams update for average linkage, merging y into x.
                for (k, &is_active) in active.iter().enumerate() {
                    if !is_active || k == x || k == y {
                        continue;
                    }
                    let (dx, dy) = (dm.get(k, x), dm.get(k, y));
                    let v = (size[x] as f32 * dx + size[y] as f32 * dy) / (size[x] + size[y]) as f32;
                    dm.set(k, x, v);
                }
                size[x] += size[y];
                active[y] = false;
                remaining -= 1;
                break;
            }
            chain.push(best);
        }
    }
    merges
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Relabel arbitrary cluster ids to 0..k in order of first appearance; returns k.
fn compact(labels: &mut [usize]) -> usize {
    let mut map = HashMap::new();
    for l in labels.iter_mut() {
        let next = map.len();
        *l = *map.entry(*l).or_insert(next);
    }
    map.len()
}

/// Flat average-linkage clusters cut at cosine distance `threshold`, or into exactly
/// `num_clusters` when given. Labels are 0..k in order of first appearance.
pub(crate) fn ahc(embs: &[Vec<f32>], threshold: f32, num_clusters: Option<usize>) -> Vec<usize> {
    let n = embs.len();
    if n <= 1 {
        return vec![0; n];
    }
    let mut d = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            d.push((1.0 - cosine(&embs[i], &embs[j])).max(0.0));
        }
    }
    let mut merges = nn_chain(Condensed { n, d });
    merges.sort_by(|a, b| a.2.total_cmp(&b.2));
    let apply = match num_clusters {
        Some(k) => n.saturating_sub(k.max(1)),
        None => merges.iter().take_while(|m| m.2 <= threshold).count(),
    };
    let mut parent: Vec<usize> = (0..n).collect();
    for &(a, b, _) in merges.iter().take(apply) {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        parent[rb] = ra;
    }
    let mut labels: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    compact(&mut labels);
    labels
}

/// Mean of the L2-normalised members of each of the `k` clusters, re-normalised.
fn centroids(embs: &[Vec<f32>], labels: &[usize], k: usize) -> Vec<Vec<f32>> {
    let dim = embs.first().map_or(0, Vec::len);
    let mut c = vec![vec![0f32; dim]; k];
    for (e, &l) in embs.iter().zip(labels) {
        let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for (a, b) in c[l].iter_mut().zip(e) {
            *a += b / norm;
        }
    }
    c.iter_mut().for_each(|v| l2_normalize(v));
    c
}

/// Fold clusters with fewer than `min_size` members or less than `min_frac` of all clean frames
/// into the most similar large cluster. Returns the new cluster count.
fn fold_small_clusters(
    embs: &[Vec<f32>],
    mass: &[usize],
    labels: &mut [usize],
    k: usize,
    min_size: usize,
    min_frac: f32,
) -> usize {
    let mut sizes = vec![0usize; k];
    let mut cluster_mass = vec![0usize; k];
    for (&l, &m) in labels.iter().zip(mass) {
        sizes[l] += 1;
        cluster_mass[l] += m;
    }
    let total: usize = cluster_mass.iter().sum();
    let is_small = |c: usize| sizes[c] < min_size || (cluster_mass[c] as f32) < min_frac * total as f32;
    let large: Vec<usize> = (0..k).filter(|&c| !is_small(c)).collect();
    if large.is_empty() || large.len() == k {
        return k;
    }
    let cents = centroids(embs, labels, k);
    for (e, l) in embs.iter().zip(labels.iter_mut()) {
        if is_small(*l) {
            *l = *large
                .iter()
                .max_by(|&&a, &&b| cosine(e, &cents[a]).total_cmp(&cosine(e, &cents[b])))
                .expect("large is non-empty");
        }
    }
    compact(labels)
}

/// Cluster the embeddings into global speakers and return one L2-normalised centroid per speaker.
/// Only embeddings with at least ~1 s of clean speech are clustered, unless fewer than two exist.
pub(crate) fn global_centroids(items: &[Embedded], params: &ClusterParams) -> Vec<Vec<f32>> {
    let long: Vec<&Embedded> = items.iter().filter(|it| it.clean_frames >= LONG_EMBED_FRAMES).collect();
    let train: Vec<&Embedded> = if long.len() >= 2 { long } else { items.iter().collect() };
    let embs: Vec<Vec<f32>> = train.iter().map(|it| it.emb.clone()).collect();
    let mass: Vec<usize> = train.iter().map(|it| it.clean_frames).collect();
    let mut labels = ahc(&embs, params.threshold, params.num_speakers);
    let mut k = labels.iter().max().map_or(0, |m| m + 1);
    // An explicit speaker count overrides the small-cluster heuristic.
    if params.num_speakers.is_none() && k > 1 {
        k = fold_small_clusters(&embs, &mass, &mut labels, k, params.min_cluster_size, params.min_cluster_frac);
    }
    centroids(&embs, &labels, k)
}

/// Map each window's local speakers to global speakers, choosing per window the one-to-one
/// mapping that maximises total cosine similarity (brute force over at most three locals). When
/// there are fewer global speakers than locals in a window, locals may share a speaker.
pub(crate) fn assign_chunks(
    items: &[Embedded],
    num_chunks: usize,
    cents: &[Vec<f32>],
) -> Vec<[Option<usize>; NUM_LOCAL]> {
    let mut chunk_map = vec![[None; NUM_LOCAL]; num_chunks];
    if cents.is_empty() {
        return chunk_map;
    }
    let mut by_chunk: Vec<Vec<&Embedded>> = vec![Vec::new(); num_chunks];
    for it in items {
        by_chunk[it.chunk].push(it);
    }
    for (c, members) in by_chunk.iter().enumerate() {
        let sims: Vec<Vec<f32>> =
            members.iter().map(|it| cents.iter().map(|ct| cosine(&it.emb, ct)).collect()).collect();
        let mut best = (f32::MIN, Vec::new());
        best_mapping(&sims, cents.len(), &mut Vec::new(), 0.0, &mut best);
        for (it, &g) in members.iter().zip(&best.1) {
            chunk_map[c][it.local] = Some(g);
        }
    }
    chunk_map
}

fn best_mapping(sims: &[Vec<f32>], k: usize, cur: &mut Vec<usize>, score: f32, best: &mut (f32, Vec<usize>)) {
    let p = cur.len();
    if p == sims.len() {
        if score > best.0 {
            *best = (score, cur.clone());
        }
        return;
    }
    let exclusive = k >= sims.len();
    for g in 0..k {
        if exclusive && cur.contains(&g) {
            continue;
        }
        cur.push(g);
        best_mapping(sims, k, cur, score + sims[p][g], best);
        cur.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so the tests need no RNG dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        fn vec(&mut self, dim: usize) -> Vec<f32> {
            (0..dim).map(|_| self.next()).collect()
        }
    }

    fn blob(rng: &mut Rng, center: &[f32], n: usize, spread: f32) -> Vec<Vec<f32>> {
        (0..n).map(|_| center.iter().map(|c| c + spread * rng.next()).collect()).collect()
    }

    fn axis(dim: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    }

    fn three_blobs(rng: &mut Rng) -> Vec<Vec<f32>> {
        let mut embs = Vec::new();
        for c in 0..3 {
            embs.extend(blob(rng, &axis(16, c), 8, 0.15));
        }
        embs
    }

    #[test]
    fn three_separated_blobs_give_three_clusters() {
        let embs = three_blobs(&mut Rng(7));
        let labels = ahc(&embs, 0.6, None);
        assert_eq!(labels.iter().max(), Some(&2));
        for c in 0..3 {
            let block = &labels[c * 8..(c + 1) * 8];
            assert!(block.iter().all(|&l| l == block[0]), "blob {c} split: {labels:?}");
        }
    }

    #[test]
    fn num_speakers_override_is_honoured() {
        let embs = three_blobs(&mut Rng(11));
        assert_eq!(ahc(&embs, 0.6, Some(2)).iter().max(), Some(&1));
        assert_eq!(ahc(&embs, 0.6, Some(5)).iter().max(), Some(&4));

        let items: Vec<Embedded> = embs
            .into_iter()
            .enumerate()
            .map(|(i, emb)| Embedded { chunk: i, local: 0, clean_frames: 100, emb })
            .collect();
        // An explicit count is not undone by folding small clusters.
        let params =
            ClusterParams { threshold: 0.6, num_speakers: Some(4), min_cluster_size: 12, min_cluster_frac: 0.02 };
        assert_eq!(global_centroids(&items, &params).len(), 4);
    }

    /// Naive O(n³) average linkage: merge the closest pair, where cluster distance is the mean
    /// pairwise distance between original members.
    fn naive_heights(embs: &[Vec<f32>]) -> Vec<f32> {
        let dist = |i: usize, j: usize| (1.0 - cosine(&embs[i], &embs[j])).max(0.0) as f64;
        let mut clusters: Vec<Vec<usize>> = (0..embs.len()).map(|i| vec![i]).collect();
        let mut heights = Vec::new();
        while clusters.len() > 1 {
            let mut best = (f64::INFINITY, 0, 0);
            for a in 0..clusters.len() {
                for b in a + 1..clusters.len() {
                    let sum: f64 = clusters[a]
                        .iter()
                        .flat_map(|&i| clusters[b].iter().map(move |&j| (i, j)))
                        .map(|(i, j)| dist(i, j))
                        .sum();
                    let d = sum / (clusters[a].len() * clusters[b].len()) as f64;
                    if d < best.0 {
                        best = (d, a, b);
                    }
                }
            }
            let merged = clusters.remove(best.2);
            clusters[best.1].extend(merged);
            heights.push(best.0 as f32);
        }
        heights
    }

    #[test]
    fn nn_chain_matches_naive_average_linkage() {
        let mut rng = Rng(12345);
        for trial in 0..5 {
            let embs: Vec<Vec<f32>> = (0..25).map(|_| rng.vec(8)).collect();
            let n = embs.len();
            let mut d = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    d.push((1.0 - cosine(&embs[i], &embs[j])).max(0.0));
                }
            }
            let mut fast: Vec<f32> = nn_chain(Condensed { n, d }).iter().map(|m| m.2).collect();
            fast.sort_by(f32::total_cmp);
            let mut slow = naive_heights(&embs);
            slow.sort_by(f32::total_cmp);
            for (a, b) in fast.iter().zip(&slow) {
                assert!((a - b).abs() < 1e-4, "trial {trial}: {fast:?} vs {slow:?}");
            }
            // Same flat partition at a mid-range cut.
            let labels = ahc(&embs, 0.9, None);
            let expected_k = n - slow.iter().filter(|&&h| h <= 0.9).count();
            assert_eq!(labels.iter().max().unwrap() + 1, expected_k, "trial {trial}");
        }
    }

    #[test]
    fn fold_threshold_scales_with_step() {
        assert_eq!(min_cluster_size(3.0), 4);
        assert_eq!(min_cluster_size(1.0), 12);
        assert_eq!(min_cluster_size(10.0), 2);

        // At a 3 s step a 3-member cluster is folded, a 4-member cluster is kept.
        let mut rng = Rng(99);
        let mut embs = blob(&mut rng, &axis(16, 0), 20, 0.1);
        embs.extend(blob(&mut rng, &axis(16, 1), 4, 0.1));
        embs.extend(blob(&mut rng, &axis(16, 2), 3, 0.1));
        let items: Vec<Embedded> = embs
            .into_iter()
            .enumerate()
            .map(|(i, emb)| Embedded { chunk: i, local: 0, clean_frames: 100, emb })
            .collect();
        let params = ClusterParams {
            threshold: 0.6,
            num_speakers: None,
            min_cluster_size: min_cluster_size(3.0),
            min_cluster_frac: 0.02,
        };
        assert_eq!(global_centroids(&items, &params).len(), 2);
    }

    #[test]
    fn small_share_of_speech_is_folded() {
        let mut rng = Rng(5);
        let mut embs = blob(&mut rng, &axis(16, 0), 10, 0.1);
        embs.extend(blob(&mut rng, &axis(16, 1), 10, 0.1));
        let items: Vec<Embedded> = embs
            .into_iter()
            .enumerate()
            // The second voice has 600 of 10 500 clean frames (5.7%).
            .map(|(i, emb)| Embedded { chunk: i, local: 0, clean_frames: if i < 10 { 990 } else { 60 }, emb })
            .collect();
        let params = ClusterParams { threshold: 0.6, num_speakers: None, min_cluster_size: 2, min_cluster_frac: 0.02 };
        assert_eq!(global_centroids(&items, &params).len(), 2);
        let params = ClusterParams { min_cluster_frac: 0.1, ..params };
        assert_eq!(global_centroids(&items, &params).len(), 1);
    }

    #[test]
    fn locals_in_one_chunk_never_share_a_global_speaker() {
        let cents = vec![axis(4, 0), axis(4, 1), axis(4, 2)];
        // Both locals of chunk 0 are closest to speaker 0; the second is a bit less so.
        let items = vec![
            Embedded { chunk: 0, local: 0, clean_frames: 100, emb: vec![1.0, 0.1, 0.0, 0.0] },
            Embedded { chunk: 0, local: 2, clean_frames: 100, emb: vec![0.9, 0.5, 0.0, 0.0] },
            Embedded { chunk: 1, local: 1, clean_frames: 100, emb: vec![0.0, 0.0, 1.0, 0.0] },
        ];
        let map = assign_chunks(&items, 2, &cents);
        assert_eq!(map[0], [Some(0), None, Some(1)]);
        assert_eq!(map[1], [None, Some(2), None]);

        let mut rng = Rng(3);
        let cents: Vec<Vec<f32>> = (0..4).map(|_| rng.vec(8)).collect();
        for chunk in 0..50 {
            let items: Vec<Embedded> =
                (0..3).map(|local| Embedded { chunk: 0, local, clean_frames: 100, emb: rng.vec(8) }).collect();
            let map = assign_chunks(&items, 1, &cents)[0];
            let globals: Vec<usize> = map.iter().flatten().copied().collect();
            let mut unique = globals.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), 3, "chunk {chunk}: {map:?}");
        }
    }
}
