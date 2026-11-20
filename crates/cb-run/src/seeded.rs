//! Seeded draws: the same seed gives the same sample and the same moments, on any machine.
//!
//! The stream is ChaCha20 keyed by the seed. Ranges and samples are worked out here from
//! the raw stream, not by a library's sampling routines, so a result depends on the seed
//! and on nothing that could change with a library version.

use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

/// A stream of draws fixed by its seed.
#[derive(Clone, Debug)]
pub struct Draws(ChaCha20Rng);

impl Draws {
    pub fn new(seed: u64) -> Self {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&seed.to_le_bytes());
        Self(ChaCha20Rng::from_seed(key))
    }

    /// A stream of its own for one named thing under a seed: what is drawn for the thing
    /// depends on the seed and its name alone, not on which other things exist.
    pub fn keyed(seed: u64, name: &str) -> Self {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&seed.to_le_bytes());
        key[8..16].copy_from_slice(&cb_types::fnv1a64(name.as_bytes()).to_le_bytes());
        key[16..24].copy_from_slice(&(name.len() as u64).to_le_bytes());
        Self(ChaCha20Rng::from_seed(key))
    }

    /// A whole number in `[0, n)`, every value equally likely. `n` of 0 gives 0.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        // Draws above the last whole multiple of `n` are thrown away, so no value is favoured.
        let last_fair = u64::MAX - (u64::MAX % n + 1) % n;
        loop {
            let draw = self.0.next_u64();
            if draw <= last_fair {
                return draw % n;
            }
        }
    }

    /// A whole number in `[low, high]`, every value equally likely; `low` when the range is empty.
    pub fn between(&mut self, low: i64, high: i64) -> i64 {
        if high <= low {
            return low;
        }
        let span = (i128::from(high) - i128::from(low) + 1) as u64;
        (i128::from(low) + i128::from(self.below(span))) as i64
    }

    /// `count` of the items (all of them when there are fewer), each set of that size
    /// equally likely, in the order they were drawn.
    pub fn sample<T>(&mut self, mut items: Vec<T>, count: usize) -> Vec<T> {
        let count = count.min(items.len());
        for i in 0..count {
            let pick = i + self.below((items.len() - i) as u64) as usize;
            items.swap(i, pick);
        }
        items.truncate(count);
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_draws_and_another_seed_others() {
        let take = |seed| {
            let mut draws = Draws::new(seed);
            (0..20).map(|_| draws.below(1_000_000)).collect::<Vec<_>>()
        };
        assert_eq!(take(42), take(42));
        assert_ne!(take(42), take(43));
    }

    #[test]
    fn a_named_stream_depends_on_the_seed_and_the_name_only() {
        let draw = |seed, name| Draws::keyed(seed, name).between(0, 1_000_000_000);
        assert_eq!(draw(42, "d7528b29"), draw(42, "d7528b29"));
        assert_ne!(draw(42, "d7528b29"), draw(42, "d7528b2a"));
        assert_ne!(draw(42, "d7528b29"), draw(43, "d7528b29"));
        assert_ne!(draw(42, "d7528b29"), Draws::new(42).between(0, 1_000_000_000));
    }

    #[test]
    fn a_draw_stays_in_its_range() {
        let mut draws = Draws::new(7);
        for _ in 0..2_000 {
            assert!(draws.below(3) < 3);
            let moment = draws.between(-5, 5);
            assert!((-5..=5).contains(&moment));
        }
        assert_eq!(draws.below(0), 0);
        assert_eq!(draws.below(1), 0);
        assert_eq!(draws.between(9, 9), 9);
        assert_eq!(draws.between(9, 3), 9);
        let wide = draws.between(i64::MIN / 2, i64::MAX / 2);
        assert!((i64::MIN / 2..=i64::MAX / 2).contains(&wide));
    }

    #[test]
    fn every_value_of_a_small_range_comes_up() {
        let mut draws = Draws::new(1);
        let mut seen = [0u32; 5];
        for _ in 0..5_000 {
            seen[draws.below(5) as usize] += 1;
        }
        // 1,000 expected each; a fair draw stays well within these bounds.
        assert!(seen.iter().all(|n| (850..=1150).contains(n)), "{seen:?}");
    }

    #[test]
    fn a_sample_has_the_asked_size_no_repeats_and_is_fixed_by_the_seed() {
        let items: Vec<u32> = (0..100).collect();
        let sample = Draws::new(42).sample(items.clone(), 10);
        assert_eq!(sample.len(), 10);
        let mut sorted = sample.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
        assert_eq!(sample, Draws::new(42).sample(items.clone(), 10));
        assert_ne!(sample, Draws::new(43).sample(items.clone(), 10));
        // Asking for more than there is gives all of them.
        let mut all = Draws::new(42).sample(items.clone(), 500);
        all.sort_unstable();
        assert_eq!(all, items);
    }
}
