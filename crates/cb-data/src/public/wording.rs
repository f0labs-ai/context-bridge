//! How a question is worded.
//!
//! A question is one of a few things a caller wants to know about a product
//! (is it having problems now, what is the status, is it fixed, is there a
//! workaround), said in one of several ways, and it names only the product:
//! a component of the status page or the service itself. Nothing in it is
//! taken from the update that answers it.

use super::text::{contains_run, words};
use rand::Rng;

/// What the caller wants to know.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Intent {
    /// Is it having problems now?
    Now,
    /// What is the status?
    Status,
    /// Is it fixed?
    Fixed,
    /// Is there a workaround?
    Workaround,
}

/// Where the product's name goes in a phrasing.
const SLOT: &str = "{x}";

/// A run of this many words shared between a phrasing and the answering
/// update counts as copying it.
pub const COPIED_RUN: usize = 4;

/// The ways of asking for one intent. The name always follows a preposition or
/// stands as an object, so a phrasing reads right for "GitHub Actions" and for
/// "GitHub pull requests" alike.
pub fn pool(intent: Intent) -> &'static [&'static str] {
    match intent {
        Intent::Now => &[
            "Is there a problem with {x} right now?",
            "Are there any known issues with {x} at the moment?",
            "I'm having trouble with {x}. Is something going on?",
            "Is something wrong with {x} right now?",
            "Is there an outage affecting {x} at the moment?",
            "I can't get {x} to work. Is there an incident right now?",
            "Are you seeing problems with {x} right now?",
            "Is anything currently broken with {x}?",
        ],
        Intent::Status => &[
            "What is the current status of {x}?",
            "What's the latest on the problem with {x}?",
            "Do you have an update on the issue with {x}?",
            "Where do things stand with {x}?",
            "What's happening with {x}?",
            "Can you tell me the latest about the incident affecting {x}?",
            "What do we know so far about the trouble with {x}?",
            "Any news on {x}?",
        ],
        Intent::Fixed => &[
            "Is the problem with {x} fixed?",
            "Has the issue with {x} been resolved?",
            "Is the incident affecting {x} over?",
            "Did the trouble with {x} get sorted out?",
            "Can I rely on {x} again, or is it still broken?",
            "Are things back to normal with {x}?",
            "Has service been restored for {x}?",
            "Is it safe to use {x} again?",
        ],
        Intent::Workaround => &[
            "Is there a workaround for the problem with {x}?",
            "What can I do in the meantime about the problem with {x}?",
            "Is there any way around the issue with {x}?",
            "How can I keep working while the incident with {x} is going on?",
            "Is there anything I can do myself about the trouble with {x}?",
            "Do you have a temporary fix for the issue with {x}?",
            "What should I do until the problem with {x} is fixed?",
            "Can I work around the outage affecting {x}?",
        ],
    }
}

/// Whether a phrasing shares a run of [`COPIED_RUN`] words with `answer`
/// (lower-cased words of the answering update). The product's name is not
/// part of the phrasing, so naming the product is never copying.
fn copies(phrasing: &str, answer: &[String]) -> bool {
    phrasing.split(SLOT).any(|part| {
        let part = words(part);
        part.windows(COPIED_RUN).any(|run| contains_run(answer, run))
    })
}

/// One question for `intent` about `name`, worded so that it does not copy
/// `answer`, the body of the update that answers it. `None` when every
/// phrasing of the intent would copy it.
pub fn ask(intent: Intent, name: &str, answer: &str, rng: &mut impl Rng) -> Option<String> {
    let answer = words(answer);
    let usable: Vec<&str> = pool(intent)
        .iter()
        .copied()
        .filter(|p| !copies(p, &answer))
        .collect();
    if usable.is_empty() {
        return None;
    }
    let phrasing = usable[rng.random_range(0..usable.len())];
    Some(phrasing.replace(SLOT, name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::collections::BTreeSet;

    const INTENTS: [Intent; 4] = [Intent::Now, Intent::Status, Intent::Fixed, Intent::Workaround];

    #[test]
    fn every_intent_has_several_phrasings_with_one_slot_each() {
        let mut all = BTreeSet::new();
        for intent in INTENTS {
            assert!(pool(intent).len() >= 8, "{intent:?}");
            for phrasing in pool(intent) {
                assert_eq!(phrasing.matches(SLOT).count(), 1, "{phrasing}");
                assert!(phrasing.ends_with('?'), "{phrasing}");
                assert!(all.insert(*phrasing), "said twice: {phrasing}");
            }
        }
    }

    #[test]
    fn a_question_names_the_product_and_the_seed_fixes_the_wording() {
        let ask_with = |seed| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            ask(
                Intent::Now,
                "GitHub Actions",
                "We are investigating delays.",
                &mut rng,
            )
            .unwrap()
        };
        assert!(ask_with(1).contains("GitHub Actions"));
        assert!(!ask_with(1).contains(SLOT));
        assert_eq!(ask_with(1), ask_with(1));
        let seen: BTreeSet<String> = (0..64).map(ask_with).collect();
        assert_eq!(seen.len(), pool(Intent::Now).len(), "every phrasing is reachable");
    }

    #[test]
    fn a_phrasing_that_would_copy_the_answer_is_not_used() {
        // The update happens to say "what is the current status of".
        let answer = "Customers asked what is the current status of the fix. We are still investigating.";
        for seed in 0..64 {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let question = ask(Intent::Status, "GitHub Pages", answer, &mut rng).unwrap();
            assert_ne!(question, "What is the current status of GitHub Pages?");
        }
        // Naming the product is not copying, however long the name.
        let answer =
            "We are investigating degraded performance for the Google Cloud infrastructure components.";
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let name = "the Google Cloud infrastructure components";
        assert!(ask(Intent::Now, name, answer, &mut rng).is_some());
    }

    #[test]
    fn no_question_is_asked_when_every_phrasing_would_copy() {
        // An answer that holds every phrasing of the intent, with "it" as the product.
        let answer: String = pool(Intent::Fixed)
            .iter()
            .map(|p| p.replace(SLOT, "it"))
            .collect::<Vec<_>>()
            .join(" ");
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        assert_eq!(ask(Intent::Fixed, "it", &answer, &mut rng), None);
    }
}
