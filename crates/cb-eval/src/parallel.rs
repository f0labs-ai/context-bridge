//! Spreading independent pieces of work over threads without changing any result.

use crate::error::EvalError;

/// How many threads to use: `requested`, or every core when it is 0.
pub fn threads_to_use(requested: usize) -> usize {
    match requested {
        0 => std::thread::available_parallelism().map_or(1, usize::from),
        n => n,
    }
}

/// `work` applied to every item on up to `threads` threads. The results keep
/// the items' order, so the outcome is the same for any number of threads.
pub fn map_in_parallel<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    work: impl Fn(&T) -> Result<R, EvalError> + Sync,
) -> Result<Vec<R>, EvalError> {
    let threads = threads.clamp(1, items.len().max(1));
    if threads == 1 {
        return items.iter().map(&work).collect();
    }
    let work = &work;
    let chunk = items.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(work).collect::<Result<Vec<R>, EvalError>>()))
            .collect();
        let mut out = Vec::with_capacity(items.len());
        for handle in handles {
            let part = handle
                .join()
                .map_err(|_| EvalError::invalid("a worker thread stopped without an answer"))??;
            out.extend(part);
        }
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_keep_the_order_of_the_items_for_any_number_of_threads() {
        let items: Vec<u64> = (0..103).collect();
        let square = |x: &u64| -> Result<u64, EvalError> { Ok(x * x) };
        let one = map_in_parallel(&items, 1, square).unwrap();
        for threads in [2, 7, 200] {
            assert_eq!(map_in_parallel(&items, threads, square).unwrap(), one);
        }
        assert_eq!(one[10], 100);
        assert!(map_in_parallel(&[] as &[u64], 4, square).unwrap().is_empty());
        assert!(threads_to_use(0) >= 1);
        assert_eq!(threads_to_use(3), 3);
    }

    #[test]
    fn the_first_error_is_returned() {
        let items: Vec<u64> = (0..20).collect();
        let fails_at_13 = |x: &u64| -> Result<u64, EvalError> {
            if *x == 13 {
                Err(EvalError::invalid("thirteen"))
            } else {
                Ok(*x)
            }
        };
        assert!(
            map_in_parallel(&items, 4, fails_at_13)
                .unwrap_err()
                .to_string()
                .contains("thirteen")
        );
        assert!(map_in_parallel(&items, 1, fails_at_13).is_err());
    }
}
