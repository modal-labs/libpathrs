// SPDX-License-Identifier: MPL-2.0 OR LGPL-3.0-or-later
/*
 * libpathrs: safe path resolution on Linux
 * Copyright (C) 2019-2025 SUSE LLC
 * Copyright (C) 2026 Aleksa Sarai <cyphar@cyphar.com>
 *
 * == MPL-2.0 ==
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Alternatively, this Source Code Form may also (at your option) be used
 * under the terms of the GNU Lesser General Public License Version 3, as
 * described below:
 *
 * == LGPL-3.0-or-later ==
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or (at
 *  your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY  or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License  for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#![forbid(unsafe_code)]

//! Process-wide retry policy for scoped `openat2(2)` lookups.

use std::time::{Duration, Instant};

use once_cell::sync::OnceCell;
use rustix::io::Errno;

/// Retry policy for `EAGAIN` from scoped `openat2(2)` lookups.
///
/// Racing renames or mount changes can prevent the kernel from verifying that
/// a lookup stays within its root. Retrying allows the kernel to repeat those
/// checks. This policy only retries `EAGAIN` from lookups using
/// `RESOLVE_IN_ROOT` or `RESOLVE_BENEATH`; other errors are returned
/// immediately. Exhaustion returns the last `EAGAIN` error.
///
/// Configure the process-wide policy with [`set_default_retry_policy`]. Each
/// syscall retry loop has its own budget: a multi-step operation can have
/// several such loops. This does not impose an overall operation deadline or
/// change the userspace resolver's behavior. Retries are immediate, without
/// backoff.
///
/// The default is [`RetryPolicy::Count`] with 128 retries, allowing 129 syscall
/// attempts. [`RetryPolicy::default`] always returns this built-in value; use
/// [`default_retry_policy`] to read the configured process-wide default.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RetryPolicy {
    /// Maximum number of retries after the initial attempt.
    ///
    /// Zero disables retries. This mode does not read the clock.
    Count(u32),
    /// Retry while less than this duration has elapsed since the first attempt.
    ///
    /// Uses a monotonic clock. Zero disables retries. The budget is checked
    /// after each retryable failure, before retrying. It cannot interrupt a
    /// syscall or prevent scheduling delays, and a successful syscall is
    /// accepted even if it finishes after the budget expires.
    Timeout(Duration),
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::Count(128)
    }
}

static DEFAULT_RETRY_POLICY: OnceCell<RetryPolicy> = OnceCell::new();

/// Get the process-wide retry policy for scoped `openat2(2)` lookups.
///
/// Returns [`RetryPolicy::default`] until [`set_default_retry_policy`]
/// succeeds. Reading the default does not prevent subsequent configuration.
pub fn default_retry_policy() -> RetryPolicy {
    DEFAULT_RETRY_POLICY.get().copied().unwrap_or_default()
}

/// Set the process-wide retry policy for scoped `openat2(2)` lookups once.
///
/// Applications should call this during startup, before serving requests.
/// This affects subsequent lookup loops, including operations on existing
/// handles, internal helpers, and C API calls through the same library
/// instance. Separately linked copies of libpathrs have independent defaults.
/// This function does not add a configuration entry point to the C API.
///
/// Each lookup loop snapshots its policy before its first attempt, so a loop
/// already in progress keeps its original budget. Libraries should leave this
/// setting to the application, since all users of this library instance share
/// it.
///
/// # Errors
///
/// Returns `Err(policy)` with the attempted policy if already configured, even
/// if the requested value is unchanged. Concurrent setters are safe; exactly
/// one succeeds. The setting cannot be reset.
///
/// ```rust
/// use pathrs::{set_default_retry_policy, RetryPolicy, Root};
/// use std::time::Duration;
///
/// set_default_retry_policy(RetryPolicy::Timeout(Duration::from_millis(10)))
///     .expect("configure libpathrs once at startup");
/// let root = Root::open(".")?;
/// let handle = root.resolve(".")?;
/// # let _ = handle;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn set_default_retry_policy(policy: RetryPolicy) -> Result<(), RetryPolicy> {
    DEFAULT_RETRY_POLICY.set(policy)
}

impl RetryPolicy {
    pub(crate) fn retry_openat2<T>(
        self,
        scoped: bool,
        operation: impl FnMut() -> Result<T, Errno>,
    ) -> Result<T, Errno> {
        self.retry_with_clock(scoped, operation, Instant::now)
    }

    fn retry_with_clock<T>(
        self,
        scoped: bool,
        mut operation: impl FnMut() -> Result<T, Errno>,
        mut now: impl FnMut() -> Instant,
    ) -> Result<T, Errno> {
        if !scoped {
            return operation();
        }
        // Count mode must not read the clock. Keep a remaining count so even
        // the largest supported retry limit cannot overflow.
        let mut remaining = match self {
            Self::Count(count) => count,
            Self::Timeout(_) => 0,
        };
        let start = match self {
            Self::Count(_) => None,
            Self::Timeout(_) => Some(now()),
        };
        loop {
            match operation() {
                Err(Errno::AGAIN) => match self {
                    Self::Count(_) if remaining > 0 => remaining -= 1,
                    Self::Timeout(timeout)
                        if now().saturating_duration_since(start.expect("timeout start"))
                            < timeout => {}
                    _ => return Err(Errno::AGAIN),
                },
                result => return result,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_default() {
        // Isolate the irreversible global setting from all other tests.
        const CHILD: &str = "LIBPATHRS_TEST_RETRY_DEFAULT_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "retry::tests::process_default", "--nocapture"])
                .env(CHILD, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        assert_eq!(default_retry_policy(), RetryPolicy::Count(128));
        let root = crate::Root::open(".").unwrap();
        let snapshot = default_retry_policy();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let setters: Vec<_> = (0..8)
            .map(|i| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let policy = RetryPolicy::Count(i);
                    barrier.wait();
                    (policy, set_default_retry_policy(policy))
                })
            })
            .collect();
        let mut winner = None;
        for setter in setters {
            let (policy, result) = setter.join().unwrap();
            match result {
                Ok(()) => assert!(winner.replace(policy).is_none()),
                Err(rejected) => assert_eq!(rejected, policy),
            }
        }
        let winner = winner.expect("one setter must succeed");
        assert_eq!(default_retry_policy(), winner);
        assert_eq!(set_default_retry_policy(winner), Err(winner));
        assert_eq!(snapshot, RetryPolicy::Count(128));
        assert_eq!(RetryPolicy::default(), RetryPolicy::Count(128));
        root.resolve(".").unwrap();
    }

    fn no_clock() -> Instant {
        panic!("count mode must not read the clock")
    }

    #[test]
    fn count_exhaustion() {
        for (policy, expected_attempts) in [
            (RetryPolicy::Count(0), 1),
            (RetryPolicy::Count(1), 2),
            (RetryPolicy::default(), 129),
            (RetryPolicy::Count(1024), 1025),
        ] {
            let mut attempts = 0;
            let result: Result<(), _> = policy.retry_with_clock(
                true,
                || {
                    attempts += 1;
                    Err(Errno::AGAIN)
                },
                no_clock,
            );
            assert_eq!(result, Err(Errno::AGAIN));
            assert_eq!(attempts, expected_attempts);
        }
    }

    #[test]
    fn count_success_and_other_errors() {
        for failures in [0, 1, 128, 1024] {
            for final_result in [
                Ok(42),
                Err(Errno::INTR),
                Err(Errno::XDEV),
                Err(Errno::NOENT),
            ] {
                let mut attempts = 0;
                let result = RetryPolicy::Count(u32::MAX).retry_with_clock(
                    true,
                    || {
                        attempts += 1;
                        if attempts <= failures {
                            Err(Errno::AGAIN)
                        } else {
                            final_result
                        }
                    },
                    no_clock,
                );
                assert_eq!(result, final_result);
                assert_eq!(attempts, failures + 1);
            }
        }
    }

    #[test]
    fn success_on_last_attempt() {
        let mut results = [Err(Errno::AGAIN), Ok(42)].into_iter();
        assert_eq!(
            RetryPolicy::Count(1).retry_with_clock(true, || results.next().unwrap(), no_clock),
            Ok(42)
        );
        assert!(results.next().is_none());
    }

    #[test]
    fn unscoped_lookup_does_not_retry() {
        for policy in [
            RetryPolicy::Count(10),
            RetryPolicy::Timeout(Duration::from_secs(1)),
        ] {
            let mut attempts = 0;
            let result: Result<(), _> = policy.retry_with_clock(
                false,
                || {
                    attempts += 1;
                    Err(Errno::AGAIN)
                },
                no_clock,
            );
            assert_eq!(result, Err(Errno::AGAIN));
            assert_eq!(attempts, 1);
        }
    }

    #[test]
    fn timeout_exhaustion() {
        let start = Instant::now();
        for (timeout_ms, ticks, expected_attempts) in [
            (0, vec![0, 0], 1),
            (10, vec![0, 10], 1),
            (10, vec![0, 3, 9, 10], 3),
            (10, vec![0, 11], 1),
        ] {
            let mut ticks = ticks.into_iter();
            let mut attempts = 0;
            let result: Result<(), _> = RetryPolicy::Timeout(Duration::from_millis(timeout_ms))
                .retry_with_clock(
                    true,
                    || {
                        attempts += 1;
                        Err(Errno::AGAIN)
                    },
                    || start + Duration::from_millis(ticks.next().expect("unexpected clock read")),
                );
            assert_eq!(result, Err(Errno::AGAIN));
            assert_eq!(attempts, expected_attempts);
            assert!(ticks.next().is_none());
        }
    }

    #[test]
    fn timeout_preserves_success_and_other_errors() {
        let start = Instant::now();
        for final_result in [Ok(42), Err(Errno::INTR), Err(Errno::XDEV)] {
            let mut results = [Err(Errno::AGAIN), final_result].into_iter();
            // No clock read after success or a non-retryable error, even if
            // that syscall might have completed after the timeout expired.
            let mut ticks = [start, start + Duration::from_millis(9)].into_iter();
            assert_eq!(
                RetryPolicy::Timeout(Duration::from_millis(10)).retry_with_clock(
                    true,
                    || results.next().unwrap(),
                    || ticks.next().expect("unexpected clock read"),
                ),
                final_result
            );
            assert!(results.next().is_none());
            assert!(ticks.next().is_none());
        }
    }

    #[test]
    fn timeout_budget_is_per_lookup() {
        let start = Instant::now();
        let policy = RetryPolicy::Timeout(Duration::from_millis(10));
        for offset in [0, 100] {
            let mut ticks = [offset, offset + 9].into_iter();
            let mut results = [Err(Errno::AGAIN), Ok(42)].into_iter();
            assert_eq!(
                policy.retry_with_clock(
                    true,
                    || results.next().unwrap(),
                    || start + Duration::from_millis(ticks.next().unwrap()),
                ),
                Ok(42)
            );
        }
    }
}
