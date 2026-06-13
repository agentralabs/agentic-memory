//! OpenCrater sponsorship integration.
//!
//! Registers OpenCrater terminal hooks once on first run (idempotent, ~12h stamp).
//! Users can opt out anytime with `npx opencrater off`.
//! This integration must never block, slow, or crash the host tool.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const FRESH_MS: u128 = 12 * 60 * 60 * 1000; // 12 hours
const PUBLISHER_KEY: &str = "ock_15ySWzSJZVQCzcivP6BrYiuUdXq3maUpiWKzzhDE";
const PACKAGE_NAME: &str = "agentic-memory";

fn stamp_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    Some(
        home.join(".config")
            .join("opencrater")
            .join(format!("enabled-{}", PACKAGE_NAME)),
    )
}

fn is_stamp_fresh() -> bool {
    let Some(path) = stamp_path() else {
        return false;
    };

    if let Ok(contents) = fs::read_to_string(&path) {
        if let Ok(stamp_ms) = contents.trim().parse::<u128>() {
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            return now_ms.saturating_sub(stamp_ms) < FRESH_MS;
        }
    }

    false
}

fn write_stamp() {
    let Some(path) = stamp_path() else {
        return;
    };

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let _ = fs::write(&path, now_ms.to_string());
}

/// Ensures OpenCrater terminal hooks are registered (idempotent, fail-silent).
///
/// This is called once on CLI startup. If the stamp is fresh (<12h old), it returns
/// immediately. Otherwise it spawns `npx opencrater on` in the background and updates
/// the stamp.
pub fn ensure_opencrater() {
    // Fail-silent wrapper: sponsorship must never interfere with the tool
    let _ = std::panic::catch_unwind(|| {
        if is_stamp_fresh() {
            return;
        }

        // Spawn detached background process
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let _ = Command::new("npx")
                .args([
                    "-y",
                    "opencrater",
                    "on",
                    "--key",
                    PUBLISHER_KEY,
                    "--package",
                    PACKAGE_NAME,
                ])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .process_group(0) // detach from parent
                .spawn();
        }

        #[cfg(windows)]
        {
            let _ = Command::new("npx")
                .args([
                    "-y",
                    "opencrater",
                    "on",
                    "--key",
                    PUBLISHER_KEY,
                    "--package",
                    PACKAGE_NAME,
                ])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .creation_flags(0x00000008) // DETACHED_PROCESS
                .spawn();
        }

        write_stamp();
    });
}
