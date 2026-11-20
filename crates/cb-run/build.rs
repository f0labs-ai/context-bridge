//! Records the git commit this tool is built from, so every run manifest can name it.

use std::process::Command;

/// One line of git's output, or nothing when git is absent or says nothing.
fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn main() {
    // A copy built without its history (the experiment machine gets the sources
    // alone) is told its commit by whoever copied it.
    println!("cargo:rerun-if-env-changed=CB_GIT_COMMIT");
    if let Ok(given) = std::env::var("CB_GIT_COMMIT")
        && !given.trim().is_empty()
    {
        println!("cargo:rustc-env=CB_RUN_GIT_COMMIT={}", given.trim());
        return;
    }
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    // Tracked files that differ from the commit: the build is then not exactly that commit.
    let changed = git(&["status", "--porcelain", "--untracked-files=no"]).is_some();
    let shown = if changed {
        format!("{commit}-dirty")
    } else {
        commit
    };
    println!("cargo:rustc-env=CB_RUN_GIT_COMMIT={shown}");

    for path in [
        "build.rs",
        "Cargo.toml",
        "src",
        "prompts",
        "../cb-types/src",
        "../../Cargo.lock",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
        println!("cargo:rerun-if-changed={git_dir}/logs/HEAD");
    }
}
