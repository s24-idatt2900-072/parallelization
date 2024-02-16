// build.rs
use std::process::Command;

fn main() {
    // Try to detect Python version using the `python` command.
    let output = Command::new("python3")
        .arg("--version")
        .output()
        .expect("Failed to execute python command");

    let version_str = String::from_utf8_lossy(&output.stdout);
    // Simple parsing to extract the major and minor version numbers.
    // This will need to be adapted based on the output format and what's installed (python2, python3, etc.)
    let version = version_str.trim().split_whitespace().nth(1).unwrap_or("3.10"); // Default to 3.10
    let versions: Vec<&str> = version.split('.').collect();
    let major = versions[0];
    let minor = versions[1];

    let lib_name = format!("python{}.{}", major, minor);

    // Print out the cargo commands to link against the detected Python version.
    //println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib={}", lib_name);
    //println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    //println!("cargo:rustc-link-lib=python3.10");
}