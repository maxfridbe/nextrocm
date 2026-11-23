//! Infrastructure-only source/build/gate checks. Never opens a device.
use std::path::Path;
use std::process::Command;

#[test]
fn architecture_build_and_gate_contracts() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new("python3")
        .arg("-B")
        .arg("tooling/test_architecture.py")
        .current_dir(root)
        .output()
        .expect("architecture tests require python3 (standard library only)");
    assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
}
