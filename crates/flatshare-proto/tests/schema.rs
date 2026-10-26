use std::fs;
use std::path::PathBuf;

#[test]
fn checked_in_schemas_are_current() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("schema");
    for (name, schema) in flatshare_proto::schemas() {
        let path = dir.join(format!("{name}.schema.json"));
        let on_disk = fs::read_to_string(&path).unwrap_or_default();
        assert!(
            on_disk == flatshare_proto::schema_json(&schema),
            "{} is stale; regenerate with `cargo run -p flatshare-proto --bin flatshare-proto-schema`",
            path.display()
        );
    }
}
