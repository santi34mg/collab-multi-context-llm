//! Archive Criterion data and compare its mean latency estimates.
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn collect(root: &Path, baseline: &str, rows: &mut Vec<Value>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        if path.file_name().and_then(|s| s.to_str()) == Some(baseline)
            && path.join("estimates.json").is_file()
        {
            let read = |file| -> Result<Value> {
                Ok(serde_json::from_slice(&fs::read(path.join(file))?)?)
            };
            rows.push(json!({"benchmark": read("benchmark.json")?["full_id"],
                "estimates": read("estimates.json")?, "sample": read("sample.json")?}));
        } else {
            collect(&path, baseline, rows)?;
        }
    }
    Ok(())
}

fn read(path: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn save(baseline: &str, output: &Path) -> Result<()> {
    let mut rows = Vec::new();
    collect(Path::new("target/criterion"), baseline, &mut rows)?;
    if rows.is_empty() {
        return Err("no matching Criterion baseline found".into());
    }
    rows.sort_by(|a, b| a["benchmark"].as_str().cmp(&b["benchmark"].as_str()));
    fs::write(
        output,
        serde_json::to_string_pretty(&json!({"baseline": baseline, "benchmarks": rows}))? + "\n",
    )?;
    Ok(())
}

fn run(binary: &str, name: &str, cpus: &str, filter: &str) -> Result<()> {
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') || name.is_empty() {
        return Err("baseline name must contain only ASCII letters, digits and hyphens".into());
    }
    let out = Path::new("docs/performance/data");
    fs::create_dir_all(out)?;
    let output = out.join(format!("{name}.json"));
    if output.exists() {
        return Err(format!("refusing to overwrite {}", output.display()).into());
    }
    let log = fs::File::create(out.join(format!("{name}.log")))?;
    let status = Command::new("taskset")
        .args([
            "-c",
            cpus,
            binary,
            "--bench",
            "--save-baseline",
            name,
            filter,
        ])
        .env_remove("ALLOCATION_REPORT")
        .env(
            "PERF_RESOURCE_REPORT",
            out.join(format!("resources-{name}.json")),
        )
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .status()?;
    if !status.success() {
        return Err(format!("Criterion failed: {status}").into());
    }
    save(name, &output)?;
    let allocation_path = out.join(format!("allocations-{name}.jsonl"));
    fs::File::create(&allocation_path)?;
    let status = Command::new(binary)
        .arg("--test")
        .env_remove("PERF_RESOURCE_REPORT")
        .env("ALLOCATION_REPORT", allocation_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !status.success() {
        return Err(format!("allocation probe failed: {status}").into());
    }
    let mut environment =
        json!({"binary": binary, "baseline": name, "cpus": cpus, "filter": filter});
    let build_metadata = Path::new(binary).with_extension("build.json");
    if build_metadata.is_file() {
        environment["build"] = serde_json::from_slice(&fs::read(build_metadata)?)?;
    }
    for (key, executable, arguments) in [
        ("rustc", "rustc", vec!["-Vv"]),
        ("cargo", "cargo", vec!["-V"]),
        ("kernel", "uname", vec!["-a"]),
        ("cpu", "lscpu", vec![]),
    ] {
        let output = Command::new(executable).args(arguments).output()?;
        environment[key] = String::from_utf8_lossy(&output.stdout).into_owned().into();
    }
    for key in [
        "RUSTFLAGS",
        "CARGO_BUILD_TARGET",
        "CARGO_PROFILE_RELEASE_LTO",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
    ] {
        environment[key] = std::env::var(key).ok().into();
    }
    fs::write(
        out.join(format!("environment-{name}.json")),
        serde_json::to_string_pretty(&environment)?,
    )?;
    println!("Saved Criterion samples, estimates, resources and allocations for {name}");
    Ok(())
}

fn build(variant: &str) -> Result<()> {
    if !["baseline", "optimized", "native", "thin-lto"].contains(&variant) {
        return Err("build variant must be baseline, optimized, native, or thin-lto".into());
    }
    let target = format!("target/performance/build-{variant}");
    let mut command = Command::new("cargo");
    command.args([
        "bench",
        "--bench",
        "performance",
        "--no-run",
        "--locked",
        "--offline",
        "--message-format=json",
        "--target-dir",
        &target,
    ]);
    if variant == "baseline" {
        command.args([
            "--manifest-path",
            "target/performance/original-project/Cargo.toml",
        ]);
    }
    if variant == "native" {
        command.env("RUSTFLAGS", "-C target-cpu=native");
    }
    if variant == "thin-lto" {
        command
            .env("CARGO_PROFILE_RELEASE_LTO", "thin")
            .env("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "1");
    }
    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()?;
    if !output.status.success() {
        return Err(format!("benchmark build failed: {}", output.status).into());
    }
    let executable = String::from_utf8(output.stdout)?
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find_map(|row| {
            (row["target"]["name"] == "performance")
                .then(|| row["executable"].as_str().map(str::to_owned))
                .flatten()
        })
        .ok_or("Cargo did not report the benchmark executable")?;
    let destination = format!("target/performance/{variant}-bench");
    fs::copy(&executable, &destination)?;
    fs::write(
        format!("{destination}.build.json"),
        serde_json::to_string_pretty(&json!({
            "variant": variant, "cargo_command": format!("{command:?}")
        }))?,
    )?;
    println!("Built {variant}: {destination}");
    Ok(())
}

fn describe(path: &str) -> Result<()> {
    let data = read(path)?;
    for row in data["benchmarks"].as_array().ok_or("missing benchmarks")? {
        let mean = row["estimates"]["mean"]["point_estimate"].as_f64().unwrap();
        let ci = &row["estimates"]["mean"]["confidence_interval"];
        let cv = 100.0
            * row["estimates"]["std_dev"]["point_estimate"]
                .as_f64()
                .unwrap()
            / mean;
        println!(
            "{}: {:.2} us, 95% CI [{:.2}, {:.2}], CV {:.1}%",
            row["benchmark"].as_str().unwrap(),
            mean / 1000.0,
            ci["lower_bound"].as_f64().unwrap() / 1000.0,
            ci["upper_bound"].as_f64().unwrap() / 1000.0,
            cv
        );
    }
    Ok(())
}

fn prepare_baseline(revision: &str) -> Result<()> {
    let root = Path::new("target/performance/original-project");
    if root.exists() {
        return Err("baseline directory already exists; use a fresh checkout to prepare it".into());
    }
    let files = Command::new("git")
        .args(["--no-pager", "ls-tree", "-r", "--name-only", revision])
        .output()?;
    if !files.status.success() {
        return Err("cannot list baseline revision".into());
    }
    for file in String::from_utf8(files.stdout)?.lines() {
        let path = root.join(file);
        fs::create_dir_all(path.parent().ok_or("invalid baseline path")?)?;
        let data = Command::new("git")
            .args(["--no-pager", "show", &format!("{revision}:{file}")])
            .output()?;
        if !data.status.success() {
            return Err(format!("cannot read baseline file {file}").into());
        }
        fs::write(path, data.stdout)?;
    }
    for file in ["Cargo.toml", "Cargo.lock", "benches/performance.rs"] {
        let path = root.join(file);
        fs::create_dir_all(path.parent().ok_or("invalid infrastructure path")?)?;
        fs::copy(file, path)?;
    }
    println!(
        "Prepared original production sources with identical benchmark infrastructure at {}",
        root.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("prepare-baseline") if args.len() == 3 => prepare_baseline(&args[2])?,
        Some("build") if args.len() == 3 => build(&args[2])?,
        Some("describe") if args.len() == 3 => describe(&args[2])?,
        Some("run") if (4..=6).contains(&args.len()) => {
            run(
                &args[2],
                &args[3],
                args.get(4).map_or("0", String::as_str),
                args.get(5).map_or("", String::as_str),
            )?;
        }
        Some("save") if args.len() == 4 => save(&args[2], Path::new(&args[3]))?,
        Some("compare") if args.len() == 4 => {
            let before = read(&args[2])?;
            let after = read(&args[3])?;
            let index: BTreeMap<_, _> = after["benchmarks"]
                .as_array()
                .ok_or("missing benchmarks")?
                .iter()
                .map(|row| (row["benchmark"].as_str().unwrap(), row))
                .collect();
            println!(
                "| Benchmark | Baseline µs | Optimized µs | Speedup | Latency reduction |\n|---|---:|---:|---:|---:|"
            );
            for row in before["benchmarks"]
                .as_array()
                .ok_or("missing benchmarks")?
            {
                let name = row["benchmark"].as_str().ok_or("missing benchmark name")?;
                if let Some(after) = index.get(name) {
                    let a = row["estimates"]["mean"]["point_estimate"].as_f64().unwrap();
                    let b = after["estimates"]["mean"]["point_estimate"]
                        .as_f64()
                        .unwrap();
                    println!(
                        "| {} | {:.2} | {:.2} | {:.2}× | {:.1}% |",
                        name.trim_end_matches("/operation"),
                        a / 1000.0,
                        b / 1000.0,
                        a / b,
                        100.0 * (1.0 - b / a)
                    );
                }
            }
        }
        _ => {
            return Err(
                "usage: performance_results run BINARY NAME [CPUS] [FILTER] | save NAME OUTPUT | compare BEFORE AFTER | describe DATA | build VARIANT | prepare-baseline REVISION".into(),
            );
        }
    }
    Ok(())
}
