//! Criterion workloads and separately enabled allocation accounting.
//! See docs/performance/REPORT.md.
use agent_collab::context::{self, ContextBudget};
use agent_collab::distillation::{self, TeacherSample};
use agent_collab::experiments::small_config;
use agent_collab::inference::{ChatMessage, GenerationConfig};
use agent_collab::memory::{MemoryDraft, MemoryKind, MemoryStore};
use agent_collab::model::TinyTransformer;
use agent_collab::state::{Engine, EngineConfig, Operation};
use agent_collab::training::{self, TrainingEpisode};
use criterion::{BatchSize, Criterion, SamplingMode, Throughput};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

struct CountingAllocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
fn count(size: usize) {
    if COUNT.load(Relaxed) {
        ALLOCATIONS.fetch_add(1, Relaxed);
        BYTES.fetch_add(size as u64, Relaxed);
    }
}
// SAFETY: every operation forwards its original pointer/layout unchanged to
// System. Counters neither dereference pointers nor allocate. Instrumentation
// lives only in this benchmark executable, never in the library allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocation_probe(name: &str, units: usize, mut operation: impl FnMut()) {
    if let Ok(path) = std::env::var("ALLOCATION_REPORT") {
        ALLOCATIONS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
        COUNT.store(true, Relaxed);
        operation();
        COUNT.store(false, Relaxed);
        let row = serde_json::json!({"benchmark": name, "units": units,
            "allocations": ALLOCATIONS.load(Relaxed), "allocated_bytes": BYTES.load(Relaxed)});
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(file, "{row}").unwrap();
    }
}

fn measure(c: &mut Criterion, name: &str, units: usize, mut operation: impl FnMut()) {
    allocation_probe(name, units, &mut operation);
    let mut group = c.benchmark_group(name);
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(units as u64));
    group.bench_function("operation", |b| b.iter(&mut operation));
    group.finish();
}

fn fixture(payload: usize, streams: usize) -> Engine {
    let mut engine = Engine::new(
        EngineConfig {
            streams,
            ..EngineConfig::default()
        },
        MemoryStore::new(1024 * 1024),
    )
    .unwrap();
    for stream in 0..streams {
        engine.seed_entry(stream, "a".repeat(payload)).unwrap();
    }
    engine
        .step_actions(vec![(
            0,
            Operation::Publish {
                target: None,
                content: "b".repeat(payload),
            },
        )])
        .unwrap();
    engine
}

fn main() {
    let start = Instant::now();
    let mut c = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1))
        .nresamples(10_000)
        .noise_threshold(0.03)
        .configure_from_args();
    let budget = ContextBudget::default();
    let config = small_config();
    let model = TinyTransformer::new(config.clone(), 42).unwrap();
    for (size, payload) in [("small", 16), ("medium", 96), ("large", 256)] {
        let engine = fixture(payload, 2);
        let input = context::encode(&engine, &config, &budget).unwrap().input;
        eprintln!(
            "{size}: {} encoded tokens, {} queries",
            input.tokens.len(),
            input.queries.len()
        );
        measure(
            &mut c,
            &format!("encode/{size}"),
            input.tokens.len(),
            || {
                black_box(context::encode(black_box(&engine), &config, &budget).unwrap());
            },
        );
        measure(&mut c, &format!("predict/{size}"), 2, || {
            black_box(model.predict(black_box(&input)).unwrap());
        });
        measure(&mut c, &format!("round/{size}"), 2, || {
            black_box(context::predict_round(black_box(&engine), &model, &budget).unwrap());
        });
        let name = format!("train_step/{size}");
        let mut student = model.clone();
        allocation_probe(&name, 2, || {
            black_box(
                student
                    .train_step(black_box(&input), &[65, 66], 0.001)
                    .unwrap(),
            );
        });
        let mut group = c.benchmark_group(&name);
        group
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(2));
        // Fixed starting weights on every iteration; cloning/setup and input
        // destruction are excluded by Criterion's batched-reference runner.
        group.bench_function("operation", |b| {
            b.iter_batched_ref(
                || model.clone(),
                |student| {
                    black_box(
                        student
                            .train_step(black_box(&input), &[65, 66], 0.001)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
    {
        let engine = fixture(256, 2);
        let input = context::encode(&engine, &config, &budget).unwrap().input;
        let mut eviction = vec![0_u8; 32 * 1024 * 1024];
        let mut group = c.benchmark_group("predict_evicted/large");
        group
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(2));
        group.bench_function("operation", |b| {
            b.iter_batched_ref(
                || {
                    for line in eviction.chunks_mut(64) {
                        line[0] = line[0].wrapping_add(1);
                    }
                    black_box(&eviction);
                },
                |()| {
                    black_box(model.predict(black_box(&input)).unwrap());
                },
                BatchSize::PerIteration,
            )
        });
        group.finish();
    }
    for streams in [1, 2, 4, 8] {
        let mut config = config.clone();
        config.max_streams = streams + 1;
        let model = TinyTransformer::new(config, 42).unwrap();
        let engine = fixture(32, streams);
        measure(&mut c, &format!("stream_round/{streams}"), streams, || {
            black_box(context::predict_round(black_box(&engine), &model, &budget).unwrap());
        });
    }
    for rounds in [64, 512, 4096] {
        measure(&mut c, &format!("byte_replay/{rounds}"), rounds * 2, || {
            let mut engine = fixture(16, 2);
            for _ in 0..rounds {
                black_box(
                    engine
                        .step_tokens(black_box(vec![(0, b'x'), (1, b'y')]))
                        .unwrap(),
                );
            }
            black_box(engine);
        });
    }
    let episode: Vec<TrainingEpisode> =
        serde_json::from_str(include_str!("../examples/collaboration-dataset.json")).unwrap();
    measure(&mut c, "validate_episode/example", 1, || {
        black_box(training::validate_episode(
            black_box(&episode[0]),
            &config,
            &budget,
        ))
        .unwrap();
    });
    measure(&mut c, "train_episode/example", 1, || {
        let mut student = TinyTransformer::new(config.clone(), 42).unwrap();
        black_box(
            training::train_episode(&mut student, black_box(&episode[0]), 0.001, &budget).unwrap(),
        );
    });
    let samples = vec![TeacherSample {
        format_version: 1,
        id: "bench".into(),
        messages: vec![ChatMessage::user("What is 17 plus 25? Explain briefly.")],
        teacher: "offline-fixture".into(),
        response: "17 plus 25 equals 42.".into(),
        prompt_tokens: 12,
        completion_tokens: 8,
        finish_reason: "stop".into(),
        generation: GenerationConfig::default(),
    }];
    measure(
        &mut c,
        "distill/one_epoch",
        samples[0].response.len(),
        || {
            let mut student = TinyTransformer::new(config.clone(), 42).unwrap();
            black_box(
                distillation::train_samples(&mut student, black_box(&samples), 1, 0.001).unwrap(),
            );
        },
    );
    for entries in [16, 256, 4096] {
        let mut store = MemoryStore::new(1024 * 1024);
        for index in 0..entries {
            store
                .commit(MemoryDraft {
                    content: format!("entry {index} alpha beta Café 東京 {}", index % 7),
                    scope: if index % 4 == 0 { "other" } else { "default" }.into(),
                    kind: MemoryKind::Fact,
                    source_references: vec!["fixture".into()],
                })
                .unwrap();
        }
        measure(&mut c, &format!("memory_query/{entries}"), entries, || {
            black_box(store.query("default", black_box("BETA café absent beta"), 4, 256));
        });
    }
    // Independent requests, not parallelizing a single short model operation.
    // Pool setup is excluded; synchronization is intentionally included.
    for threads in [1, 2, 4, 8, 16] {
        let engine = fixture(96, 2);
        let input = context::encode(&engine, &config, &budget).unwrap().input;
        let barrier = std::sync::Barrier::new(threads + 1);
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let shared_model = &model;
            for _ in 0..threads {
                let (input, barrier, stop) = (&input, &barrier, &stop);
                scope.spawn(move || {
                    loop {
                        barrier.wait();
                        if stop.load(Relaxed) {
                            break;
                        }
                        for _ in 0..32 {
                            black_box(shared_model.predict(black_box(input)).unwrap());
                        }
                        barrier.wait();
                    }
                });
            }
            measure(
                &mut c,
                &format!("concurrent/{threads}"),
                threads * 32 * 2,
                || {
                    barrier.wait();
                    barrier.wait();
                },
            );
            stop.store(true, Relaxed);
            barrier.wait();
        });
    }
    c.final_summary();
    if let Ok(path) = std::env::var("PERF_RESOURCE_REPORT") {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
        let fields: Vec<_> = stat
            .rsplit_once(')')
            .unwrap()
            .1
            .split_whitespace()
            .collect();
        let ticks = std::process::Command::new("getconf")
            .arg("CLK_TCK")
            .output()
            .unwrap();
        let ticks: f64 = String::from_utf8(ticks.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let user: f64 = fields[11].parse::<f64>().unwrap() / ticks;
        let system: f64 = fields[12].parse::<f64>().unwrap() / ticks;
        let peak_rss: u64 = status
            .lines()
            .find_map(|line| line.strip_prefix("VmHWM:"))
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let resources = serde_json::json!({"wall_seconds": start.elapsed().as_secs_f64(),
            "user_seconds": user, "system_seconds": system, "peak_rss_kib": peak_rss,
            "status": status, "cpu_ticks_per_second": ticks});
        std::fs::write(path, serde_json::to_string_pretty(&resources).unwrap()).unwrap();
    }
}
