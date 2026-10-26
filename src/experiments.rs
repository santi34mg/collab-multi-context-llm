//! A small exhaustive smoke experiment, not an architecture benchmark.
//!
//! Both streams have a private A/B bit. After scripted publication, each must
//! predict the OTHER stream's bit in the same forward pass. Training all four
//! pairs tests the plumbing; it is not a held-out generalization evaluation.

use crate::context::{self, CONTEXT_KINDS, ContextBudget};
use crate::memory::MemoryStore;
use crate::model::{ModelConfig, TinyTransformer};
use crate::state::{Engine, EngineConfig, Operation};

pub fn small_config() -> ModelConfig {
    ModelConfig {
        width: 16,
        max_positions: 512,
        max_streams: 3,
        context_kinds: CONTEXT_KINDS,
    }
}

pub fn published_bits(left: u8, right: u8) -> Result<(Engine, Vec<usize>), String> {
    if !b"AB".contains(&left) || !b"AB".contains(&right) {
        return Err("published-bit values must be A or B".into());
    }
    let mut engine = Engine::new(EngineConfig::default(), MemoryStore::new(1024))?;
    engine.seed_entry(0, (left as char).to_string())?;
    engine.seed_entry(1, (right as char).to_string())?;
    engine.step_actions(vec![
        (
            0,
            Operation::Publish {
                target: Some(1),
                content: (left as char).to_string(),
            },
        ),
        (
            1,
            Operation::Publish {
                target: Some(0),
                content: (right as char).to_string(),
            },
        ),
    ])?;
    Ok((engine, vec![right as usize, left as usize]))
}

pub fn train_published_bits(
    model: &mut TinyTransformer,
    steps: usize,
) -> Result<(f32, f32), String> {
    if steps == 0 {
        return Err("training steps must be positive".into());
    }
    let budget = ContextBudget::default();
    let mut cases = Vec::new();
    for left in *b"AB" {
        for right in *b"AB" {
            let (engine, targets) = published_bits(left, right)?;
            cases.push((
                context::encode(&engine, model.config(), &budget)?.input,
                targets,
            ));
        }
    }
    let loss = |model: &TinyTransformer| -> Result<f32, String> {
        let mut total = 0.0;
        for (input, targets) in &cases {
            for (probabilities, &target) in model.predict(input)?.iter().zip(targets) {
                total -= probabilities[target].max(f32::MIN_POSITIVE).ln();
            }
        }
        Ok(total / 8.0)
    };
    let initial = loss(model)?;
    for step in 0..steps {
        let (input, targets) = &cases[step % cases.len()];
        model.train_step(input, targets, 0.03)?;
    }
    Ok((initial, loss(model)?))
}

/// Number of correct stream predictions out of eight on the training truth table.
pub fn published_bit_accuracy(
    model: &TinyTransformer,
    read_publications: bool,
) -> Result<usize, String> {
    let budget = ContextBudget {
        read_publications,
        ..ContextBudget::default()
    };
    let mut correct = 0;
    for left in *b"AB" {
        for right in *b"AB" {
            let (engine, targets) = published_bits(left, right)?;
            let predictions = context::predict_round(&engine, model, &budget)?;
            correct += predictions
                .iter()
                .zip(targets)
                .filter(|((_, byte), target)| *byte as usize == *target)
                .count();
        }
    }
    Ok(correct)
}
