//! Collaborative multi-context language model research prototype.
//!
//! A tiny trainable transformer is separate from the deterministic engine that
//! executes its byte-framed operations and manages versioned persistent state.

pub mod commands;
pub mod context;
pub mod distillation;
pub mod experiments;
pub mod inference;
pub mod memory;
pub mod model;
pub mod state;
pub mod training;
