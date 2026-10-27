//! A small, CPU-only, single-layer, single-head transformer decoder.
//!
//! Architecture (row vectors): `x = token + position + stream + kind`,
//! `Q = x Wq`, `K = x Wk`, `V = x Wv`,
//! `a = x_query + softmax(mask(Q_query K^T / sqrt(width))) V Wo + bo`,
//! `f = tanh(a W1 + b1)`, `h = a + f W2 + b2`, `logits = h Wlm + blm`.
//! The FFN has twice the model width. There is no layer normalization, dropout,
//! weight tying, or implicit causal mask: callers supply the entire visibility
//! policy. All embeddings, matrices, and biases receive analytic gradients.
//! Training uses mean next-token cross entropy and plain SGD, without clipping.
//! Only requested output rows are evaluated; duplicate queries are supported.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::ops::Range;
use std::path::Path;

pub const VOCAB_SIZE: usize = 257;
pub const BOS: usize = 256;
const CHECKPOINT_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct InputToken {
    pub token: usize,
    pub position: usize,
    pub stream: usize,
    pub kind: usize,
}

#[derive(Clone, Debug)]
pub struct ModelInput {
    pub tokens: Vec<InputToken>,
    /// `visible[query_token_index][key_token_index]`; true means visible.
    pub visible: Vec<Vec<bool>>,
    /// Token indices at which to produce next-token distributions, in order.
    pub queries: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub width: usize,
    pub max_positions: usize,
    pub max_streams: usize,
    pub context_kinds: usize,
}

#[derive(Clone, Debug)]
pub struct TinyTransformer {
    config: ModelConfig,
    parameters: Vec<f32>,
    layout: Layout,
}

// A single contiguous parameter buffer keeps optimizer updates and checkpoint
// validation uniform. Matrices are row-major, indexed [input, output].
#[derive(Clone, Debug)]
struct Layout {
    token: Range<usize>,
    position: Range<usize>,
    stream: Range<usize>,
    kind: Range<usize>,
    q: Range<usize>,
    k: Range<usize>,
    v: Range<usize>,
    out: Range<usize>,
    out_bias: Range<usize>,
    ff1: Range<usize>,
    ff1_bias: Range<usize>,
    ff2: Range<usize>,
    ff2_bias: Range<usize>,
    lm: Range<usize>,
    lm_bias: Range<usize>,
    ff_width: usize,
    total: usize,
}

impl Layout {
    fn new(config: &ModelConfig) -> Result<Self, String> {
        if config.width == 0
            || config.max_positions == 0
            || config.max_streams == 0
            || config.context_kinds == 0
        {
            return Err("all model configuration dimensions must be positive".into());
        }
        let d = config.width;
        let ff_width = d.checked_mul(2).ok_or("FFN width overflow")?;
        let mut total = 0usize;
        let mut allocate = |rows: usize, cols: usize| -> Result<Range<usize>, String> {
            let count = rows.checked_mul(cols).ok_or("parameter shape overflow")?;
            let start = total;
            total = total.checked_add(count).ok_or("parameter count overflow")?;
            if total > isize::MAX as usize / std::mem::size_of::<f32>() {
                return Err("parameter buffer exceeds addressable memory".into());
            }
            Ok(start..total)
        };
        let token = allocate(VOCAB_SIZE, d)?;
        let position = allocate(config.max_positions, d)?;
        let stream = allocate(config.max_streams, d)?;
        let kind = allocate(config.context_kinds, d)?;
        let q = allocate(d, d)?;
        let k = allocate(d, d)?;
        let v = allocate(d, d)?;
        let out = allocate(d, d)?;
        let out_bias = allocate(1, d)?;
        let ff1 = allocate(d, ff_width)?;
        let ff1_bias = allocate(1, ff_width)?;
        let ff2 = allocate(ff_width, d)?;
        let ff2_bias = allocate(1, d)?;
        let lm = allocate(d, VOCAB_SIZE)?;
        let lm_bias = allocate(1, VOCAB_SIZE)?;
        Ok(Self {
            token,
            position,
            stream,
            kind,
            q,
            k,
            v,
            out,
            out_bias,
            ff1,
            ff1_bias,
            ff2,
            ff2_bias,
            lm,
            lm_bias,
            ff_width,
            total,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    version: u32,
    config: ModelConfig,
    parameters: Vec<f32>,
}

#[derive(Serialize)]
struct CheckpointRef<'a> {
    version: u32,
    config: &'a ModelConfig,
    parameters: &'a [f32],
}

// SplitMix64 has a fully specified wrapping integer sequence; no platform RNG
// or randomized hash state participates in initialization.
struct Rng(u64);

impl Rng {
    fn symmetric(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        let unit = (z >> 40) as f32 / 16_777_216.0;
        (2.0 * unit - 1.0) * scale
    }
}

fn zeros(len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| format!("allocation failed: {e}"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn checked_len(rows: usize, cols: usize) -> Result<usize, String> {
    rows.checked_mul(cols)
        .ok_or_else(|| "activation shape overflow".into())
}

fn finite(values: &[f32], name: &str) -> Result<(), String> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(format!("{name} contains a non-finite value"))
    } else {
        Ok(())
    }
}

/// Row-vector matrix multiplication, with an optional bias.
fn linear(x: &[f32], weights: &[f32], bias: Option<&[f32]>, output: &mut [f32]) {
    if let Some(bias) = bias {
        output.copy_from_slice(bias);
    } else {
        output.fill(0.0);
    }
    let cols = output.len();
    for (i, &value) in x.iter().enumerate() {
        for (out, &weight) in output.iter_mut().zip(&weights[i * cols..(i + 1) * cols]) {
            *out += value * weight;
        }
    }
}

/// Accumulate both input and matrix gradients for a row-vector linear map.
fn linear_backward(x: &[f32], weights: &[f32], dy: &[f32], dx: &mut [f32], dw: &mut [f32]) {
    let cols = dy.len();
    for (i, (&value, input_gradient)) in x.iter().zip(dx.iter_mut()).enumerate() {
        let row = i * cols..(i + 1) * cols;
        // Keep the reduction's addition order, but separate it from independent
        // weight updates so LLVM can vectorize those without reassociating floats.
        for (&weight, &output_gradient) in weights[row.clone()].iter().zip(dy) {
            *input_gradient += weight * output_gradient;
        }
        for (weight_gradient, &output_gradient) in dw[row].iter_mut().zip(dy) {
            *weight_gradient += value * output_gradient;
        }
    }
}

fn add_assign(destination: &mut [f32], source: &[f32]) {
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
}

/// Return (maximum logit, log of the shifted exponential sum). Keeping these
/// separate avoids cancellation in cross entropy even for enormous tied logits
/// or a target whose probability underflows in f32.
fn softmax(logits: &[f32], probabilities: &mut [f32]) -> Result<(f64, f64), String> {
    finite(logits, "softmax logits")?;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("softmax has no finite, visible entries".into());
    }
    for (&logit, probability) in logits.iter().zip(probabilities) {
        *probability = ((logit as f64 - max).exp() / sum) as f32;
    }
    Ok((max, sum.ln()))
}

struct Forward {
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    mixed: Vec<f32>,
    a: Vec<f32>,
    ff: Vec<f32>,
    h: Vec<f32>,
    logits: Vec<f32>,
    probabilities: Vec<Vec<f32>>,
    log_partitions: Vec<(f64, f64)>,
}

impl TinyTransformer {
    pub fn new(config: ModelConfig, seed: u64) -> Result<Self, String> {
        let layout = Layout::new(&config)?;
        let mut parameters = zeros(layout.total)?;
        let mut rng = Rng(seed);
        for range in [
            &layout.token,
            &layout.position,
            &layout.stream,
            &layout.kind,
        ] {
            for parameter in &mut parameters[range.clone()] {
                *parameter = rng.symmetric(0.1);
            }
        }
        let d = config.width;
        for (range, inputs, outputs) in [
            (&layout.q, d, d),
            (&layout.k, d, d),
            (&layout.v, d, d),
            (&layout.out, d, d),
            (&layout.ff1, d, layout.ff_width),
            (&layout.ff2, layout.ff_width, d),
            (&layout.lm, d, VOCAB_SIZE),
        ] {
            let scale = (6.0 / (inputs as f64 + outputs as f64)).sqrt() as f32;
            for parameter in &mut parameters[range.clone()] {
                *parameter = rng.symmetric(scale);
            }
        }
        Ok(Self {
            config,
            parameters,
            layout,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn validate_input(&self, input: &ModelInput) -> Result<(), String> {
        let n = input.tokens.len();
        if n == 0 {
            return Err("input must contain at least one token".into());
        }
        if input.visible.len() != n || input.visible.iter().any(|row| row.len() != n) {
            return Err(format!("visibility mask must be {n} by {n}"));
        }
        for (index, token) in input.tokens.iter().enumerate() {
            if token.token >= VOCAB_SIZE
                || token.position >= self.config.max_positions
                || token.stream >= self.config.max_streams
                || token.kind >= self.config.context_kinds
            {
                return Err(format!(
                    "token {index} has an out-of-range token, position, stream, or kind ID"
                ));
            }
        }
        if input.queries.is_empty() {
            return Err("input must contain at least one query".into());
        }
        for &query in &input.queries {
            if query >= n {
                return Err(format!(
                    "query index {query} is out of range for {n} tokens"
                ));
            }
            if !input.visible[query].iter().any(|&v| v) {
                return Err(format!("query {query} has no visible keys"));
            }
        }
        checked_len(n, self.config.width)?;
        checked_len(input.queries.len(), n)?;
        checked_len(input.queries.len(), self.layout.ff_width)?;
        checked_len(input.queries.len(), VOCAB_SIZE)?;
        Ok(())
    }

    pub fn predict(&self, input: &ModelInput) -> Result<Vec<Vec<f32>>, String> {
        Ok(self.forward(input)?.probabilities)
    }

    fn forward(&self, input: &ModelInput) -> Result<Forward, String> {
        self.validate_input(input)?;
        let d = self.config.width;
        let f = self.layout.ff_width;
        let n = input.tokens.len();
        let m = input.queries.len();
        let p = &self.parameters;
        let l = &self.layout;
        let mut x = zeros(checked_len(n, d)?)?;
        let mut q = zeros(checked_len(m, d)?)?;
        let mut k = zeros(x.len())?;
        let mut v = zeros(x.len())?;
        for (i, token) in input.tokens.iter().enumerate() {
            let row = &mut x[i * d..(i + 1) * d];
            for (range, id) in [
                (&l.token, token.token),
                (&l.position, token.position),
                (&l.stream, token.stream),
                (&l.kind, token.kind),
            ] {
                add_assign(row, &p[range.start + id * d..range.start + (id + 1) * d]);
            }
            linear(row, &p[l.k.clone()], None, &mut k[i * d..(i + 1) * d]);
            linear(row, &p[l.v.clone()], None, &mut v[i * d..(i + 1) * d]);
        }
        finite(&x, "embeddings")?;
        finite(&k, "keys")?;
        finite(&v, "values")?;
        let mut attention = zeros(checked_len(m, n)?)?;
        let mut mixed = zeros(checked_len(m, d)?)?;
        let mut a = zeros(mixed.len())?;
        let mut ff = zeros(checked_len(m, f)?)?;
        let mut h = zeros(mixed.len())?;
        let mut logits = zeros(checked_len(m, VOCAB_SIZE)?)?;
        let mut probabilities = Vec::with_capacity(m);
        let mut log_partitions = Vec::with_capacity(m);
        let scale = 1.0 / (d as f32).sqrt();
        let mut scores = zeros(n)?;
        let mut visible_probabilities = zeros(n)?;
        for (r, &query) in input.queries.iter().enumerate() {
            let qr = &mut q[r * d..(r + 1) * d];
            linear(&x[query * d..(query + 1) * d], &p[l.q.clone()], None, qr);
            finite(qr, "queries")?;
            // Compact the visible scores instead of using a large negative
            // sentinel. Masked entries have exactly zero probability/gradient.
            let mut count = 0;
            for (j, &visible) in input.visible[query].iter().enumerate() {
                if visible {
                    scores[count] = qr
                        .iter()
                        .zip(&k[j * d..(j + 1) * d])
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        * scale;
                    count += 1;
                }
            }
            softmax(&scores[..count], &mut visible_probabilities[..count])?;
            let context = &mut mixed[r * d..(r + 1) * d];
            let mut index = 0;
            for (j, &visible) in input.visible[query].iter().enumerate() {
                if visible {
                    let probability = visible_probabilities[index];
                    attention[r * n + j] = probability;
                    for c in 0..d {
                        context[c] += probability * v[j * d + c];
                    }
                    index += 1;
                }
            }
            let ar = &mut a[r * d..(r + 1) * d];
            linear(context, &p[l.out.clone()], Some(&p[l.out_bias.clone()]), ar);
            add_assign(ar, &x[query * d..(query + 1) * d]);
            let fr = &mut ff[r * f..(r + 1) * f];
            linear(ar, &p[l.ff1.clone()], Some(&p[l.ff1_bias.clone()]), fr);
            finite(fr, "FFN preactivations")?;
            for value in fr.iter_mut() {
                *value = value.tanh();
            }
            let hr = &mut h[r * d..(r + 1) * d];
            linear(fr, &p[l.ff2.clone()], Some(&p[l.ff2_bias.clone()]), hr);
            add_assign(hr, ar);
            let lr = &mut logits[r * VOCAB_SIZE..(r + 1) * VOCAB_SIZE];
            linear(hr, &p[l.lm.clone()], Some(&p[l.lm_bias.clone()]), lr);
            let mut pr = zeros(VOCAB_SIZE)?;
            log_partitions.push(softmax(lr, &mut pr)?);
            probabilities.push(pr);
        }
        // Catch overflow before a later nonlinearity could hide it.
        finite(&mixed, "attention outputs")?;
        finite(&a, "attention residuals")?;
        finite(&h, "FFN residuals")?;
        Ok(Forward {
            x,
            q,
            k,
            v,
            attention,
            mixed,
            a,
            ff,
            h,
            logits,
            probabilities,
            log_partitions,
        })
    }

    fn loss_and_gradient(
        &self,
        input: &ModelInput,
        targets: &[usize],
    ) -> Result<(f32, Vec<f32>), String> {
        if targets.len() != input.queries.len() {
            return Err("target count must equal query count".into());
        }
        if targets.iter().any(|&target| target >= VOCAB_SIZE) {
            return Err("target token ID is out of range".into());
        }
        let cache = self.forward(input)?;
        let d = self.config.width;
        let f = self.layout.ff_width;
        let n = input.tokens.len();
        let m = input.queries.len();
        let p = &self.parameters;
        let l = &self.layout;
        let mut gradient = zeros(p.len())?;
        let mut dx = zeros(cache.x.len())?;
        let mut dk = zeros(cache.k.len())?;
        let mut dv = zeros(cache.v.len())?;
        let mut dq = zeros(d)?;
        let mut dh = zeros(d)?;
        let mut dff = zeros(f)?;
        let mut da = zeros(d)?;
        let mut dc = zeros(d)?;
        let mut dp = zeros(n)?;
        let mut dlogits = zeros(VOCAB_SIZE)?;
        let mean = 1.0 / m as f32;
        let scale = 1.0 / (d as f32).sqrt();
        let mut loss = 0.0f64;
        for (r, (&query, &target)) in input.queries.iter().zip(targets).enumerate() {
            let (maximum, log_sum) = cache.log_partitions[r];
            loss += (maximum - cache.logits[r * VOCAB_SIZE + target] as f64 + log_sum) / m as f64;
            for (grad, &probability) in dlogits.iter_mut().zip(&cache.probabilities[r]) {
                *grad = probability * mean;
            }
            dlogits[target] -= mean;
            dh.fill(0.0);
            linear_backward(
                &cache.h[r * d..(r + 1) * d],
                &p[l.lm.clone()],
                &dlogits,
                &mut dh,
                &mut gradient[l.lm.clone()],
            );
            add_assign(&mut gradient[l.lm_bias.clone()], &dlogits);
            da.copy_from_slice(&dh); // h = a + FFN(a)
            dff.fill(0.0);
            let fr = &cache.ff[r * f..(r + 1) * f];
            linear_backward(
                fr,
                &p[l.ff2.clone()],
                &dh,
                &mut dff,
                &mut gradient[l.ff2.clone()],
            );
            add_assign(&mut gradient[l.ff2_bias.clone()], &dh);
            for (grad, &activation) in dff.iter_mut().zip(fr) {
                *grad *= 1.0 - activation * activation;
            }
            linear_backward(
                &cache.a[r * d..(r + 1) * d],
                &p[l.ff1.clone()],
                &dff,
                &mut da,
                &mut gradient[l.ff1.clone()],
            );
            add_assign(&mut gradient[l.ff1_bias.clone()], &dff);
            add_assign(&mut dx[query * d..(query + 1) * d], &da);
            dc.fill(0.0);
            linear_backward(
                &cache.mixed[r * d..(r + 1) * d],
                &p[l.out.clone()],
                &da,
                &mut dc,
                &mut gradient[l.out.clone()],
            );
            add_assign(&mut gradient[l.out_bias.clone()], &da);
            dp.fill(0.0);
            let attention = &cache.attention[r * n..(r + 1) * n];
            for (j, &visible) in input.visible[query].iter().enumerate() {
                if !visible {
                    continue;
                }
                for c in 0..d {
                    dp[j] += dc[c] * cache.v[j * d + c];
                    dv[j * d + c] += attention[j] * dc[c];
                }
            }
            let average: f32 = attention.iter().zip(&dp).map(|(a, g)| a * g).sum();
            dq.fill(0.0);
            for (j, &visible) in input.visible[query].iter().enumerate() {
                if !visible {
                    continue;
                }
                let ds = attention[j] * (dp[j] - average) * scale;
                for c in 0..d {
                    dq[c] += ds * cache.k[j * d + c];
                    dk[j * d + c] += ds * cache.q[r * d + c];
                }
            }
            linear_backward(
                &cache.x[query * d..(query + 1) * d],
                &p[l.q.clone()],
                &dq,
                &mut dx[query * d..(query + 1) * d],
                &mut gradient[l.q.clone()],
            );
        }
        for (i, token) in input.tokens.iter().enumerate() {
            let xr = &cache.x[i * d..(i + 1) * d];
            let dxr = &mut dx[i * d..(i + 1) * d];
            linear_backward(
                xr,
                &p[l.k.clone()],
                &dk[i * d..(i + 1) * d],
                dxr,
                &mut gradient[l.k.clone()],
            );
            linear_backward(
                xr,
                &p[l.v.clone()],
                &dv[i * d..(i + 1) * d],
                dxr,
                &mut gradient[l.v.clone()],
            );
            for (range, id) in [
                (&l.token, token.token),
                (&l.position, token.position),
                (&l.stream, token.stream),
                (&l.kind, token.kind),
            ] {
                add_assign(
                    &mut gradient[range.start + id * d..range.start + (id + 1) * d],
                    dxr,
                );
            }
        }
        finite(&gradient, "parameter gradients")?;
        let loss = loss as f32;
        if !loss.is_finite() {
            return Err("cross entropy is non-finite".into());
        }
        Ok((loss, gradient))
    }

    /// Perform one plain-SGD step and return mean cross entropy BEFORE updating.
    /// The learning rate must be finite and positive. Validation or numerical
    /// failure leaves all parameters unchanged.
    pub fn train_step(
        &mut self,
        input: &ModelInput,
        targets: &[usize],
        learning_rate: f32,
    ) -> Result<f32, String> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        let (loss, mut gradient) = self.loss_and_gradient(input, targets)?;
        // Reuse the gradient allocation as a candidate parameter buffer, so a
        // numerical failure cannot leave a partially updated model behind.
        for (candidate, &parameter) in gradient.iter_mut().zip(&self.parameters) {
            *candidate = parameter - learning_rate * *candidate;
        }
        finite(&gradient, "updated parameters")?;
        self.parameters = gradient;
        Ok(loss)
    }

    /// Save a versioned JSON checkpoint of the complete inference/training state.
    /// SGD has no optimizer state; initialization RNG state is not needed again.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        finite(&self.parameters, "checkpoint parameters")?;
        let path = path.as_ref();
        let file =
            File::create(path).map_err(|e| format!("create checkpoint {}: {e}", path.display()))?;
        let mut writer = BufWriter::new(file);
        let checkpoint = CheckpointRef {
            version: CHECKPOINT_VERSION,
            config: &self.config,
            parameters: &self.parameters,
        };
        serde_json::to_writer(&mut writer, &checkpoint)
            .map_err(|e| format!("serialize checkpoint {}: {e}", path.display()))?;
        writer
            .flush()
            .map_err(|e| format!("flush checkpoint {}: {e}", path.display()))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file =
            File::open(path).map_err(|e| format!("open checkpoint {}: {e}", path.display()))?;
        let checkpoint: Checkpoint = serde_json::from_reader(BufReader::new(file))
            .map_err(|e| format!("parse checkpoint {}: {e}", path.display()))?;
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(format!(
                "unsupported checkpoint version {}",
                checkpoint.version
            ));
        }
        let layout = Layout::new(&checkpoint.config)?;
        if checkpoint.parameters.len() != layout.total {
            return Err(format!(
                "checkpoint has {} parameters; expected {}",
                checkpoint.parameters.len(),
                layout.total,
            ));
        }
        finite(&checkpoint.parameters, "checkpoint parameters")?;
        Ok(Self {
            config: checkpoint.config,
            parameters: checkpoint.parameters,
            layout,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn backward_kernel_preserves_scalar_accumulation_bits() {
        for rows in [1, 3, 16] {
            for cols in [1, 3, 16, 32, VOCAB_SIZE] {
                let mut rng = Rng(42);
                let mut values = |n| (0..n).map(|_| rng.symmetric(1.0)).collect::<Vec<_>>();
                let x = values(rows);
                let weights = values(rows * cols);
                let dy = values(cols);
                let mut dx = values(rows);
                let mut dw = values(rows * cols);
                let (mut expected_dx, mut expected_dw) = (dx.clone(), dw.clone());
                for i in 0..rows {
                    for j in 0..cols {
                        expected_dx[i] += weights[i * cols + j] * dy[j];
                        expected_dw[i * cols + j] += x[i] * dy[j];
                    }
                }
                linear_backward(&x, &weights, &dy, &mut dx, &mut dw);
                assert!(
                    dx.iter()
                        .zip(expected_dx)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert!(
                    dw.iter()
                        .zip(expected_dw)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }

    fn config() -> ModelConfig {
        ModelConfig {
            width: 4,
            max_positions: 32,
            max_streams: 3,
            context_kinds: 3,
        }
    }

    fn input() -> ModelInput {
        ModelInput {
            tokens: vec![
                InputToken {
                    token: BOS,
                    position: 0,
                    stream: 0,
                    kind: 0,
                },
                InputToken {
                    token: 65,
                    position: 1,
                    stream: 1,
                    kind: 1,
                },
                InputToken {
                    token: 66,
                    position: 2,
                    stream: 2,
                    kind: 2,
                },
            ],
            visible: vec![
                vec![true, false, false],
                vec![true, true, false],
                vec![true, true, true],
            ],
            queries: vec![1, 2, 1],
        }
    }

    fn loss(model: &TinyTransformer, input: &ModelInput, targets: &[usize]) -> f64 {
        let forward = model.forward(input).unwrap();
        targets
            .iter()
            .enumerate()
            .map(|(r, &target)| {
                let (maximum, log_sum) = forward.log_partitions[r];
                (maximum - forward.logits[r * VOCAB_SIZE + target] as f64 + log_sum)
                    / targets.len() as f64
            })
            .sum()
    }

    #[test]
    fn deterministic_normalized_predictions_and_duplicate_queries() {
        let a = TinyTransformer::new(config(), 7).unwrap();
        let b = TinyTransformer::new(config(), 7).unwrap();
        let c = TinyTransformer::new(config(), 8).unwrap();
        assert_eq!(a.parameters, b.parameters);
        assert_ne!(a.parameters, c.parameters);
        let probabilities = a.predict(&input()).unwrap();
        assert_eq!(probabilities.len(), 3);
        assert_eq!(probabilities[0], probabilities[2]);
        for row in probabilities {
            assert_eq!(row.len(), VOCAB_SIZE);
            assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn analytic_gradients_match_finite_differences_for_every_parameter_group() {
        let mut model = TinyTransformer::new(config(), 19).unwrap();
        // More pronounced, but nonsaturating activations make the small Q/K
        // derivatives measurable with f32 forward operations.
        for range in [
            &model.layout.token,
            &model.layout.position,
            &model.layout.stream,
            &model.layout.kind,
        ] {
            for value in &mut model.parameters[range.clone()] {
                *value *= 3.0;
            }
        }
        let input = input();
        let targets = [67, 68, 69];
        let (_, gradient) = model.loss_and_gradient(&input, &targets).unwrap();
        let l = &model.layout;
        let groups = [
            l.token.clone(),
            l.position.clone(),
            l.stream.clone(),
            l.kind.clone(),
            l.q.clone(),
            l.k.clone(),
            l.v.clone(),
            l.out.clone(),
            l.out_bias.clone(),
            l.ff1.clone(),
            l.ff1_bias.clone(),
            l.ff2.clone(),
            l.ff2_bias.clone(),
            l.lm.clone(),
            l.lm_bias.clone(),
        ];
        let epsilon = 0.002f32;
        for group in groups {
            // Select several nonzero entries per group, including embeddings
            // shared across query/key/value paths and duplicate query outputs.
            let mut indices: Vec<usize> = group.collect();
            indices.sort_by(|&a, &b| gradient[b].abs().total_cmp(&gradient[a].abs()));
            assert!(gradient[indices[0]].abs() > 1e-8);
            for &index in indices.iter().take(3) {
                let original = model.parameters[index];
                model.parameters[index] = original + epsilon;
                let plus = loss(&model, &input, &targets);
                model.parameters[index] = original - epsilon;
                let minus = loss(&model, &input, &targets);
                model.parameters[index] = original;
                let numerical = ((plus - minus) / (2.0 * epsilon as f64)) as f32;
                let analytic = gradient[index];
                let tolerance = 2e-5 + 0.015 * analytic.abs().max(numerical.abs());
                assert!(
                    (analytic - numerical).abs() < tolerance,
                    "parameter {index}: analytic {analytic}, finite difference {numerical}"
                );
            }
        }
    }

    #[test]
    fn masked_tokens_cannot_affect_predictions_or_gradients() {
        let model = TinyTransformer::new(config(), 31).unwrap();
        let mut original = input();
        original.queries = vec![1];
        // Empty rows are legal for tokens that are not queried.
        original.visible[2].fill(false);
        let mut modified = original.clone();
        modified.tokens[2] = InputToken {
            token: 200,
            position: 9,
            stream: 1,
            kind: 0,
        };
        assert_eq!(
            model.predict(&original).unwrap(),
            model.predict(&modified).unwrap()
        );
        let (_, ga) = model.loss_and_gradient(&original, &[70]).unwrap();
        let (_, gb) = model.loss_and_gradient(&modified, &[70]).unwrap();
        assert_eq!(ga, gb);
        let unused = model.layout.token.start + 66 * model.config.width;
        assert!(
            ga[unused..unused + model.config.width]
                .iter()
                .all(|&g| g == 0.0)
        );
    }

    #[test]
    fn training_reduces_loss_and_updates_all_parameter_groups() {
        let mut model = TinyTransformer::new(config(), 41).unwrap();
        let input = input();
        let targets = [70, 71, 70];
        let initial_parameters = model.parameters.clone();
        let initial = loss(&model, &input, &targets);
        for _ in 0..120 {
            assert!(
                model
                    .train_step(&input, &targets, 0.08)
                    .unwrap()
                    .is_finite()
            );
        }
        let final_loss = loss(&model, &input, &targets);
        assert!(
            final_loss < initial * 0.3,
            "initial {initial}, final {final_loss}"
        );
        let l = &model.layout;
        for range in [
            &l.token,
            &l.position,
            &l.stream,
            &l.kind,
            &l.q,
            &l.k,
            &l.v,
            &l.out,
            &l.out_bias,
            &l.ff1,
            &l.ff1_bias,
            &l.ff2,
            &l.ff2_bias,
            &l.lm,
            &l.lm_bias,
        ] {
            assert_ne!(
                model.parameters[range.clone()],
                initial_parameters[range.clone()]
            );
        }
    }

    #[test]
    fn rejects_invalid_inputs_configs_and_training_arguments_without_mutation() {
        for field in 0..4 {
            let mut c = config();
            match field {
                0 => c.width = 0,
                1 => c.max_positions = 0,
                2 => c.max_streams = 0,
                _ => c.context_kinds = 0,
            }
            assert!(TinyTransformer::new(c, 0).is_err());
        }
        let mut overflow = config();
        overflow.width = usize::MAX;
        assert!(TinyTransformer::new(overflow, 0).is_err());
        let mut model = TinyTransformer::new(config(), 0).unwrap();
        let original = model.parameters.clone();
        let mut invalid_inputs = Vec::new();
        let mut bad = input();
        bad.tokens.clear();
        invalid_inputs.push(bad);
        let mut bad = input();
        bad.visible.pop();
        invalid_inputs.push(bad);
        let mut bad = input();
        bad.visible[0].pop();
        invalid_inputs.push(bad);
        let mut bad = input();
        bad.queries.clear();
        invalid_inputs.push(bad);
        let mut bad = input();
        bad.queries[0] = 3;
        invalid_inputs.push(bad);
        let mut bad = input();
        bad.visible[1].fill(false);
        invalid_inputs.push(bad);
        for field in 0..4 {
            let mut bad = input();
            match field {
                0 => bad.tokens[0].token = VOCAB_SIZE,
                1 => bad.tokens[0].position = model.config.max_positions,
                2 => bad.tokens[0].stream = model.config.max_streams,
                _ => bad.tokens[0].kind = model.config.context_kinds,
            }
            invalid_inputs.push(bad);
        }
        for bad in invalid_inputs {
            assert!(model.predict(&bad).is_err());
            assert!(model.train_step(&bad, &[0, 0, 0], 0.1).is_err());
        }
        for rate in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(model.train_step(&input(), &[0, 0, 0], rate).is_err());
        }
        assert!(model.train_step(&input(), &[0], 0.1).is_err());
        assert!(
            model
                .train_step(&input(), &[0, VOCAB_SIZE, 0], 0.1)
                .is_err()
        );
        assert_eq!(model.parameters, original);
    }

    #[test]
    fn softmax_and_cross_entropy_are_stable_for_extreme_logits() {
        let mut probabilities = [0.0; 3];
        let (maximum, log_sum) =
            softmax(&[10_000.0, 10_001.0, -10_000.0], &mut probabilities).unwrap();
        assert!((probabilities[0] - 0.26894143).abs() < 1e-6);
        assert!((probabilities[1] - 0.7310586).abs() < 1e-6);
        assert_eq!(probabilities[2], 0.0);
        assert!((maximum + log_sum - 10_001.31326168752).abs() < 1e-8);
        assert!(softmax(&[f32::NAN], &mut [0.0]).is_err());
        let mut model = TinyTransformer::new(config(), 0).unwrap();
        model.parameters[model.layout.lm_bias.start] = 1000.0;
        let (loss, _) = model.loss_and_gradient(&input(), &[1, 1, 1]).unwrap();
        assert!(loss.is_finite() && loss > 900.0);
        model.parameters[model.layout.lm_bias.clone()].fill(1e30);
        let (loss, _) = model.loss_and_gradient(&input(), &[1, 1, 1]).unwrap();
        assert!((loss - (VOCAB_SIZE as f32).ln()).abs() < 1e-6);
    }

    #[test]
    fn overflowing_update_leaves_parameters_unchanged() {
        let mut model = TinyTransformer::new(config(), 0).unwrap();
        model.parameters.fill(0.0);
        model.parameters[model.layout.token.start + BOS * model.config.width] = 10.0;
        let input = ModelInput {
            tokens: vec![InputToken {
                token: BOS,
                position: 0,
                stream: 0,
                kind: 0,
            }],
            visible: vec![vec![true]],
            queries: vec![0],
        };
        let original = model.parameters.clone();
        assert!(model.predict(&input).is_ok());
        assert!(model.train_step(&input, &[0], f32::MAX).is_err());
        assert_eq!(model.parameters, original);
    }

    fn checkpoint_path() -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "agent-collab-model-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ))
    }

    #[test]
    fn checkpoint_roundtrip_and_training_continuation() {
        let path = checkpoint_path();
        let mut model = TinyTransformer::new(config(), 53).unwrap();
        model.train_step(&input(), &[70, 71, 70], 0.1).unwrap();
        model.save(&path).unwrap();
        let mut restored = TinyTransformer::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(model.parameters, restored.parameters);
        assert_eq!(model.config().width, restored.config().width);
        assert_eq!(
            model.predict(&input()).unwrap(),
            restored.predict(&input()).unwrap()
        );
        let a = model.train_step(&input(), &[70, 71, 70], 0.1).unwrap();
        let b = restored.train_step(&input(), &[70, 71, 70], 0.1).unwrap();
        assert_eq!(a, b);
        assert_eq!(model.parameters, restored.parameters);
    }

    #[test]
    fn rejects_malformed_or_incompatible_checkpoints_and_nonfinite_computation() {
        let path = checkpoint_path();
        let model = TinyTransformer::new(config(), 0).unwrap();
        for (version, config, parameters) in [
            (CHECKPOINT_VERSION + 1, config(), model.parameters.clone()),
            (CHECKPOINT_VERSION, config(), vec![0.0]),
            (
                CHECKPOINT_VERSION,
                ModelConfig {
                    width: 0,
                    ..config()
                },
                vec![],
            ),
        ] {
            let checkpoint = Checkpoint {
                version,
                config,
                parameters,
            };
            std::fs::write(&path, serde_json::to_vec(&checkpoint).unwrap()).unwrap();
            assert!(TinyTransformer::load(&path).is_err());
        }
        std::fs::write(&path, b"not JSON").unwrap();
        assert!(TinyTransformer::load(&path).is_err());
        std::fs::write(
            &path,
            b"{\"version\":1,\"config\":{},\"parameters\":[1e999]}",
        )
        .unwrap();
        assert!(TinyTransformer::load(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        let mut model = model;
        model.parameters[model.layout.token.start + BOS * model.config.width] = f32::MAX;
        model.parameters[model.layout.position.start] = f32::MAX;
        assert!(model.predict(&input()).is_err());
        assert!(model.train_step(&input(), &[1, 1, 1], 0.1).is_err());
        model.parameters[0] = f32::NAN;
        assert!(model.save(&path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn width_16_twenty_token_training_smoke() {
        let mut model = TinyTransformer::new(
            ModelConfig {
                width: 16,
                ..config()
            },
            61,
        )
        .unwrap();
        let tokens = (0..20)
            .map(|i| InputToken {
                token: if i == 0 { BOS } else { 65 + i % 3 },
                position: i,
                stream: i % 3,
                kind: i % 3,
            })
            .collect();
        let input = ModelInput {
            tokens,
            visible: (0..20).map(|i| (0..20).map(|j| j <= i).collect()).collect(),
            queries: (0..20).collect(),
        };
        let targets: Vec<_> = (0..20).map(|i| 65 + (i + 1) % 3).collect();
        let initial = loss(&model, &input, &targets);
        for _ in 0..200 {
            model.train_step(&input, &targets, 0.05).unwrap();
        }
        assert!(loss(&model, &input, &targets) < initial * 0.3);
    }
}
