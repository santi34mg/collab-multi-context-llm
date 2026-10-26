use agent_collab::context::{self, ContextBudget, SHARED};
use agent_collab::experiments::{self, published_bits};
use agent_collab::model::TinyTransformer;

#[test]
#[ignore = "12,000 SGD steps: run cargo test --release --test learning -- --ignored"]
fn learned_peer_dependence_survives_counterfactual_tests() {
    let mut model = TinyTransformer::new(experiments::small_config(), 7).unwrap();
    let (initial, final_loss) = experiments::train_published_bits(&mut model, 12000).unwrap();
    assert!(final_loss < 0.05, "loss {initial} -> {final_loss}");
    assert_eq!(
        experiments::published_bit_accuracy(&model, true).unwrap(),
        8
    );
    assert!(experiments::published_bit_accuracy(&model, false).unwrap() <= 4);

    let mut counterfactual_correct = 0;
    let mut peer_masked_correct = 0;
    for left in *b"AB" {
        for right in *b"AB" {
            let (engine, targets) = published_bits(left, right).unwrap();
            let original = context::encode(&engine, model.config(), &ContextBudget::default())
                .unwrap()
                .input;
            let probabilities = model.predict(&original).unwrap();
            for (row, &target) in targets.iter().enumerate() {
                assert!(probabilities[row][target] > 0.95);
                let query = original.queries[row];
                let mut flipped = original.clone();
                let mut masked = original.clone();
                for (key, token) in original.tokens.iter().enumerate() {
                    if token.kind == SHARED && token.stream != row && original.visible[query][key] {
                        flipped.tokens[key].token = if token.token == b'A' as usize {
                            b'B' as usize
                        } else {
                            b'A' as usize
                        };
                        masked.visible[query][key] = false;
                    }
                }
                let opposite = if target == b'A' as usize {
                    b'B' as usize
                } else {
                    b'A' as usize
                };
                if argmax(&model.predict(&flipped).unwrap()[row]) == opposite {
                    counterfactual_correct += 1;
                }
                if argmax(&model.predict(&masked).unwrap()[row]) == target {
                    peer_masked_correct += 1;
                }
            }
        }
    }
    assert_eq!(counterfactual_correct, 8);
    assert!(peer_masked_correct <= 4);
}

fn argmax(probabilities: &[f32]) -> usize {
    (0..256)
        .max_by(|&a, &b| probabilities[a].total_cmp(&probabilities[b]))
        .unwrap()
}
