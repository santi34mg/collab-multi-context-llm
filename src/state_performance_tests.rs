use super::*;

// Frozen from 5bfef5f1fa05101728437b619fc17470016b4755: only receiver/call syntax changed.
fn legacy_step_actions(
    engine: &mut Engine,
    mut actions: Vec<(StreamId, Operation)>,
) -> Result<RoundReport, String> {
    engine.validate_actors(actions.iter().map(|(id, _)| *id))?;
    actions.sort_by_key(|(id, _)| *id);
    let snapshot = engine.clone();
    let mut candidate = engine.clone();
    let next_round = candidate.round.checked_add(1).ok_or("round overflow")?;
    let mut outcomes = Vec::new();
    for (stream, operation) in actions {
        let result = snapshot
            .validate_operation(stream, &operation)
            .and_then(|()| candidate.apply_operation(stream, &operation));
        let (accepted, detail) = match result {
            Ok(detail) => (true, detail),
            Err(e) => (false, e),
        };
        let outcome = ActionOutcome {
            stream,
            operation: Some(operation),
            accepted,
            detail,
        };
        candidate.record_outcome(snapshot.round, &outcome);
        outcomes.push(outcome);
    }
    candidate.round = next_round;
    *engine = candidate;
    Ok(RoundReport {
        round: snapshot.round,
        outcomes,
    })
}

fn legacy_step_tokens(
    engine: &mut Engine,
    mut tokens: Vec<(StreamId, u8)>,
) -> Result<RoundReport, String> {
    engine.validate_actors(tokens.iter().map(|(id, _)| *id))?;
    tokens.sort_by_key(|(id, _)| *id);
    let mut candidate = engine.clone();
    let round = engine.round;
    let mut actions = Vec::new();
    let mut malformed = Vec::new();
    for (id, byte) in tokens {
        let stream = candidate.streams.get_mut(&id).expect("validated actor");
        if stream.history.len() >= engine.config.max_history_bytes {
            return Err(format!(
                "stream {id} exhausted its generation-history budget"
            ));
        }
        stream.history.push(byte);
        candidate.trace.push(TraceEvent::Token {
            round,
            stream: id,
            byte,
        });
        if stream.discarding_frame {
            if byte == b'\n' {
                stream.discarding_frame = false;
            }
            continue;
        }
        if byte != b'\n' {
            if stream.frame.len() >= engine.config.max_frame_bytes {
                stream.frame.clear();
                stream.discarding_frame = true;
                malformed.push((id, "operation frame exceeded its byte budget".to_string()));
            } else {
                stream.frame.push(byte);
            }
            continue;
        }
        let frame = std::mem::take(&mut stream.frame);
        if frame.is_empty() {
            continue;
        }
        match serde_json::from_slice::<Operation>(&frame) {
            Ok(operation) => actions.push((id, operation)),
            Err(e) => malformed.push((id, format!("invalid operation frame: {e}"))),
        }
    }
    // Appended bytes are not editable state and do not affect operation
    // authorization. step_actions still validates all operations together.
    let mut report = legacy_step_actions(&mut candidate, actions)?;
    for (stream, detail) in malformed {
        let outcome = ActionOutcome {
            stream,
            operation: None,
            accepted: false,
            detail,
        };
        candidate.record_outcome(round, &outcome);
        report.outcomes.push(outcome);
    }
    report.outcomes.sort_by_key(|outcome| outcome.stream);
    *engine = candidate;
    Ok(report)
}

type Pair = (Engine, Engine);
fn pair(config: EngineConfig) -> Pair {
    let engine = Engine::new(config, MemoryStore::new(8192)).unwrap();
    (engine.clone(), engine)
}
fn compare<T: Clone>(
    engines: &mut Pair,
    input: T,
    optimized: fn(&mut Engine, T) -> Result<RoundReport, String>,
    legacy: fn(&mut Engine, T) -> Result<RoundReport, String>,
) -> Result<RoundReport, String> {
    let before = format!("{:?}", engines.0);
    let actual = optimized(&mut engines.0, input.clone());
    let expected = legacy(&mut engines.1, input);
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    assert_eq!(format!("{:?}", engines.0), format!("{:?}", engines.1));
    if actual.is_err() {
        assert_eq!(format!("{:?}", engines.0), before);
    }
    actual
}
fn actions(e: &mut Pair, input: Vec<(StreamId, Operation)>) -> Result<RoundReport, String> {
    compare(e, input, Engine::step_actions, legacy_step_actions)
}
fn tokens(e: &mut Pair, input: Vec<(StreamId, u8)>) -> Result<RoundReport, String> {
    compare(e, input, Engine::step_tokens, legacy_step_tokens)
}
fn publish(content: &str) -> Operation {
    Operation::Publish {
        target: None,
        content: content.into(),
    }
}
// Right-align frames so their terminating newlines share a snapshot round.
fn frames(e: &mut Pair, frames: Vec<(StreamId, Vec<u8>)>) {
    let width = frames.iter().map(|(_, b)| b.len()).max().unwrap_or(0);
    for offset in 0..width {
        let input = frames
            .iter()
            .rev()
            .filter_map(|(id, bytes)| {
                offset
                    .checked_sub(width - bytes.len())
                    .map(|i| (*id, bytes[i]))
            })
            .collect();
        tokens(e, input).unwrap();
    }
}

fn replay(e: &mut Pair, byte_mode: bool, round: Vec<(StreamId, Operation)>) {
    if byte_mode {
        frames(
            e,
            round.into_iter().map(|(id, op)| (id, op.frame())).collect(),
        );
    } else {
        actions(e, round).unwrap();
    }
}

#[test]
fn differential_deterministic_json_and_arbitrary_bytes() {
    let mut e = pair(EngineConfig::default());
    let mut rng = 0x91e1_0da5u32;
    for i in 0..32 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        let bytes = match i % 5 {
            0 => publish(&format!("valid {rng}: λ")).frame(),
            1 => b"{\"op\":\"publish\",\"target\":null,\"content\":\" \"}\n".to_vec(),
            2 => b"{\"op\":\"finish\",\"result\":\"x\",\"scope\":\"bad\"}\n".to_vec(),
            3 => vec![0xff, 0xc0, b'\n'],
            _ => {
                let mut bytes = Vec::new();
                for _ in 0..(i + 1) {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    bytes.push(rng as u8);
                }
                bytes.push(b'\n');
                bytes
            }
        };
        frames(&mut e, vec![(0, bytes), (1, publish("peer").frame())]);
    }
    assert_eq!(e.0.streams[&0].notices.len(), 8);
}

#[test]
fn differential_frame_limit_discard_and_recovery() {
    let good = publish("ok").frame();
    let limit = good.len() - 1;
    let mut e = pair(EngineConfig {
        max_frame_bytes: limit,
        ..EngineConfig::default()
    });
    frames(&mut e, vec![(0, good.clone())]); // Exactly the limit, excluding newline.
    let oversized = [vec![b'x'; limit + 1], good.clone()].concat();
    frames(&mut e, vec![(0, oversized), (1, vec![b'\n'])]);
    assert_eq!(e.0.workspace.len(), 1, "discarded suffix must not execute");
    assert!(!e.0.streams[&0].discarding_frame);
    frames(&mut e, vec![(0, good), (1, vec![b'\n'])]);
    assert_eq!(e.0.workspace.len(), 2);
}

#[test]
fn differential_atomic_history_round_and_actor_errors() {
    let mut e = pair(EngineConfig {
        max_history_bytes: 1,
        ..EngineConfig::default()
    });
    tokens(&mut e, vec![(1, b'\n')]).unwrap();
    for engine in [&mut e.0, &mut e.1] {
        engine.streams.get_mut(&0).unwrap().frame = publish("must roll back").frame();
        engine.streams.get_mut(&0).unwrap().frame.pop();
    }
    assert!(tokens(&mut e, vec![(1, b'x'), (0, b'\n')]).is_err());
    tokens(&mut e, vec![(0, b'\n')]).unwrap();
    assert!(tokens(&mut e, vec![(0, b'x')]).is_err());
    actions(&mut e, vec![(0, publish("history does not limit actions"))]).unwrap();
    for engine in [&mut e.0, &mut e.1] {
        engine.round = u64::MAX;
    }
    assert!(tokens(&mut e, vec![(1, b'x')]).is_err());
    assert_eq!(tokens(&mut e, vec![]).unwrap_err(), "round overflow");
    assert!(actions(&mut e, vec![(0, publish("overflow"))]).is_err());
    let mut e = pair(EngineConfig::default());
    for engine in [&mut e.0, &mut e.1] {
        engine.round = u64::MAX;
    }
    assert!(tokens(&mut e, vec![(0, b'\n'), (1, 0xff)]).is_err());
    let mut e = pair(EngineConfig::default());
    for ids in [[0, 0], [1, 99], [99, 0]] {
        assert!(actions(&mut e, ids.map(|id| (id, publish("x"))).to_vec()).is_err());
        assert!(tokens(&mut e, ids.map(|id| (id, b'\n')).to_vec()).is_err());
    }
    let finish = Operation::Finish {
        result: "done".into(),
    };
    actions(&mut e, vec![(0, finish.clone())]).unwrap();
    assert!(actions(&mut e, vec![(1, publish("x")), (0, publish("x"))]).is_err());
    assert!(tokens(&mut e, vec![(1, b'\n'), (0, b'\n')]).is_err());
    actions(&mut e, vec![(1, finish)]).unwrap();
    assert!(actions(&mut e, vec![]).is_err());
    assert!(tokens(&mut e, vec![]).is_err());
}

#[test]
fn differential_snapshot_edits_and_ordered_application() {
    for byte_mode in [false, true] {
        let mut e = pair(EngineConfig {
            max_workspace_bytes: 5,
            ..EngineConfig::default()
        });
        for engine in [&mut e.0, &mut e.1] {
            engine.seed_entry(0, "old").unwrap();
        }
        let proposal = |version| Operation::ProposeEdit {
            target: 0,
            entry_id: 1,
            expected_version: version,
            replacement: "new".into(),
        };
        let accept = |id| Operation::AcceptEdit { proposal_id: id };
        let reject = |id| Operation::RejectEdit { proposal_id: id };
        let finish = Operation::Finish {
            result: "done".into(),
        };
        let rounds = vec![
            vec![(1, publish("bbb")), (0, publish("aaa"))],
            vec![(1, proposal(1)), (0, accept(1))], // Cannot accept a new proposal.
            vec![(1, proposal(1)), (0, accept(1))], // Proposal sees the old version.
            vec![(0, accept(2))],                   // Stale at application.
            vec![(0, reject(2))],
            vec![(0, reject(2))],                // Already decided.
            vec![(0, finish), (1, proposal(2))], // Target is unfinished in snapshot.
        ];
        for round in rounds {
            replay(&mut e, byte_mode, round);
        }
        assert_eq!(e.0.workspace.len(), 1);
        assert_eq!(e.0.proposals[&3].status, ProposalStatus::Pending);
    }
}

#[test]
fn differential_complete_collaboration_example_actions_and_frames() {
    #[derive(Deserialize)]
    struct Script {
        scope: String,
        working: Vec<Vec<String>>,
        rounds: Vec<Vec<Output>>,
    }
    #[derive(Deserialize)]
    struct Output {
        stream: StreamId,
        operation: Operation,
    }
    for byte_mode in [false, true] {
        let scripts: Vec<Script> =
            serde_json::from_str(include_str!("../examples/collaboration-dataset.json")).unwrap();
        for script in scripts {
            let config = EngineConfig {
                scope: script.scope,
                streams: script.working.len(),
                ..EngineConfig::default()
            };
            let mut e = pair(config);
            for (id, entries) in script.working.iter().enumerate() {
                for entry in entries {
                    for engine in [&mut e.0, &mut e.1] {
                        engine.seed_entry(id, entry.clone()).unwrap();
                    }
                }
            }
            for round in script.rounds {
                let round = round.into_iter().map(|o| (o.stream, o.operation)).collect();
                replay(&mut e, byte_mode, round);
            }
            assert!(e.0.is_finished());
            assert_eq!(
                format!("{:?}", e.0.commit_memory()),
                format!("{:?}", e.1.commit_memory())
            );
            assert_eq!(format!("{:?}", e.0), format!("{:?}", e.1));
            assert_eq!(e.0.memory.active("arithmetic").len(), 1);
        }
    }
}
