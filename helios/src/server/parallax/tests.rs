use super::{Record, evaluation::*, observer::Observer, prediction::*};
use crate::parallax::{LayerDemand, RoutingSample, now_ns};
use serde_json::json;
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

pub(super) fn profile(name: &str) -> Profile {
    let path = std::env::temp_dir().join(format!(
        "flare-native-parallax-{}-{name}-{}",
        std::process::id(),
        now_ns()
    ));
    Profile {
        settings: Settings {
            version: 1,
            catalog: "unused".into(),
            hardware: "unused".into(),
            output: path,
            predictors: vec![
                Selection {
                    name: "recent".into(),
                    config: json!({}),
                },
                Selection {
                    name: "ewma".into(),
                    config: json!({"alpha":0.5}),
                },
            ],
            target_offset: 1,
            horizon_ns: 10000,
            budget_ns: 100_000_000,
            max_frames: 64,
            evaluation: None,
            active: None,
        },
        model: Catalog {
            identity: "synthetic;package=test".into(),
            architecture: "fixture".into(),
            experts: vec![
                Expert {
                    layer: 0,
                    expert: 0,
                },
                Expert {
                    layer: 0,
                    expert: 1,
                },
            ],
            expert_bytes: vec![100, 100],
        },
        hardware: Hardware {
            identity: "SYNTHETIC native test".into(),
            hbm_bytes: 100,
            backing_bytes: 200,
            bandwidth_bytes_per_ns: 1.,
            transfer_latency_ns: 10,
            synchronization_ns: 5,
            cold_cost_ns: 1000,
            transfer_calibration: None,
        },
    }
}
#[test]
fn registration_and_configuration_reject_unknown_and_duplicate_providers() {
    let mut registry = Registry::builtins();
    assert!(registry.register("recent", |_| panic!()).is_err());
    assert!(
        registry
            .create(&Selection {
                name: "not-compiled".into(),
                config: json!({})
            })
            .is_err()
    );
    assert!(
        registry
            .create(&Selection {
                name: "ewma".into(),
                config: json!({"alpha":2})
            })
            .is_err()
    );
    assert!(
        registry
            .create(&Selection {
                name: "ewma".into(),
                config: json!({"alhpa":0.5})
            })
            .is_err()
    );
    let mut p = profile("config");
    p.settings.predictors.push(p.settings.predictors[0].clone());
    assert!(p.validate().is_err());
}
pub(super) fn fixture() -> Capture {
    let p = profile("evaluate");
    let expert = p.model.experts[1];
    let observations = (0..5)
        .map(|i| Observation {
            seq: i,
            demand_ns: i * 10000,
            available_ns: i * 10000 + 1,
            phase: "decode".into(),
            batch_size: 1,
            concurrency: 1,
            demands: vec![(expert, 2)],
        })
        .collect::<Vec<_>>();
    let forecasts = observations
        .iter()
        .map(|o| Forecast {
            provider: "recent".into(),
            provider_version: "1".into(),
            cutoff: o.seq,
            target: o.seq + 1,
            available_ns: o.available_ns,
            started_ns: o.available_ns,
            completed_ns: o.available_ns + 10,
            compute_ns: 10,
            expected_demand_ns: o.available_ns + 10000,
            expires_ns: o.available_ns + 20000,
            scores: vec![(expert, 2.)],
        })
        .collect::<Vec<_>>();
    Capture {
        native_capture_version: 1,
        profile: p,
        observations,
        forecasts: [("recent".into(), forecasts)].into(),
        workers: [(
            "recent".into(),
            WorkerCounts {
                submitted: 5,
                completed: 5,
                ..WorkerCounts::default()
            },
        )]
        .into(),
        error: None,
        limit_reason: None,
    }
}
#[test]
fn policy_states_are_independent_and_late_forecasts_cannot_change_past_hits() {
    let mut c = fixture();
    let r = evaluate(&c);
    assert_eq!(r["contenders"]["recent/static"]["hits"], 0);
    assert_eq!(r["contenders"]["recent/economic"]["hits"], 8);
    assert_eq!(
        r["contenders"]["recent/economic"]["forecast_precision"],
        1.0
    );
    let original = c.forecasts["recent"].clone();
    c.forecasts.get_mut("recent").unwrap()[0].completed_ns = 15000;
    assert!(
        evaluate(&c)["contenders"]["recent/economic"]["hits"]
            .as_u64()
            .unwrap()
            < 8
    );
    c.forecasts.insert("recent".into(), original);
    assert_eq!(evaluate(&c), r);
    c.error = Some("routing gap".into());
    assert_eq!(evaluate(&c)["contenders"], json!({}));
}
#[test]
fn native_capture_replays_exactly_and_rejects_invalid_forecasts() {
    let c = fixture();
    std::fs::create_dir(&c.profile.settings.output).unwrap();
    let path = c.profile.settings.output.join("capture.json");
    std::fs::write(&path, serde_json::to_vec(&c).unwrap()).unwrap();
    let mut expected = evaluate(&c);
    expected["capture_limit_reached"] = json!(false);
    assert_eq!(super::replay(&path).unwrap(), expected);
    let mut changed = c.clone();
    changed.forecasts.get_mut("recent").unwrap()[0]
        .scores
        .push((
            Expert {
                layer: 99,
                expert: 0,
            },
            1.,
        ));
    std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(super::replay(&path).is_err());
    std::fs::remove_dir_all(&c.profile.settings.output).unwrap();
}
struct Slow;
impl Predictor for Slow {
    fn version(&self) -> &'static str {
        "1"
    }
    fn predict(&mut self, _: &Context) -> Result<Vec<(Expert, f64)>, String> {
        thread::sleep(Duration::from_millis(200));
        Ok(vec![])
    }
}
fn record(sequence: u64) -> Arc<Record> {
    let now = now_ns();
    Arc::new(Record {
        sequence,
        graph: 0,
        phase: "decode",
        batch_size: 1,
        concurrency: 1,
        sample: RoutingSample {
            residency_generation: None,
            gpu_ns: None,
            submission: sequence,
            demand_lower_bound_ns: now,
            available_ns: now,
            layers: vec![LayerDemand {
                layer: 0,
                tokens: 2,
                top_k: 1,
                counts: vec![0, 2],
            }],
        },
    })
}
#[test]
fn slow_plugin_is_disabled_without_delaying_other_predictors_or_shutdown() {
    let mut p = profile("slow");
    p.settings.budget_ns = 10_000_000;
    p.settings.predictors[1] = Selection {
        name: "slow".into(),
        config: json!({}),
    };
    let output = p.settings.output.clone();
    let mut registry = Registry::builtins();
    registry.register("slow", |_| Ok(Box::new(Slow))).unwrap();
    let observer = Observer::start(p, registry).unwrap();
    for i in 1..=5 {
        observer.publish(record(i));
        thread::sleep(Duration::from_millis(8));
    }
    let start = Instant::now();
    observer.close();
    assert!(start.elapsed() < Duration::from_millis(150));
    let report = observer.snapshot();
    assert_eq!(report["workers"]["slow"]["timeouts"], 1);
    assert!(report["workers"]["recent"]["completed"].as_u64().unwrap() > 0);
    assert_eq!(report["routing_frames"], 5);
    assert!(report["error"].is_null());
    std::fs::remove_dir_all(output).unwrap();
}
#[test]
fn internal_observation_does_not_depend_on_the_http_retention_ring() {
    let p = profile("retention");
    let output = p.settings.output.clone();
    let observer = Observer::start(p, Registry::builtins()).unwrap();
    // No HTTP reader or terminal reader runs while more than 32 routes are published.
    for i in 1..=60 {
        observer.publish(record(i));
        thread::sleep(Duration::from_millis(2));
    }
    observer.close();
    assert_eq!(observer.snapshot()["routing_frames"], 60);
    assert!(observer.snapshot()["error"].is_null());
    std::fs::remove_dir_all(output).unwrap();
}
#[test]
fn uniform_canary_runs_through_the_same_observer_and_replay_as_recent() {
    let mut p = profile("canary-observer");
    p.settings.predictors[1] = Selection {
        name: "uniform_canary".into(),
        config: json!({}),
    };
    let output = p.settings.output.clone();
    let observer = Observer::start(p, Registry::builtins()).unwrap();
    for seq in 1..=8 {
        observer.publish(record(seq));
        thread::sleep(Duration::from_millis(5));
    }
    observer.close();
    let capture: Capture = serde_json::from_slice(
        &std::fs::read(output.join("capture.json")).unwrap(),
    ).unwrap();
    for name in ["recent", "uniform_canary"] {
        assert!(!capture.forecasts[name].is_empty());
        assert_eq!(capture.workers[name].failures, 0);
        for f in &capture.forecasts[name] {
            f.validate(&capture.profile).unwrap();
        }
    }
    for f in &capture.forecasts["uniform_canary"] {
        assert_eq!(f.scores, vec![
            (Expert { layer: 0, expert: 0 }, 1.),
            (Expert { layer: 0, expert: 1 }, 1.),
        ]);
    }
    assert_eq!(super::replay(&output.join("capture.json")).unwrap(), evaluate(&capture));
    std::fs::remove_dir_all(output).unwrap();
}
#[test]
fn disabled_telemetry_never_starts_prediction_workers() {
    let telemetry = super::Telemetry::default();
    assert_eq!(telemetry.predictions()["enabled"], false);
    assert!(telemetry.start_predictions(profile("disabled")).is_err());
}

#[test]
fn speculative_routing_reports_target_work_as_verification() {
    let telemetry = super::Telemetry::default();
    telemetry.publish(crate::parallax::RoutingReadout::Valid(RoutingSample {
            residency_generation: None,
            gpu_ns: None,
        submission: 7, demand_lower_bound_ns: 100, available_ns: 200,
        layers: vec![LayerDemand { layer: 3, tokens: 5, top_k: 2, counts: vec![6, 4] }],
    }), 0, crate::execution::GraphKind::Speculative, 2, 2);
    let report = telemetry.snapshot(0, "synthetic");
    assert_eq!(report["frames"][0]["phase"], "verify");
    assert_eq!(report["frames"][0]["sample"]["layers"][0]["tokens"], 5);
    assert_eq!(report["frames"][0]["sample"]["layers"][0]["counts"], json!([6, 4]));
    assert_eq!(report["invalid_total"], 0);
}

#[test]
fn invalid_last_observation_cannot_be_hidden_by_shutdown() {
    let p = profile("invalid-close");
    let output = p.settings.output.clone();
    let observer = Observer::start(p, Registry::builtins()).unwrap();
    observer.invalidate();
    observer.close();
    assert!(
        observer.snapshot()["error"]
            .as_str()
            .unwrap()
            .contains("delivery failure")
    );
    assert_eq!(observer.snapshot()["contenders"], json!({}));
    assert!(
        super::replay(&output.join("capture.json"))
            .unwrap_err()
            .contains("delivery failure")
    );
    std::fs::remove_dir_all(output).unwrap();
}

#[test]
fn trace_limits_preserve_captures_while_predictors_continue() {
    for (name, max_frames, cells, reason) in [
        ("cell-limit", 64, 24, "demand_scores"),
        ("frame-limit", 4, 2_000_000, "frames"),
    ] {
        let mut p = profile(name);
        p.settings.max_frames = max_frames;
        let output = p.settings.output.clone();
        let observer = Observer::start_with_limit(p, Registry::builtins(), cells).unwrap();
        for i in 1..=40 {
            observer.publish(record(i));
            thread::sleep(Duration::from_millis(5));
        }
        observer.close();
        let capture: Capture = serde_json::from_slice(
            &std::fs::read(output.join("capture.json")).unwrap(),
        ).unwrap();
        let retained = capture.observations.iter().map(|o| o.demands.len()).sum::<usize>()
            + capture.forecasts.values().flatten().map(|f| f.scores.len()).sum::<usize>();
        assert!(retained <= cells);
        assert!(!capture.observations.is_empty());
        assert!(capture.observations.len() < 40);
        for counts in capture.workers.values() {
            assert_eq!(counts.submitted, counts.completed);
            assert_eq!(counts.failures, 0);
            assert_eq!(counts.timeouts, 0);
            assert!(!counts.disabled);
        }
        let report = observer.snapshot();
        assert!(report["error"].is_null());
        assert_eq!(report["routing_frames"], 40);
        assert_eq!(report["continuous"], true);
        assert_eq!(report["trace_complete"], true);
        assert!(report["evaluation_window_frames"].as_u64().unwrap() <= max_frames as u64);
        assert_eq!(evaluate(&capture)["capture_limit_reason"], reason);
        assert_eq!(report["contenders"].as_object().unwrap().len(), 4);
        assert_eq!(super::replay(&output.join("capture.json")).unwrap(), evaluate(&capture));
        std::fs::remove_dir_all(output).unwrap();
    }
}

#[test]
fn capture_without_retention_reason_remains_readable() {
    let c = fixture();
    std::fs::create_dir(&c.profile.settings.output).unwrap();
    let path = c.profile.settings.output.join("capture.json");
    let mut legacy = serde_json::to_value(&c).unwrap();
    legacy.as_object_mut().unwrap().remove("limit_reason");
    std::fs::write(&path, serde_json::to_vec(&legacy).unwrap()).unwrap();
    assert_eq!(super::replay(&path).unwrap(), evaluate(&c));
    std::fs::remove_dir_all(c.profile.settings.output).unwrap();
}

#[test]
fn rolling_statistics_preserve_lifetime_residency_and_pending_transfers() {
    let mut c = fixture();
    c.observations.clear();
    c.forecasts.get_mut("recent").unwrap().clear();
    c.profile.settings.max_frames = 2;
    let mut live = Continuous::new(&c, 12);
    let template = fixture();
    for seq in 0..100 {
        let mut o = template.observations[0].clone();
        o.seq = seq;
        o.demand_ns = seq * 10000;
        o.available_ns = o.demand_ns + 1;
        live.observe(&o);
        let mut f = template.forecasts["recent"][0].clone();
        f.cutoff = seq;
        f.target = seq + 1;
        f.available_ns = o.available_ns;
        f.started_ns = o.available_ns;
        f.completed_ns = o.available_ns + 10;
        f.expected_demand_ns = o.available_ns + 10000;
        f.expires_ns = o.available_ns + 20000;
        live.forecast(f);
    }
    let (window, totals) = live.snapshot(&c.workers);
    let report = evaluate_continuous(&window, totals);
    assert_eq!(report["routing_frames"], 100);
    assert_eq!(report["evaluation_window_frames"], 2);
    assert_eq!(report["contenders"]["recent/economic"]["hits"], 198);
    assert_eq!(report["contenders"]["recent/economic"]["promotion_count"], 1);
    assert_eq!(report["contenders"]["recent/static"]["hits"], 0);
    assert_eq!(report["contenders"]["recent/static"]["promotion_count"], 0);
}

#[test]
fn failed_transfer_calibration_keeps_predictions_but_suppresses_movement() {
    let mut c = fixture();
    c.profile.hardware.transfer_calibration = Some(super::calibration::Calibration {
        version: 1,
        memory_path: "synthetic failed calibration".into(),
        gpu_uuid: "fixture".into(),
        elapsed_ns: 1_000_000_000,
        host_to_device: None,
        device_to_host: None,
        samples: vec![],
        failure: Some("sampling budget exhausted".into()),
    });
    let report = evaluate(&c);
    assert!(report["error"].is_null());
    assert_eq!(report["contenders"]["recent/economic"]["promotion_count"], 0);
    assert_eq!(report["contenders"]["recent/economic"]["forecast_precision"], 1.0);
}

#[test]
fn per_layer_ranking_does_not_confuse_coverage_with_a_residency_budget() {
    let mut c = fixture();
    c.profile.model.experts = (0..2).flat_map(|layer|(0..10).map(move |expert|Expert{layer,expert})).collect();
    c.profile.model.expert_bytes = vec![100;20];
    c.profile.hardware.hbm_bytes = 1000;
    c.profile.hardware.backing_bytes = 2000;
    c.profile.hardware.cold_cost_ns = 0;
    for o in &mut c.observations { o.demands = vec![(Expert{layer:1,expert:9},2)]; }
    for f in c.forecasts.values_mut().flatten() {
        f.scores = (0..10).map(|expert|(Expert{layer:0,expert},100.-expert as f64)).collect();
        f.scores.push((Expert{layer:1,expert:9},1.));
    }
    assert_eq!(evaluate(&c)["contenders"]["recent/static"]["demand_coverage"],0.);
    c.profile.settings.evaluation=Some(EvaluationConfig{version:2,top_k_per_layer:8,max_promotions_per_forecast:128});
    let r=evaluate(&c);
    let row=&r["contenders"]["recent/static"];
    assert_eq!(row["forecast_recall"],1.);
    assert_eq!(row["demand_coverage"],0.8);
    assert_eq!(row["residency_budget_ranking"]["timely_demand_coverage"],0.);
    assert_eq!(row["coverage_by_layer_k"].as_object().unwrap().len(),4);
    assert!(row.get("estimated_net_route_cost_saved_ns").is_none());
    assert_eq!(row["modeled_balance_ns"],0.);
}

#[test]
fn frozen_frequency_uses_each_eligible_observation_once() {
    let p=profile("frequency");
    let mut predictor=Registry::builtins().create(&Selection{name:"frequency".into(),config:json!({"warmup_frames":2})}).unwrap();
    let observations=fixture().observations;
    let mut context=Context{model:Arc::new(p.model),target_offset:1,horizon_ns:10000,history:vec![Arc::new(observations[0].clone())],available_ns:1};
    let first=predictor.predict(&context).unwrap();
    assert_eq!(first,vec![(Expert{layer:0,expert:1},2.)]);
    context.history.push(Arc::new(observations[1].clone()));
    assert_eq!(predictor.predict(&context).unwrap(),first);
    let mut changed=observations[2].clone(); changed.demands=vec![(Expert{layer:0,expert:0},1000)];
    context.history.push(Arc::new(changed));
    assert_eq!(predictor.predict(&context).unwrap(),first);
    assert!(Registry::builtins().create(&Selection{name:"frequency".into(),config:json!({"warmup_frames":"bad"})}).is_err());
}

#[test]
fn alternate_evaluation_preserves_source_and_matches_continuous_ledger() {
    let mut c=fixture();
    std::fs::create_dir(&c.profile.settings.output).unwrap();
    let path=c.profile.settings.output.join("capture.json");
    let original=serde_json::to_vec(&c).unwrap();
    std::fs::write(&path,&original).unwrap();
    let config=EvaluationConfig{version:2,top_k_per_layer:32,max_promotions_per_forecast:128};
    let alternate=super::reevaluate(&path,config.clone()).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(),original);
    assert_eq!(super::replay(&path).unwrap(),evaluate(&c));
    c.profile.settings.evaluation=Some(config);
    let full=evaluate(&c);
    assert_eq!(alternate["contenders"],full["contenders"]);
    let mut empty=c.clone();empty.observations.clear();empty.forecasts.get_mut("recent").unwrap().clear();
    let mut live=Continuous::new(&empty,10000);
    for (observation,forecast) in c.observations.iter().zip(&c.forecasts["recent"]) {
        live.observe(observation);live.forecast(forecast.clone());
    }
    let (window,totals)=live.snapshot(&c.workers);
    let online=evaluate_continuous(&window,totals);
    for key in ["hits","misses","cold_demand_avoided","eviction_induced_misses","modeled_balance_ns","predictor_runtime_total_ns","modeled_movement_cost_ns"] {
        assert_eq!(online["contenders"]["recent/economic"][key],full["contenders"]["recent/economic"][key],"{key}");
    }
    std::fs::remove_dir_all(c.profile.settings.output).unwrap();
}

#[test]
fn expanded_live_workers_keep_a_ledger_after_trace_retention_fills() {
    let mut p=profile("expanded-live");
    p.settings.max_frames=4;
    p.settings.evaluation=Some(EvaluationConfig{version:2,top_k_per_layer:32,max_promotions_per_forecast:128});
    p.settings.predictors.push(Selection{name:"frequency".into(),config:json!({"warmup_frames":4})});
    let output=p.settings.output.clone();
    let observer=Observer::start(p,Registry::builtins()).unwrap();
    for i in 1..=16 { observer.publish(record(i));thread::sleep(Duration::from_millis(5)); }
    observer.close();
    let report=observer.snapshot();
    assert_eq!(report["routing_frames"],16);
    assert_eq!(report["trace_complete"],true);
    assert_eq!(report["evaluation"]["version"],2);
    assert_eq!(report["contenders"].as_object().unwrap().len(),6);
    assert!(report["error"].is_null());
    assert!(report["workers"]["frequency"]["completed"].as_u64().unwrap()>0);
    for row in report["contenders"].as_object().unwrap().values() {
        assert!(row.get("modeled_balance_ns").is_some());
        assert!(row.get("estimated_net_route_cost_saved_ns").is_none());
    }
    let saved:serde_json::Value=serde_json::from_slice(&std::fs::read(output.join("comparison.json")).unwrap()).unwrap();
    assert_eq!(super::replay(&output.join("capture.json")).unwrap(),saved);
    std::fs::remove_dir_all(output).unwrap();
}
