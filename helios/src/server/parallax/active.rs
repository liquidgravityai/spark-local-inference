//! Bounded economic proposals against authoritative physical residency.
use super::prediction::{Expert, Forecast, Profile};
use crate::parallax::residency::{Plan, Snapshot, execution::ExecutionBinding};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    #[default]
    EconomicExchange,
    /// Bounded exploration follows ranking without assuming an execution saving.
    PredictorRank,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub policy: Policy,
    pub predictor: String,
    pub max_promotions: usize,
    pub interval_ns: u64,
    pub hold_ns: u64,
    /// Assumed quiescent drain/publication cost for a complete transaction.
    pub publication_ns: u64,
}
impl Config {
    pub fn validate(&self, profile: &Profile) -> Result<(), String> {
        if !(1..=64).contains(&self.max_promotions)
            || !(100_000_000..=60_000_000_000).contains(&self.interval_ns)
            || self.hold_ns < self.interval_ns
            || self.hold_ns > 600_000_000_000
            || self.publication_ns == 0
            || !profile
                .settings
                .predictors
                .iter()
                .any(|p| p.name == self.predictor)
        {
            return Err("active placement requires a registered predictor, 1..64 promotions, 0.1..60 s interval, bounded hold time and positive publication cost".into());
        }
        Ok(())
    }
}

pub struct Active {
    config: Config,
    binding: ExecutionBinding,
    storage: BTreeMap<Expert, (u64, usize)>,
    held_until: BTreeMap<Expert, u64>,
    next_ns: u64,
    proposed: u64,
    no_movement: u64,
    stale_forecasts: u64,
    last_gain_ns: f64,
    failure: Option<String>,
}
impl Active {
    pub fn new(
        profile: &Profile,
        binding: Option<ExecutionBinding>,
    ) -> Result<Option<Self>, String> {
        let Some(config) = &profile.settings.active else {
            return Ok(None);
        };
        config.validate(profile)?;
        let binding = binding.ok_or("active placement requires physical expert storage")?;
        let snapshot = binding.snapshot().map_err(|e| e.to_string())?;
        if snapshot.catalog_id != profile.model.identity {
            return Err("active policy and physical storage catalogs differ".into());
        }
        let storage: BTreeMap<_, _> = binding
            .expert_storage()
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|(e, bytes, parts)| (e, (bytes, parts)))
            .collect();
        if storage.keys().copied().collect::<Vec<_>>() != profile.model.experts {
            return Err("active policy requires the complete canonical storage catalog".into());
        }
        Ok(Some(Self {
            config: config.clone(),
            binding,
            storage,
            held_until: BTreeMap::new(),
            next_ns: 0,
            proposed: 0,
            no_movement: 0,
            stale_forecasts: 0,
            last_gain_ns: 0.,
            failure: None,
        }))
    }
    pub fn disable(&mut self, reason: &str) {
        self.failure.get_or_insert(reason.into());
    }
    pub fn forecast(
        &mut self,
        forecast: &Forecast,
        profile: &Profile,
        now: u64,
        target_demand: u64,
    ) {
        if self.failure.is_some()
            || forecast.provider != self.config.predictor
            || now < self.next_ns
        {
            return;
        }
        if self.binding.latest_demand() >= target_demand {
            self.stale_forecasts += 1;
            return;
        }
        self.next_ns = now.saturating_add(self.config.interval_ns);
        let state = match self.binding.snapshot() {
            Ok(s) => s,
            Err(e) => {
                self.failure = Some(e.to_string());
                return;
            }
        };
        if state.pending || state.poisoned {
            return;
        }
        let Some((plan, gain)) = propose(
            &self.config,
            &state,
            &self.storage,
            &self.held_until,
            forecast,
            profile,
            now,
        ) else {
            self.no_movement += 1;
            return;
        };
        match self.binding.submit_for_demand(plan.clone(), target_demand) {
            Ok(()) => {
                self.proposed += 1;
                self.last_gain_ns = gain;
                for e in plan.promote.iter().chain(&plan.evict) {
                    self.held_until
                        .insert(*e, now.saturating_add(self.config.hold_ns));
                }
            }
            Err(crate::parallax::residency::Error::Busy) => {}
            Err(crate::parallax::residency::Error::Expired) => {
                self.stale_forecasts += 1;
            }
            Err(e) => self.failure = Some(e.to_string()),
        }
    }
    pub fn report(&self) -> Value {
        json!({"predictor":self.config.predictor,"policy":self.config.policy,
            "proposed_transactions":self.proposed,"no_movement_decisions":self.no_movement,"stale_forecasts":self.stale_forecasts,
            "last_estimated_gain_ns":if self.config.policy == Policy::EconomicExchange { Some(self.last_gain_ns) } else { None },"failure":self.failure,
            "publication_cost_assumed_ns":self.config.publication_ns,
            "cost_scope":if self.config.policy == Policy::PredictorRank { "bounded ranking exploration; no assumed cold penalty or promised gain" } else { "forecast demand and cold penalty are estimates; no throughput claim" }})
    }
}

fn propose(
    config: &Config,
    state: &Snapshot,
    storage: &BTreeMap<Expert, (u64, usize)>,
    held: &BTreeMap<Expert, u64>,
    f: &Forecast,
    profile: &Profile,
    now: u64,
) -> Option<(Plan, f64)> {
    let hardware = &profile.hardware;
    if hardware
        .transfer_calibration
        .as_ref()
        .is_none_or(|c| !c.usable())
        || now < f.completed_ns
        || now >= f.expected_demand_ns
        || now > f.expires_ns
    {
        return None;
    }
    let scores: BTreeMap<_, _> = f.scores.iter().copied().collect();
    let eligible = |expert: &Expert| held.get(expert).is_none_or(|until| *until <= now);
    let score = |expert: &Expert| scores.get(expert).copied().unwrap_or(0.);
    let mut cold: Vec<_> = state
        .experts
        .iter()
        .filter(|(e, hot)| !**hot && eligible(e))
        .map(|(e, _)| *e)
        .collect();
    cold.sort_by(|a, b| score(b).total_cmp(&score(a)).then(a.cmp(b)));
    let mut hot: Vec<_> = state
        .experts
        .iter()
        .filter(|(e, hot)| **hot && eligible(e))
        .map(|(e, _)| *e)
        .collect();
    hot.sort_by(|a, b| score(a).total_cmp(&score(b)).then(a.cmp(b)));
    let (mut promote, mut evict, mut removed) = (Vec::new(), Vec::new(), BTreeSet::new());
    let (mut slots, mut copy_ns, mut gain) = (state.free_slots, 0f64, 0f64);
    for expert in cold {
        if promote.len() == config.max_promotions {
            break;
        }
        let (bytes, parts) = storage[&expert];
        if parts > slots {
            continue;
        }
        // Equal representation sizes preserve the resident byte budget. The
        // backend independently admits each exact physical size class.
        let Some(victim) = hot
            .iter()
            .find(|v| !removed.contains(*v) && storage[v] == (bytes, parts))
        else {
            continue;
        };
        let copy = bytes as f64 / hardware.bandwidth_bytes_per_ns
            + hardware.transfer_latency_ns as f64 * parts as f64;
        let benefit = (score(&expert) - score(victim)) * hardware.cold_cost_ns as f64 - copy;
        if (config.policy == Policy::EconomicExchange && (!benefit.is_finite() || benefit <= 0.))
            || (config.policy == Policy::PredictorRank && score(&expert) <= score(victim)) {
            continue;
        }
        if now as f64 + copy_ns + copy + config.publication_ns as f64 > f.expected_demand_ns as f64
        {
            continue;
        }
        slots -= parts;
        copy_ns += copy;
        gain += benefit;
        removed.insert(*victim);
        promote.push(expert);
        evict.push(*victim);
    }
    gain -= config.publication_ns as f64;
    if promote.is_empty() || (config.policy == Policy::EconomicExchange && (!gain.is_finite() || gain <= 0.)) {
        return None;
    }
    Some((
        Plan {
            version: 1,
            catalog_id: state.catalog_id.clone(),
            generation: state.generation,
            valid_until_ns: f.expires_ns.min(f.expected_demand_ns),
            promote,
            evict,
        },
        gain,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::parallax::{
        calibration::{Calibration, CopyCost},
        prediction_tests::{fixture, profile},
    };
    fn inputs() -> (
        Config,
        Profile,
        Snapshot,
        BTreeMap<Expert, (u64, usize)>,
        Forecast,
    ) {
        let config = Config {
            policy: Policy::EconomicExchange,
            predictor: "recent".into(),
            max_promotions: 2,
            interval_ns: 100_000_000,
            hold_ns: 100_000_000,
            publication_ns: 100,
        };
        let mut profile = profile("physical-policy");
        profile.hardware.bandwidth_bytes_per_ns = 1.;
        profile.hardware.transfer_latency_ns = 0;
        let fit = CopyCost {
            bandwidth_bytes_per_ns: 1.,
            latency_ns: 0,
        };
        profile.hardware.transfer_calibration = Some(Calibration {
            version: 1,
            memory_path: "synthetic test".into(),
            gpu_uuid: "synthetic".into(),
            elapsed_ns: 1,
            host_to_device: Some(fit.clone()),
            device_to_host: Some(fit),
            samples: vec![],
            failure: None,
        });
        let expert = |expert| Expert { layer: 0, expert };
        let experts = (0..4).map(|e| (expert(e), e < 2)).collect();
        let state = Snapshot {
            catalog_id: profile.model.identity.clone(),
            generation: 4,
            experts,
            free_slots: 2,
            hbm_allocated_bytes: 300,
            backing_allocated_bytes: 400,
            readers: 1,
            pending: false,
            poisoned: false,
            statistics: Default::default(),
        };
        let storage = (0..4).map(|e| (expert(e), (100, 2))).collect();
        let mut forecast = fixture().forecasts["recent"][0].clone();
        forecast.completed_ns = 10;
        forecast.expected_demand_ns = 10000;
        forecast.expires_ns = 20000;
        forecast.scores = (0..4)
            .map(|e| (expert(e), if e < 2 { 1. } else { 20. }))
            .collect();
        (config, profile, state, storage, forecast)
    }
    #[test]
    fn physical_policy_charges_publication_and_only_uses_spare_slots() {
        let (config, profile, state, storage, forecast) = inputs();
        let (plan, gain) = propose(
            &config,
            &state,
            &storage,
            &BTreeMap::new(),
            &forecast,
            &profile,
            20,
        )
        .unwrap();
        assert_eq!(plan.generation, 4);
        assert_eq!(
            plan.promote,
            vec![Expert {
                layer: 0,
                expert: 2
            }]
        );
        assert_eq!(
            plan.evict,
            vec![Expert {
                layer: 0,
                expert: 0
            }]
        );
        assert_eq!(gain, 18800.);
        assert_eq!(plan.valid_until_ns, forecast.expected_demand_ns);
    }
    #[test]
    fn physical_policy_rejects_late_unprofitable_uncalibrated_and_held_moves() {
        let (mut config, mut profile, state, storage, mut forecast) = inputs();
        let held = storage.keys().map(|e| (*e, 100)).collect();
        assert!(propose(&config, &state, &storage, &held, &forecast, &profile, 20).is_none());
        forecast.expected_demand_ns = 200;
        assert!(
            propose(
                &config,
                &state,
                &storage,
                &BTreeMap::new(),
                &forecast,
                &profile,
                20
            )
            .is_none()
        );
        forecast.expected_demand_ns = 10000;
        config.publication_ns = 50000;
        assert!(
            propose(
                &config,
                &state,
                &storage,
                &BTreeMap::new(),
                &forecast,
                &profile,
                20
            )
            .is_none()
        );
        config.publication_ns = 100;
        profile.hardware.transfer_calibration = None;
        assert!(
            propose(
                &config,
                &state,
                &storage,
                &BTreeMap::new(),
                &forecast,
                &profile,
                20
            )
            .is_none()
        );
    }
    #[test]
    fn active_configuration_cannot_silently_run_without_physical_storage() {
        let (config, mut profile, _, _, _) = inputs();
        profile.settings.active = Some(config);
        assert!(Active::new(&profile, None).is_err());
        profile.settings.active.as_mut().unwrap().interval_ns = 0;
        assert!(profile.validate().is_err());
    }
    #[test]
    fn ranking_explores_without_a_cold_price_but_keeps_deadlines_and_hold_limits() {
        let (mut config, mut profile, state, storage, mut forecast) = inputs();
        config.policy = Policy::PredictorRank;
        profile.hardware.cold_cost_ns = 0;
        assert!(propose(&config,&state,&storage,&BTreeMap::new(),&forecast,&profile,20).is_some());
        let held=storage.keys().map(|e|(*e,100)).collect();
        assert!(propose(&config,&state,&storage,&held,&forecast,&profile,20).is_none());
        forecast.expected_demand_ns=100;
        assert!(propose(&config,&state,&storage,&BTreeMap::new(),&forecast,&profile,20).is_none());
        // An uninformed uniform forecast must not churn placement on tie order.
        forecast.expected_demand_ns = 10000;
        for (_, score) in &mut forecast.scores { *score = 1.; }
        assert!(propose(&config,&state,&storage,&BTreeMap::new(),&forecast,&profile,20).is_none());
    }
}
