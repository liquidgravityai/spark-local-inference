//! Build-time predictor registration and runtime configuration. No dynamic loading.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

type Result<T> = std::result::Result<T, String>;
pub use crate::parallax::ExpertId as Expert;
#[derive(Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub identity: String,
    pub architecture: String,
    pub experts: Vec<Expert>,
    pub expert_bytes: Vec<u64>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Hardware {
    pub identity: String,
    pub hbm_bytes: u64,
    pub backing_bytes: u64,
    pub bandwidth_bytes_per_ns: f64,
    pub transfer_latency_ns: u64,
    pub synchronization_ns: u64,
    pub cold_cost_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_calibration: Option<super::calibration::Calibration>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Selection {
    pub name: String,
    #[serde(default)]
    pub config: Value,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationConfig {
    pub version: u32,
    pub top_k_per_layer: usize,
    pub max_promotions_per_forecast: usize,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub version: u32,
    pub catalog: PathBuf,
    pub hardware: PathBuf,
    pub output: PathBuf,
    pub predictors: Vec<Selection>,
    pub target_offset: u64,
    pub horizon_ns: u64,
    pub budget_ns: u64,
    pub max_frames: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<EvaluationConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<super::active::Config>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Profile {
    pub settings: Settings,
    pub model: Catalog,
    pub hardware: Hardware,
}
impl Profile {
    pub fn load(path: &Path, package_id: &str) -> Result<Self> {
        let settings: Settings =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let model: Catalog =
            serde_json::from_slice(&std::fs::read(&settings.catalog).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let hardware: Hardware =
            serde_json::from_slice(&std::fs::read(&settings.hardware).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let profile = Self {
            settings,
            model,
            hardware,
        };
        profile.validate()?;
        for selection in &profile.settings.predictors {
            Registry::builtins().create(selection)?;
        }
        if !profile
            .model
            .identity
            .ends_with(&format!(";package={package_id}"))
        {
            return Err("Parallax catalog/package identity mismatch".into());
        }
        Ok(profile)
    }
    pub fn validate(&self) -> Result<()> {
        if let Some(active) = &self.settings.active { active.validate(self)?; }
        let s = &self.settings;
        if s.evaluation.as_ref().is_some_and(|e| e.version != 2
            || !(1..=4096).contains(&e.top_k_per_layer)
            || !(1..=512).contains(&e.max_promotions_per_forecast)) {
            return Err("evaluation requires version 2, 1..4096 ranks per layer and 1..512 promotions per forecast".into());
        }
        let m = &self.model;
        let h = &self.hardware;
        if s.version != 1
            || !(1..=4).contains(&s.predictors.len())
            || s.target_offset == 0
            || s.target_offset > 1024
            || s.horizon_ns == 0
            || s.horizon_ns > 60_000_000_000
            || s.budget_ns == 0
            || s.budget_ns > 10_000_000_000
            || !(1..=2048).contains(&s.max_frames)
        {
            return Err("invalid Parallax window, timing or provider limits".into());
        }
        if m.identity.is_empty()
            || m.architecture.is_empty()
            || m.experts.is_empty()
            || m.experts.len() > 65536
            || m.experts.len() != m.expert_bytes.len()
            || m.expert_bytes.contains(&0)
            || m.experts.windows(2).any(|p| p[0] >= p[1])
            || m.experts.iter().any(|e| e.layer >= 256 || e.expert >= 4096)
        {
            return Err("invalid canonical expert catalog".into());
        }
        let total = m
            .expert_bytes
            .iter()
            .try_fold(0u64, |a, b| a.checked_add(*b))
            .ok_or("expert size overflow")?;
        if h.identity.is_empty()
            || h.backing_bytes < total
            || h.hbm_bytes > total
            || !h.bandwidth_bytes_per_ns.is_finite()
            || h.bandwidth_bytes_per_ns <= 0.
            || h.bandwidth_bytes_per_ns < 0.000001
            || h.transfer_latency_ns > 1_000_000_000_000
            || h.synchronization_ns > 1_000_000_000_000
            || h.cold_cost_ns > 1_000_000_000_000
        {
            return Err("invalid hardware costs or backing capacity".into());
        }
        let worst = (m.expert_bytes.iter().copied().max().unwrap_or(0) as f64
            / h.bandwidth_bytes_per_ns
            + h.transfer_latency_ns as f64
            + h.synchronization_ns as f64)
            * s.max_frames as f64;
        if !worst.is_finite() || worst > u64::MAX as f64 / 4. {
            return Err("hardware transfer costs exceed bounded accounting range".into());
        }
        let mut names = BTreeSet::new();
        for selection in &s.predictors {
            if !names.insert(&selection.name) {
                return Err("duplicate predictor selection".into());
            }
        }
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Observation {
    pub seq: u64,
    pub demand_ns: u64,
    pub available_ns: u64,
    pub phase: String,
    pub batch_size: usize,
    pub concurrency: usize,
    pub demands: Vec<(Expert, u64)>,
}
/// Only the eligible bounded history is exposed to a provider.
pub struct Context {
    pub model: Arc<Catalog>,
    pub target_offset: u64,
    pub horizon_ns: u64,
    pub history: Vec<Arc<Observation>>,
    pub available_ns: u64,
}
/// Scores need not be calibrated probabilities. These baselines forecast demand counts.
pub trait Predictor: Send + 'static {
    fn version(&self) -> &'static str;
    fn predict(&mut self, context: &Context) -> Result<Vec<(Expert, f64)>>;
}
pub type Factory = fn(&Value) -> Result<Box<dyn Predictor>>;
pub struct Registry {
    factories: BTreeMap<&'static str, Factory>,
}
impl Registry {
    pub fn builtins() -> Self {
        let mut registry = Self {
            factories: BTreeMap::new(),
        };
        registry
            .register("recent", |config| {
                if !config.is_null() && config != &json!({}) {
                    return Err("recent accepts an empty configuration".into());
                }
                Ok(Box::new(Recent))
            })
            .unwrap();
        registry
            .register("ewma", |config| {
                if !config.is_null()
                    && (!config.is_object()
                        || config.as_object().unwrap().keys().any(|k| k != "alpha"))
                {
                    return Err("ewma accepts only alpha".into());
                }
                let alpha = if config.get("alpha").is_some() {
                    config["alpha"].as_f64().ok_or("alpha must be numeric")?
                } else {
                    0.5
                };
                if !alpha.is_finite() || alpha <= 0. || alpha > 1. {
                    return Err("alpha must be in (0,1]".into());
                }
                Ok(Box::new(Ewma { alpha }))
            })
            .unwrap();
        registry.register("frequency", |config| {
            if !config.is_object() || config.as_object().unwrap().keys().any(|k| k != "warmup_frames") {
                return Err("frequency accepts warmup_frames".into());
            }
            let warmup = match config.get("warmup_frames") { Some(value) => value.as_u64().ok_or("warmup_frames must be an integer")?, None => 128 };
            if !(1..=4096).contains(&warmup) { return Err("warmup_frames must be 1..4096".into()); }
            Ok(Box::new(Frequency { warmup, seen: 0, cutoff: None, counts: BTreeMap::new() }))
        }).unwrap();
        registry.register("uniform_canary", super::canary::create).unwrap();
        registry
    }
    pub fn register(&mut self, name: &'static str, factory: Factory) -> Result<()> {
        if name.is_empty() || self.factories.contains_key(name) {
            return Err("duplicate or empty predictor registration".into());
        }
        self.factories.insert(name, factory);
        Ok(())
    }
    pub fn create(&self, selection: &Selection) -> Result<Box<dyn Predictor>> {
        self.factories
            .get(selection.name.as_str())
            .ok_or_else(|| format!("predictor {} is not compiled into Flare", selection.name))?(
            &selection.config,
        )
    }
}
struct Recent;
impl Predictor for Recent {
    fn version(&self) -> &'static str {
        "1"
    }
    fn predict(&mut self, context: &Context) -> Result<Vec<(Expert, f64)>> {
        Ok(context
            .history
            .last()
            .ok_or("missing observation")?
            .demands
            .iter()
            .map(|(e, n)| (*e, *n as f64))
            .collect())
    }
}
struct Ewma {
    alpha: f64,
}
impl Predictor for Ewma {
    fn version(&self) -> &'static str {
        "1"
    }
    fn predict(&mut self, context: &Context) -> Result<Vec<(Expert, f64)>> {
        let mut scores = BTreeMap::new();
        for observation in &context.history {
            for score in scores.values_mut() {
                *score *= 1. - self.alpha;
            }
            for (e, n) in &observation.demands {
                *scores.entry(*e).or_insert(0.) += self.alpha * *n as f64;
            }
        }
        Ok(scores.into_iter().collect())
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Forecast {
    pub provider: String,
    pub provider_version: String,
    pub cutoff: u64,
    pub target: u64,
    pub available_ns: u64,
    pub started_ns: u64,
    pub completed_ns: u64,
    pub compute_ns: u64,
    pub expected_demand_ns: u64,
    pub expires_ns: u64,
    pub scores: Vec<(Expert, f64)>,
}
impl Forecast {
    pub fn validate(&self, profile: &Profile) -> Result<()> {
        if self.cutoff.checked_add(profile.settings.target_offset) != Some(self.target)
            || self.started_ns < self.available_ns
            || self.completed_ns < self.started_ns
            || self.compute_ns > self.completed_ns - self.started_ns
            || self.scores.len() > profile.model.experts.len()
        {
            return Err("invalid forecast timing or size".into());
        }
        let mut seen = BTreeSet::new();
        for (expert, score) in &self.scores {
            if !score.is_finite()
                || *score < 0.
                || !seen.insert(*expert)
                || profile.model.experts.binary_search(expert).is_err()
            {
                return Err("invalid forecast expert or score".into());
            }
        }
        Ok(())
    }
}

/// Fits an initial eligible observation prefix, then freezes its frequency profile.
struct Frequency {
    warmup: u64,
    seen: u64,
    cutoff: Option<u64>,
    counts: BTreeMap<Expert, u64>,
}
impl Predictor for Frequency {
    fn version(&self) -> &'static str { "1" }
    fn predict(&mut self, context: &Context) -> Result<Vec<(Expert, f64)>> {
        for observation in &context.history {
            if self.seen == self.warmup { break; }
            if self.cutoff.is_some_and(|c| observation.seq <= c) { continue; }
            for (e, n) in &observation.demands { *self.counts.entry(*e).or_default() += n; }
            self.seen += 1;
            self.cutoff = Some(observation.seq);
        }
        Ok(self.counts.iter().map(|(e,n)|(*e,*n as f64 / self.seen.max(1) as f64)).collect())
    }
}
