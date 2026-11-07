//! Uninformed control: one unit of forecast demand for every canonical expert.
//! Equal scores express no placement preference and are not probabilities.
use super::prediction::{Context, Expert, Predictor};
use serde_json::Value;

pub(super) fn create(config: &Value) -> Result<Box<dyn Predictor>, String> {
    if !config.is_null() && config.as_object().is_none_or(|fields| !fields.is_empty()) {
        return Err("uniform_canary accepts an empty configuration".into());
    }
    Ok(Box::new(UniformCanary))
}

struct UniformCanary;
impl Predictor for UniformCanary {
    fn version(&self) -> &'static str {
        "1"
    }

    fn predict(&mut self, context: &Context) -> Result<Vec<(Expert, f64)>, String> {
        context.history.last().ok_or("missing observation")?;
        Ok(context
            .model
            .experts
            .iter()
            .map(|expert| (*expert, 1.))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        prediction::{Observation, Registry, Selection},
        prediction_tests::profile,
    };
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn registered_canary_respects_sparse_catalog_and_ignores_routing_popularity() {
        let mut catalog = profile("canary").model;
        catalog.experts = vec![
            Expert {
                layer: 3,
                expert: 7,
            },
            Expert {
                layer: 44,
                expert: 287,
            },
        ];
        let mut context = Context {
            model: Arc::new(catalog),
            target_offset: 8,
            horizon_ns: 1000,
            available_ns: 10,
            history: vec![],
        };
        let mut provider = Registry::builtins()
            .create(&Selection {
                name: "uniform_canary".into(),
                config: json!({}),
            })
            .unwrap();
        assert_eq!(provider.version(), "1");
        assert!(provider.predict(&context).is_err());
        for (seq, count) in [(0, 1), (1, 10000)] {
            context.history.push(Arc::new(Observation {
                seq,
                demand_ns: 0,
                available_ns: 1,
                phase: "verify".into(),
                batch_size: 1,
                concurrency: 1,
                demands: vec![(context.model.experts[1], count)],
            }));
            assert_eq!(
                provider.predict(&context).unwrap(),
                vec![
                    (context.model.experts[0], 1.),
                    (context.model.experts[1], 1.)
                ]
            );
        }
    }
    #[test]
    fn canary_rejects_ignored_configuration() {
        for config in [json!({"alpha":0.5}), json!([]), json!(true), json!("empty")] {
            assert!(create(&config).is_err());
        }
        assert!(create(&Value::Null).is_ok());
    }
}
