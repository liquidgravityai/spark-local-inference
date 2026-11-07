//! Resolve configured canonical identities against authenticated prepared weights.
//! Configuration selects movable storage; it never supplies addresses or payloads.
use super::*;
use crate::package::{PayloadId, protection::VerifiedPackage, schema::wire};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageConfig {
    pub version: u32,
    pub package_id: String,
    pub rank: u32,
    pub planes: Vec<Plane>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Plane {
    pub layer: usize,
    pub part: usize,
    pub weight_view: String,
    /// Axis-zero order in the authenticated tensor, expressed as canonical IDs.
    pub expert_ids: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct BoundRegion {
    pub expert: ExpertId,
    pub part: usize,
    pub allocation: usize,
    pub offset: usize,
    pub address: u64,
    pub bytes: usize,
}

impl StorageConfig {
    /// Resolution requires authenticated metadata. Normal package loading still
    /// requires publisher authorization and verifies the weight payloads.
    pub fn resolve(
        &self,
        verified: &VerifiedPackage,
        canonical: &[ExpertId],
        granularity: usize,
    ) -> Result<Vec<BoundRegion>> {
        let id = PayloadId::parse(&self.package_id).map_err(|e| Error::Invalid(e.to_string()))?;
        if id.as_bytes() != &verified.package_id() {
            return Err(Error::Invalid(
                "expert storage names a different package".into(),
            ));
        }
        if verified.deployment().is_none_or(|d| d.tensor_parallel != 1) || self.rank != 0 {
            return Err(Error::Invalid(
                "expert movement requires one local execution rank".into(),
            ));
        }
        let body = verified
            .rank(self.rank)
            .ok_or_else(|| Error::Invalid("expert storage rank is unavailable".into()))?;
        self.resolve_body(body, canonical, granularity)
    }

    fn resolve_body(
        &self,
        body: &wire::RankBody,
        canonical: &[ExpertId],
        granularity: usize,
    ) -> Result<Vec<BoundRegion>> {
        let invalid = |message: &str| Error::Invalid(message.into());
        if self.version != 1 || self.planes.is_empty() || !granularity.is_power_of_two() {
            return Err(invalid(
                "invalid expert storage configuration or mapping granularity",
            ));
        }
        let catalog = body
            .prepared_weights
            .as_ref()
            .filter(|c| c.version == 1)
            .ok_or_else(|| {
                invalid("expert movement requires a supported prepared-weight catalog")
            })?;
        let views: BTreeMap<_, _> = catalog.views.iter().map(|v| (v.name.as_str(), v)).collect();
        let eligible: BTreeSet<_> = canonical.iter().copied().collect();
        if eligible.len() != canonical.len() || eligible.is_empty() {
            return Err(invalid(
                "canonical expert catalog is empty or contains duplicates",
            ));
        }
        let mut seen = BTreeSet::new();
        let mut regions = Vec::new();
        let mut initialized = BTreeMap::<&str, Vec<(u64, u64)>>::new();
        for initializer in &body.initializers {
            if let Some(source) = initializer
                .source
                .as_ref()
                .filter(|s| s.role == wire::PayloadRole::Weights as i32)
            {
                let slice = source
                    .slice
                    .as_ref()
                    .ok_or_else(|| invalid("weight initializer has no payload slice"))?;
                let end = initializer
                    .destination_offset
                    .checked_add(slice.nbytes)
                    .ok_or_else(|| invalid("weight initializer overflows"))?;
                initialized
                    .entry(&initializer.allocation)
                    .or_default()
                    .push((initializer.destination_offset, end));
            }
        }
        for ranges in initialized.values_mut() {
            ranges.sort_unstable();
        }
        for plane in &self.planes {
            let view = views
                .get(plane.weight_view.as_str())
                .ok_or_else(|| invalid("expert weight view is unavailable"))?;
            if view.shape.is_empty()
                || view.shape[0] != plane.expert_ids.len() as u64
                || view.shape.len() != view.stride.len()
                || view.element_bytes == 0
            {
                return Err(invalid(
                    "expert IDs do not match the prepared tensor geometry",
                ));
            }
            let mut elements = 1u64;
            for (&size, &stride) in view.shape.iter().zip(&view.stride).rev() {
                if size == 0 || stride != elements {
                    return Err(invalid("movable expert planes must be contiguous"));
                }
                elements = elements
                    .checked_mul(size)
                    .ok_or_else(|| invalid("expert tensor extent overflows"))?;
            }
            if elements.checked_mul(u64::from(view.element_bytes)) != Some(view.nbytes) {
                return Err(invalid(
                    "expert tensor bytes differ from its prepared extent",
                ));
            }
            let allocation_index = usize::try_from(view.allocation)
                .map_err(|_| invalid("invalid allocation index"))?;
            let allocation = body
                .allocations
                .get(allocation_index)
                .ok_or_else(|| invalid("expert allocation is absent"))?;
            if allocation.memory != 0 || allocation.sizing.is_some() {
                return Err(invalid("expert weights require fixed local device storage"));
            }
            let base = allocation
                .address
                .ok_or_else(|| invalid("expert allocation has no fixed address"))?;
            let bytes = view.nbytes / view.shape[0];
            for (index, &id) in plane.expert_ids.iter().enumerate() {
                let expert = ExpertId {
                    layer: plane.layer,
                    expert: id,
                };
                if !eligible.contains(&expert) || !seen.insert((expert, plane.part)) {
                    return Err(invalid("unknown or duplicate canonical expert part"));
                }
                let offset = (index as u64)
                    .checked_mul(bytes)
                    .and_then(|n| view.offset.checked_add(n))
                    .ok_or_else(|| invalid("expert region offset overflows"))?;
                let end = offset
                    .checked_add(bytes)
                    .filter(|end| *end <= allocation.nbytes)
                    .ok_or_else(|| invalid("expert region exceeds its allocation"))?;
                let address = base
                    .checked_add(offset)
                    .ok_or_else(|| invalid("expert address overflows"))?;
                if address % granularity as u64 != 0 || bytes % granularity as u64 != 0 {
                    return Err(invalid(
                        "movable expert regions must own complete allocation-granularity pages",
                    ));
                }
                if allocation
                    .relocations
                    .iter()
                    .any(|r| r.offset < end && r.offset.saturating_add(128) > offset)
                {
                    return Err(invalid(
                        "expert weight region contains pointer-bearing storage",
                    ));
                }
                let mut covered = offset;
                for &(start, stop) in initialized
                    .get(allocation.name.as_str())
                    .into_iter()
                    .flatten()
                {
                    if start <= covered && covered < stop {
                        covered = stop;
                    }
                    if covered >= end {
                        break;
                    }
                }
                if covered < end {
                    return Err(invalid(
                        "expert region is not fully initialized from immutable weight payloads",
                    ));
                }
                regions.push(BoundRegion {
                    expert,
                    part: plane.part,
                    allocation: allocation_index,
                    offset: usize::try_from(offset)
                        .map_err(|_| invalid("expert offset exceeds host address space"))?,
                    address,
                    bytes: usize::try_from(bytes)
                        .map_err(|_| invalid("expert bytes exceed host address space"))?,
                });
            }
        }
        regions.sort_by_key(|r| r.address);
        if regions.windows(2).any(|p| {
            p[0].address
                .checked_add(p[0].bytes as u64)
                .is_none_or(|end| end > p[1].address)
        }) {
            return Err(invalid("movable expert regions overlap"));
        }
        Ok(regions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (StorageConfig, wire::RankBody, Vec<ExpertId>) {
        let config = StorageConfig {
            version: 1,
            package_id: "00".repeat(32),
            rank: 0,
            planes: vec![Plane {
                layer: 3,
                part: 0,
                weight_view: "packed".into(),
                expert_ids: vec![7, 2],
            }],
        };
        let body = wire::RankBody {
            allocations: vec![wire::Allocation {
                name: "weights".into(),
                address: Some(4096),
                nbytes: 256,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "weights".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    role: wire::PayloadRole::Weights as i32,
                    slice: Some(wire::PayloadSlice {
                        nbytes: 256,
                        ..Default::default()
                    }),
                }),
            }],
            prepared_weights: Some(wire::PreparedWeightCatalog {
                version: 1,
                views: vec![wire::PreparedWeightView {
                    name: "packed".into(),
                    dtype: "uint8".into(),
                    element_bytes: 1,
                    shape: vec![2, 128],
                    stride: vec![128, 1],
                    nbytes: 256,
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        (
            config,
            body,
            vec![
                ExpertId {
                    layer: 3,
                    expert: 2,
                },
                ExpertId {
                    layer: 3,
                    expert: 7,
                },
            ],
        )
    }
    #[test]
    fn canonical_ids_resolve_to_authenticated_view_offsets() {
        let (config, body, experts) = fixture();
        let regions = config.resolve_body(&body, &experts, 64).unwrap();
        assert_eq!(
            (
                regions[0].expert.expert,
                regions[0].address,
                regions[0].bytes
            ),
            (7, 4096, 128)
        );
        assert_eq!(
            (
                regions[1].expert.expert,
                regions[1].address,
                regions[1].offset
            ),
            (2, 4224, 128)
        );
    }
    #[test]
    fn unsafe_storage_bindings_are_rejected() {
        for fault in 0..10 {
            let (mut config, mut body, experts) = fixture();
            match fault {
                0 => config.planes[0].expert_ids[1] = 99,
                1 => config.planes[0].expert_ids[1] = 7,
                2 => body.allocations[0].address = Some(4100),
                3 => body.allocations[0].nbytes = 128,
                4 => {
                    body.initializers[0].source.as_mut().unwrap().role =
                        wire::PayloadRole::InitialState as i32
                }
                5 => {
                    body.initializers[0]
                        .source
                        .as_mut()
                        .unwrap()
                        .slice
                        .as_mut()
                        .unwrap()
                        .nbytes = 255
                }
                6 => body.prepared_weights.as_mut().unwrap().views[0].stride[0] = 64,
                7 => body.allocations[0]
                    .relocations
                    .push(crate::runtime_package::wire::Relocation::default()),
                8 => config.planes.push(Plane {
                    part: 1,
                    ..config.planes[0].clone()
                }),
                9 => body.allocations[0].memory = 2,
                _ => unreachable!(),
            }
            assert!(
                config.resolve_body(&body, &experts, 64).is_err(),
                "fault {fault}"
            );
        }
    }

    #[test]
    #[ignore = "requires an explicitly selected GLM package and trusted publisher profiles; CPU only"]
    fn glm_package_storage_bindings_resolve_without_gpu() {
        use crate::package::{protection, reader};
        let path = std::env::var("FLARE_PARALLAX_PACKAGE").unwrap();
        let profiles = std::env::var("FLARE_PUBLISHER_PROFILES").unwrap();
        let publishers: Vec<crate::runtime_package::Publisher> =
            serde_json::from_slice(&std::fs::read(profiles).unwrap()).unwrap();
        let metadata = reader::Metadata::open(std::path::Path::new(&path)).unwrap();
        let verified = protection::verify(
            &metadata.envelope,
            &publishers,
            protection::Selection::Deployment {
                deployment: 0,
                rank: Some(0),
            },
        )
        .unwrap();
        let body = verified.rank(0).unwrap();
        let catalog = body.prepared_weights.as_ref().unwrap();
        let mut planes = Vec::new();
        for view in &catalog.views {
            let part = if view.name.ends_with(".experts.w13") {
                0
            } else if view.name.ends_with(".experts.w2") {
                1
            } else {
                continue;
            };
            let Some(layer) = view
                .group
                .strip_prefix("layer-")
                .and_then(|v| v.parse::<usize>().ok())
            else {
                continue;
            };
            if !(3..=44).contains(&layer) {
                continue;
            }
            assert_eq!(view.shape[0], 288);
            planes.push(Plane {
                layer,
                part,
                weight_view: view.name.clone(),
                expert_ids: (0..288).collect(),
            });
        }
        assert_eq!(
            planes.len(),
            84,
            "expected the two packed weight planes of every target layer"
        );
        planes.sort_by_key(|p| (p.layer, p.part));
        let config = StorageConfig {
            version: 1,
            package_id: PayloadId::from_bytes(verified.package_id()).to_hex(),
            rank: 0,
            planes,
        };
        let canonical: Vec<_> = (3..=44)
            .flat_map(|layer| (0..288).map(move |expert| ExpertId { layer, expert }))
            .collect();
        let regions = config
            .resolve(&verified, &canonical, 2 * 1024 * 1024)
            .unwrap();
        assert_eq!(regions.len(), 24_192);
        assert_eq!(
            regions.iter().map(|r| r.bytes as u64).sum::<u64>(),
            152_202_903_552
        );
        if let Ok(output) = std::env::var("FLARE_PARALLAX_STORAGE_CONFIG") {
            std::fs::write(output, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
        }
        eprintln!(
            "resolved {} page-isolated packed weight regions for {} experts from signed package {}",
            regions.len(),
            canonical.len(),
            config.package_id
        );
    }
}
