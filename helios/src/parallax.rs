//! Routing telemetry and experimental native expert-residency mechanisms.

pub mod residency;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
    any(feature = "server", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ExpertId {
    pub layer: usize,
    pub expert: usize,
}

use crate::Result;
use crate::execution::{BufferDescriptor, Dtype, GraphKind};
use std::sync::OnceLock;
use std::time::Instant;

pub const BUFFER: &str = "parallax_routes_v1";
pub const HEADER: usize = 4;
pub const MAX_LAYERS: usize = 256;
pub const MAX_EXPERTS: usize = 4096;
pub const MAX_CELLS: usize = 1 << 20;

static ORIGIN: OnceLock<Instant> = OnceLock::new();

/// A single native-process monotonic domain, independent of GPU timestamps.
pub fn now_ns() -> u64 {
    ORIGIN
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

pub fn consumes(kind: GraphKind) -> bool {
    matches!(
        kind,
        GraphKind::Decode | GraphKind::Prefill | GraphKind::ScoredPrefill | GraphKind::Speculative
    )
}

pub fn validate_buffer(buffer: &BufferDescriptor) -> Result<()> {
    let s = &buffer.shape;
    if buffer.dtype != Dtype::Int64
        || s.len() != 3
        || s[0] != 2
        || !(1..=MAX_LAYERS).contains(&s[1])
        || !(HEADER + 1..=HEADER + MAX_EXPERTS).contains(&s[2])
        || s[1] * s[2] > MAX_CELLS
        || buffer.strides != [s[1] * s[2], s[2], 1]
    {
        return Err(
            "parallax_routes_v1 requires bounded contiguous int64[2,layers,4+experts]".into(),
        );
    }
    Ok(())
}

#[derive(Debug)]
pub enum RoutingReadout {
    Valid(RoutingSample),
    Invalid,
}

#[derive(Debug)]
#[cfg_attr(feature = "server", derive(serde::Serialize))]
pub struct LayerDemand {
    pub layer: usize,
    pub tokens: u32,
    pub top_k: u32,
    pub counts: Vec<u32>,
}

#[derive(Debug)]
#[cfg_attr(feature = "server", derive(serde::Serialize))]
pub struct RoutingSample {
    pub submission: u64,
    pub residency_generation: Option<u64>,
    pub gpu_ns: Option<u64>,
    /// Conservative demand lower bound: host dispatch before H2D or graph launch.
    pub demand_lower_bound_ns: u64,
    /// Earliest native host observation after the retained readback fence.
    pub available_ns: u64,
    pub layers: Vec<LayerDemand>,
}

pub fn decode(
    bytes: &[u8],
    layers: usize,
    experts: usize,
    submission: u64,
    dispatch_ns: u64,
    available_ns: u64,
) -> Result<RoutingSample> {
    if !(1..=MAX_LAYERS).contains(&layers)
        || !(1..=MAX_EXPERTS).contains(&experts)
        || layers * (experts + HEADER) > MAX_CELLS
        || submission == 0
        || submission > i64::MAX as u64
        || bytes.len() != layers * (experts + HEADER) * 8
        || available_ns < dispatch_ns
    {
        return Err("invalid Parallax readback geometry or timing".into());
    }
    let mut decoded = Vec::with_capacity(layers);
    for (layer, bytes) in bytes.chunks_exact((experts + HEADER) * 8).enumerate() {
        let word =
            |index: usize| i64::from_ne_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap());
        let (sequence, tokens, top_k, invalid) = (word(0), word(1), word(2), word(3));
        if sequence != submission as i64
            || !(0..=262144).contains(&tokens)
            || !(0..=experts as i64).contains(&top_k)
            || invalid != 0
        {
            return Err("stale, invalid or malformed Parallax routing row".into());
        }
        let mut counts = Vec::with_capacity(experts);
        let mut total = 0u64;
        for index in 0..experts {
            let count = u32::try_from(word(HEADER + index)).map_err(|_| "invalid routing count")?;
            total += u64::from(count);
            counts.push(count);
        }
        if total != tokens as u64 * top_k as u64 || (top_k == 0 && tokens != 0) {
            return Err("Parallax route totals differ from valid token geometry".into());
        }
        if top_k != 0 {
            decoded.push(LayerDemand {
                layer,
                tokens: tokens as u32,
                top_k: top_k as u32,
                counts,
            });
        }
    }
    Ok(RoutingSample {
        residency_generation: None,
        gpu_ns: None,
        submission,
        demand_lower_bound_ns: dispatch_ns,
        available_ns,
        layers: decoded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(words: &[i64]) -> Vec<u8> {
        words.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    #[test]
    fn canonical_layers_and_counts_survive_frame_decoding() {
        let data = bytes(&[7, 0, 0, 0, 0, 0, 7, 3, 1, 0, 2, 1]);
        let sample = decode(&data, 2, 2, 7, 100, 300).unwrap();
        assert_eq!(sample.layers.len(), 1);
        assert_eq!(sample.layers[0].layer, 1);
        assert_eq!(sample.layers[0].counts, [2, 1]);
        assert_eq!(sample.demand_lower_bound_ns, 100);
        assert_eq!(sample.available_ns, 300);
    }

    #[test]
    fn stale_frame_invalid_routes_and_wrong_totals_are_rejected() {
        for words in [
            [6, 1, 1, 0, 1, 0],
            [7, 1, 1, 1, 1, 0],
            [7, 1, 1, 0, 2, 0],
            [7, 1, 1, 0, -1, 2],
        ] {
            assert!(decode(&bytes(&words), 1, 2, 7, 100, 300).is_err());
        }
        assert!(decode(&[0; 8], 1, 2, 7, 100, 300).is_err());
    }
}
