//! Greedy batch planning against a serialized copy engine and reserved HBM bytes.
use super::*;

pub(super) fn plan(f: &Forecast, state: &VirtualResidency, p: &Profile, limit: usize) -> Vec<Movement> {
    if p.hardware.transfer_calibration.as_ref().is_some_and(|c| !c.usable()) { return vec![]; }
    let scores: BTreeMap<_, _> = f.scores.iter().copied().collect();
    let pending: BTreeSet<_> = state.pending.iter().map(|(_, e)| *e).collect();
    let mut candidates: Vec<_> = f.scores.iter().filter_map(|(e, count)| {
        if state.resident.contains(e) || pending.contains(e) { return None; }
        let copy = transfer(p, *e);
        let gain = count * p.hardware.cold_cost_ns as f64 - copy as f64;
        (gain > 0.).then_some((*e, *count, copy, gain))
    }).collect();
    candidates.sort_by(|a,b| b.3.total_cmp(&a.3).then(a.0.cmp(&b.0)));
    let mut victims: Vec<_> = state.resident.iter().map(|e| (*e, scores.get(e).copied().unwrap_or(0.))).collect();
    victims.sort_by(|a,b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut removed = BTreeSet::new();
    let mut reserved = state.reserved;
    let mut copy_until = state.copy_until.max(f.completed_ns);
    let mut moves = vec![];
    for (expert, count, copy, _) in candidates {
        if moves.len() == limit { break; }
        if copy_until.saturating_add(copy) > f.expected_demand_ns { continue; }
        let size = state.sizes[&expert];
        let victim = if reserved + size <= p.hardware.hbm_bytes { None } else {
            let Some((v, popularity)) = victims.iter().find(|(v, _)|
                !removed.contains(v) && reserved - state.sizes[v] + size <= p.hardware.hbm_bytes
            ) else { continue; };
            if (count - popularity) * p.hardware.cold_cost_ns as f64 <= copy as f64 { continue; }
            Some(*v)
        };
        if let Some(v) = victim {
            removed.insert(v);
            reserved -= state.sizes[&v];
        }
        reserved += size;
        copy_until = copy_until.saturating_add(copy);
        moves.push(Movement { promote: expert, evict: victim });
    }
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::parallax::{prediction::EvaluationConfig, prediction_tests::{profile,fixture}};
    #[test]
    fn batch_reserves_capacity_serializes_copies_and_accounts_for_eviction_losses() {
        let mut p=profile("batch");
        p.model.experts=(0..12).map(|expert|Expert{layer:0,expert}).collect();
        p.model.expert_bytes=vec![100;12];
        p.hardware.hbm_bytes=600; p.hardware.backing_bytes=1200;
        p.settings.evaluation=Some(EvaluationConfig{version:2,top_k_per_layer:32,max_promotions_per_forecast:128});
        let baseline=VirtualResidency::new(&p);
        let mut state=VirtualResidency::new(&p);
        let mut f=fixture().forecasts["recent"][0].clone();
        f.completed_ns=10; f.expected_demand_ns=610;
        f.scores=(6..12).map(|expert|(Expert{layer:0,expert},2.)).collect();
        let moves=Economic.plan(&f,&state,&p);
        assert_eq!(moves.len(),5); // A sixth copy would miss the declared deadline.
        for movement in moves { state.apply(movement,10,&p); }
        assert_eq!(state.reserved,600);
        assert_eq!(state.promotions,5);
        assert_eq!(state.evictions,5);
        assert_eq!(state.pending.iter().map(|(at,_)|*at).collect::<Vec<_>>(),vec![125,240,355,470,585]);
        let mut o=fixture().observations[0].clone();
        o.demand_ns=200; o.demands=vec![(Expert{layer:0,expert:6},2),(Expert{layer:0,expert:0},3)];
        state.demand(&o);
        assert_eq!(state.cold_avoided,2); assert_eq!(state.eviction_misses,3);
        let mut row=json!({}); super::super::apply_ledger(&mut row,&state,&baseline,123,&p);
        assert_eq!(row["cold_execution_saved_ns"],2000.);
        assert_eq!(row["eviction_execution_loss_ns"],3000.);
        assert_eq!(row["modeled_balance_ns"],-1575.);
        assert_eq!(row["predictor_runtime_total_ns"],123);
        assert!(Economic.plan(&f,&state,&p).is_empty());
    }
    #[test]
    fn rejected_movement_cannot_evict_or_overcommit_memory() {
        let mut p=profile("variable-size");
        p.model.expert_bytes=vec![100,200];
        p.hardware.backing_bytes=300;
        let mut state=VirtualResidency::new(&p);
        state.apply(Movement{promote:Expert{layer:0,expert:1},evict:Some(Expert{layer:0,expert:0})},0,&p);
        assert_eq!(state.reserved,100); assert_eq!(state.evictions,0);
        assert_eq!(state.resident.len(),1); assert!(state.pending.is_empty());
    }
}
