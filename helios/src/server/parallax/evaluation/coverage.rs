//! Ranking coverage is separate from placement, cold residency and transfer feasibility.
use super::*;
#[derive(Default)]
struct Counts { predicted: u64, correct: u64, selected: u64, covered: u64 }
impl Counts {
    fn add(&mut self, chosen: &[(Expert, f64)], actual: &BTreeMap<Expert,u64>, timely: bool) {
        self.predicted += chosen.len() as u64;
        self.selected += actual.len() as u64;
        self.correct += chosen.iter().filter(|(e,_)| actual.contains_key(e)).count() as u64;
        if timely { self.covered += chosen.iter().filter_map(|(e,_)|actual.get(e)).sum::<u64>(); }
    }
    fn report(&self, demand: u64) -> Value {
        json!({"precision":ratio(self.correct,self.predicted),"recall":ratio(self.correct,self.selected),
            "timely_demand_coverage":ratio(self.covered,demand),"selected_predictions":self.predicted})
    }
}
pub(super) struct Coverage { curves: BTreeMap<usize, Counts>, budget: Counts }
impl Coverage {
    pub fn new(k: usize) -> Self {
        Self { curves: [8,16,32,64,k].into_iter().map(|n|(n,Counts::default())).collect(), budget: Counts::default() }
    }
    pub fn observe(&mut self, f: &Forecast, o: &Observation, p: &Profile) {
        let actual: BTreeMap<_,_> = o.demands.iter().copied().collect();
        let timely = f.completed_ns <= o.demand_ns && f.expires_ns > o.demand_ns;
        let mut ranked = f.scores.clone();
        ranked.retain(|(_,s)|*s>0.);
        ranked.sort_by(|a,b|b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut layer_rank = BTreeMap::<usize,usize>::new();
        let ranked: Vec<_> = ranked.into_iter().map(|(e,s)|{
            let rank = layer_rank.entry(e.layer).or_default(); *rank+=1; (e,s,*rank)
        }).collect();
        for (k,c) in &mut self.curves {
            let chosen: Vec<_> = ranked.iter().filter(|(_,_,rank)|rank<=k).map(|(e,s,_)|(*e,*s)).collect();
            c.add(&chosen,&actual,timely);
        }
        let mut bytes = 0;
        let chosen: Vec<_> = ranked.iter().filter_map(|(e,s,_)| {
            let size=p.model.expert_bytes[p.model.experts.binary_search(e).unwrap()];
            if bytes+size<=p.hardware.hbm_bytes { bytes+=size;Some((*e,*s)) } else { None }
        }).collect();
        self.budget.add(&chosen,&actual,timely);
    }
    pub fn apply(&self, report: &mut Value, k: usize, demand: u64) {
        let chosen = self.curves[&k].report(demand);
        report["forecast_precision"] = chosen["precision"].clone();
        report["forecast_recall"] = chosen["recall"].clone();
        report["demand_coverage"] = chosen["timely_demand_coverage"].clone();
        report["coverage_by_layer_k"] = Value::Object(self.curves.iter().map(|(k,c)|(k.to_string(),c.report(demand))).collect());
        report["residency_budget_ranking"] = self.budget.report(demand);
        report["ranking_scope"] = json!("Top K positive scores per layer; capacity ranking fits the HBM byte budget. Timely coverage checks arrival/expiry, not transfer completion or placement profitability.");
        report.as_object_mut().unwrap().remove("actionable_rate");
    }
}
