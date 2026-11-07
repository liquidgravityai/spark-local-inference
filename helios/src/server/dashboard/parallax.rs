//! The serving dashboard reads observation results; it never executes predictors.
use super::*;
use ratatui::widgets::{Row, Table, Wrap};
use serde_json::Value;
fn number(row: &Value, key: &str, scale: f64) -> String {
    row[key]
        .as_f64()
        .map(|v| format!("{:.2}", v * scale))
        .unwrap_or_else(|| "—".into())
}
fn mode(report: &Value) -> &'static str {
    if report["active_policy"].is_object() { "ACTIVE + SHADOW" }
    else if report["physical_residency"].is_object() { "STATIC + SHADOW" }
    else { "SHADOW" }
}
fn physical_status(report: &Value) -> String {
    let state = &report["physical_residency"];
    if let Some(error) = state["error"].as_str() { return format!("Physical storage: {error}"); }
    if state.is_object() {
        let control = if report["active_policy"]["failure"].is_string() { "static fallback" }
            else { report["active_policy"]["predictor"].as_str().unwrap_or("initial placement") };
        format!("Actual weights: {} HBM / {} Grace experts · {control} · generation {}",
            state["hbm_experts"],state["grace_experts"],state["generation"])
    } else {
        "No physical expert movement. Placement costs are modeled, not serving speedups.".into()
    }
}
pub(super) fn render(frame: &mut Frame, area: Rect, report: Value) {
    if report["actual_ledger"].is_object() { render_actual(frame, area, report); return; }
    if report["evaluation"]["version"] == 2 { render_ledger(frame, area, report); return; }
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(6),
        Constraint::Length(6),
    ])
    .split(area);
    let enabled = report.get("mode").is_some();
    let title = if enabled {
        format!(
            "PARALLAX · {} · {} routing frames · in-process predictors",
            mode(&report), report["routing_frames"]
        )
    } else {
        "PARALLAX · disabled".into()
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(title, Style::default().fg(CYAN))),
            Line::from(
                physical_status(&report),
            ),
        ]),
        rows[0],
    );
    if !enabled {
        frame.render_widget(Paragraph::new("Enable with --parallax-routing --parallax-profile PATH.\nProviders are compiled into Flare and selected by the runtime profile.").wrap(Wrap{trim:false}),rows[1]);
        return;
    }
    if let Some(error) = report["error"].as_str() {
        frame.render_widget(
            Paragraph::new(error)
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(GOLD)),
            rows[1],
        );
        return;
    }
    let entries = report["contenders"]
        .as_object()
        .map(|r| {
            r.iter()
                .map(|(name, v)| {
                    Row::new(vec![
                        name.clone(),
                        v["prediction_records"].to_string(),
                        number(v, "forecast_precision", 100.),
                        number(v, "demand_coverage", 100.),
                        number(v, "prediction_compute_p50_ns", 1e-6),
                        number(v, "usable_lead_p50_ns", 1e-6),
                        number(v, "hbm_hit_rate", 100.),
                        number(v, "estimated_net_route_cost_saved_ns", 1e-6),
                        format!("{}/{}", v["resident_experts"], v["cold_experts"]),
                        format!("{}/{}", v["promotion_count"], v["eviction_count"]),
                    ])
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let widths = [
        Constraint::Min(18),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Length(8),
        Constraint::Length(11),
        Constraint::Length(11),
        Constraint::Length(11),
    ];
    frame.render_widget(
        Table::new(entries, widths).header(
            Row::new([
                "Predictor / policy",
                "N",
                "P@8 %",
                "Coverage %",
                "Pred ms",
                "Lead ms",
                "HBM %*",
                "Net ms*",
                "Hot/cold*",
                "Move/evict*",
            ])
            .style(Style::default().fg(CYAN)),
        ),
        rows[1],
    );
    let workers = report["workers"]
        .as_object()
        .map(|r| {
            r.iter()
                .map(|(name, c)| {
                    format!(
                        "{name}: {} busy drops, {} failures, {} timeouts",
                        c["dropped_busy"], c["failures"], c["timeouts"]
                    )
                })
                .collect::<Vec<_>>()
                .join(" · ")
        })
        .unwrap_or_default();
    let calibration = &report["hardware"]["transfer_calibration"];
    let costs = if calibration.is_null() {
        "Transfer and cold-execution costs supplied by profile; not calibrated here.".into()
    } else if calibration["failure"].is_string() {
        "Transfer calibration unavailable; observing without placement recommendations.".into()
    } else {
        format!("Pinned copies: H→D {} GB/s · D→H {} GB/s · cold-execution penalty supplied by profile.",
            number(&calibration["host_to_device"], "bandwidth_bytes_per_ns", 1.),
            number(&calibration["device_to_host"], "bandwidth_bytes_per_ns", 1.))
    };
    frame.render_widget(Paragraph::new(format!("{}\n{}\nPrediction window: {} frames · residency: lifetime · predictors continue after trace retention fills.\n{}\n* Contender placements and additive costs are simulated. Frames are correlated.",
        report["hardware"]["identity"].as_str().unwrap_or(""), workers, report["evaluation_window_frames"], costs))
        .wrap(Wrap{trim:false}).style(Style::default().fg(MUTED)),rows[2]);
}

fn render_actual(frame: &mut Frame, area: Rect, report: Value) {
    let ledger=&report["actual_ledger"];
    let areas=Layout::vertical([Constraint::Length(3),Constraint::Length(9),Constraint::Min(4),Constraint::Length(5)]).split(area);
    frame.render_widget(Paragraph::new(vec![
        Line::from(Span::styled(format!("PARALLAX · {} · ACTUAL RESIDENCY LEDGER",mode(&report)),Style::default().fg(CYAN))),
        Line::from(physical_status(&report)),
        Line::from(format!("Controller: {} · {}",report["active_policy"]["predictor"].as_str().unwrap_or("static"),
            report["active_policy"]["policy"].as_str().unwrap_or("fixed initial placement"))),
    ]),areas[0]);
    let ms=|key| ledger[key].as_f64().map(|v|format!("{:+.2} ms",v/1e6)).unwrap_or("learning".into());
    let rows=vec![
        Row::new(vec!["Copy GPU time".into(),ms("copy_gpu_ns"),format!("{} timed transactions; startup excluded",ledger["timed_copy_transactions"])]),
        Row::new(vec!["Publication CPU time".into(),ms("publication_host_ns"),"Actual mapping calls".into()]),
        Row::new(vec!["Submission gate closed".into(),ms("submission_gate_closed_ns"),"Includes reader drain; not GPU idle time".into()]),
        Row::new(vec!["Cold touches avoided/added".into(),format!("{}/{}",ledger["cold_expert_touches_avoided_vs_static"],ledger["eviction_expert_touches_vs_static"]),"Against initial placement; one per expert/graph".into()]),
        Row::new(vec!["Promotions used/unused".into(),format!("{}/{}",ledger["promotions_used"],ledger["promotions_unused_before_eviction"]),format!("{} still awaiting use",ledger["promotions_awaiting_use"])]),
        Row::new(vec!["Execution saving (estimate)".into(),ms("estimated_execution_saved_ns"),format!("Prior timing fit on {}/{} frames",ledger["modeled_frames"],ledger["observed_frames"])]),
        Row::new(vec!["Accounting balance (est.)".into(),ms("estimated_balance_ns"),"Saving minus full copy + publication charge".into()]),
    ];
    frame.render_widget(Table::new(rows,[Constraint::Length(28),Constraint::Length(18),Constraint::Min(12)])
        .header(Row::new(["Measurement","Value","Scope"]).style(Style::default().fg(CYAN))),areas[1]);
    let workers=report["workers"].as_object().map(|workers|workers.iter().map(|(name,w)| {
        let row=&report["contenders"][format!("{name}/static")];
        Row::new(vec![name.clone(),number(row,"demand_coverage",100.),number(row,"usable_lead_p50_ns",1e-6),
            number(row,"prediction_compute_p50_ns",1e-6),w["dropped_busy"].to_string()])
    }).collect::<Vec<_>>()).unwrap_or_default();
    frame.render_widget(Table::new(workers,[Constraint::Min(16),Constraint::Length(12),Constraint::Length(12),Constraint::Length(12),Constraint::Length(8)])
        .header(Row::new(["Predictor","Coverage %","Lead ms","Compute ms","Drops"]).style(Style::default().fg(CYAN))),areas[2]);
    let error=ledger["error"].as_str().or(report["error"].as_str()).unwrap_or("Ledger complete for observed frames.");
    frame.render_widget(Paragraph::new(format!("{error}\n{} frames overlapped movement; excluded from cost fitting.\nCost model: matching graph/batch/rows; cold and total distinct weight bytes.\nBalance is observational, not measured serving improvement.\nCopy interference and predictor CPU overhead are not attributed.",ledger["movement_overlap_frames"]))
        .style(Style::default().fg(MUTED)).wrap(Wrap{trim:false}),areas[3]);
}

fn render_ledger(frame: &mut Frame, area: Rect, report: Value) {
    use ratatui::widgets::Cell;
    let provider_count=report["workers"].as_object().map_or(0,|w|w.len());
    let areas=Layout::vertical([Constraint::Length(3),Constraint::Min(7),Constraint::Length(provider_count as u16+4)]).split(area);
    let copies=if report["hardware"]["transfer_calibration"]["host_to_device"].is_object() { "measured" } else { "assumed" };
    let penalty=report["hardware"]["cold_cost_ns"].as_f64().unwrap_or(0.)/1000.;
    frame.render_widget(Paragraph::new(vec![
        Line::from(Span::styled(format!("PARALLAX · {} · {} routing frames",mode(&report),report["routing_frames"]),Style::default().fg(CYAN))),
        Line::from(format!("Copies {copies}; cold penalty assumed: {penalty:.3} µs/use. Contenders remain shadow.")),
        Line::from(physical_status(&report)),
    ]),areas[0]);
    if let Some(error)=report["error"].as_str() {
        frame.render_widget(Paragraph::new(error).wrap(Wrap{trim:false}).style(Style::default().fg(GOLD)),areas[1]);
        return;
    }
    let entries=report["contenders"].as_object().map(|rows|rows.iter().map(|(name,row)| {
        let balance=row["modeled_balance_ns"].as_f64().unwrap_or(0.)/1e6;
        Row::new(vec![
            Cell::from(name.clone()),
            Cell::from(number(row,"cold_execution_saved_ns",1e-6)),
            Cell::from(number(row,"eviction_execution_loss_ns",1e-6)),
            Cell::from(number(row,"modeled_movement_cost_ns",1e-6)),
            Cell::from(format!("{balance:+.2}")).style(Style::default().fg(if balance>0. {GREEN} else if balance<0. {GOLD} else {MUTED})),
            Cell::from(format!("{}/{}",row["promotion_count"],row["eviction_count"])),
            Cell::from(number(row,"hbm_hit_rate",100.)),
        ])
    }).collect::<Vec<_>>()).unwrap_or_default();
    frame.render_widget(Table::new(entries,[Constraint::Min(18),Constraint::Length(9),Constraint::Length(9),Constraint::Length(8),Constraint::Length(10),Constraint::Length(8),Constraint::Length(8)])
        .header(Row::new(["Predictor / policy","Saved ms","Evict ms","Copy ms","Balance*","Moves","HBM hit%" ]).style(Style::default().fg(CYAN))),areas[1]);
    let mut lines=vec![format!("Quality: top {}/layer · recent {} frames. Ledger: server lifetime.",report["evaluation"]["top_k_per_layer"],report["evaluation_window_frames"])];
    if let Some(workers)=report["workers"].as_object() {
        for (name,worker) in workers {
            let row=&report["contenders"][format!("{name}/static")];
            lines.push(format!("{name}: covered {}% · ahead {} ms · predict {} ms · {} drops",
                number(row,"demand_coverage",100.),number(row,"usable_lead_p50_ns",1e-6),number(row,"prediction_compute_p50_ns",1e-6),worker["dropped_busy"]));
        }
    }
    lines.push(report["hardware"]["identity"].as_str().unwrap_or("Unidentified cost model").into());
    lines.push("Moves = promotions/evictions. Saved/lost accesses are relative to static.".into());
    lines.push("* Balance = Saved − Evict − Copy; excludes contention. Not throughput gains.".into());
    frame.render_widget(Paragraph::new(lines.join("\n")).style(Style::default().fg(MUTED)),areas[2]);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_ledger_never_displays_the_synthetic_shadow_balance() {
        use ratatui::{Terminal,backend::TestBackend};
        let report=serde_json::json!({"actual_ledger":{"observed_frames":10,"modeled_frames":0,
            "copy_gpu_ns":1000000,"publication_host_ns":2000000,"submission_gate_closed_ns":3000000},
            "physical_residency":{"hbm_experts":8,"grace_experts":4,"generation":2},
            "active_policy":{"predictor":"ewma","policy":"predictor_rank"},"workers":{},
            "contenders":{"ewma/economic":{"modeled_balance_ns":99999999999.}}});
        for width in [80,120] {
            let mut terminal=Terminal::new(TestBackend::new(width,24)).unwrap();
            terminal.draw(|frame|render(frame,frame.area(),report.clone())).unwrap();
            let text=terminal.backend().buffer().content.iter().map(|c|c.symbol()).collect::<String>();
            for label in ["ACTUAL RESIDENCY LEDGER","Copy GPU time","learning","predictor_rank"] {assert!(text.contains(label),"{label}");}
            assert!(!text.contains("99999"));assert!(!text.contains("cold penalty assumed"));
        }
    }
    #[test]
    fn ledger_labels_and_signed_balances_fit_normal_terminal_widths() {
        use ratatui::{Terminal,backend::TestBackend};
        let report=serde_json::json!({"evaluation":{"version":2,"top_k_per_layer":32},"routing_frames":1000,
            "evaluation_window_frames":100,"hardware":{"cold_cost_ns":1000},"workers":{"ewma":{"dropped_busy":0}},
            "contenders":{"ewma/economic":{"cold_execution_saved_ns":10000000.,"eviction_execution_loss_ns":2000000.,
                "modeled_movement_cost_ns":1000000,"modeled_balance_ns":7000000.,"promotion_count":3,"eviction_count":3,"hbm_hit_rate":0.6}}});
        for width in [80,120] {
            let mut terminal=Terminal::new(TestBackend::new(width,24)).unwrap();
            terminal.draw(|frame|render(frame,frame.area(),report.clone())).unwrap();
            let text=terminal.backend().buffer().content.iter().map(|c|c.symbol()).collect::<String>();
            for label in ["Saved ms","Evict ms","Copy ms","Balance*","+7.00","cold penalty assumed"] { assert!(text.contains(label),"missing {label} at width {width}"); }
            assert!(!text.contains("Net ms"));
        }
    }
    #[test]
    fn physical_storage_is_distinct_from_shadow_ledger() {
        let mut report=serde_json::json!({"physical_residency":{"hbm_experts":8,"grace_experts":4,"generation":12}});
        assert_eq!(mode(&report),"STATIC + SHADOW");
        assert!(physical_status(&report).contains("8 HBM / 4 Grace"));
        report["active_policy"]=serde_json::json!({"predictor":"ewma","failure":null});
        assert_eq!(mode(&report),"ACTIVE + SHADOW");
        assert!(physical_status(&report).contains("ewma"));
        report["active_policy"]["failure"]=serde_json::json!("provider failed");
        assert!(physical_status(&report).contains("static fallback"));
    }

}
