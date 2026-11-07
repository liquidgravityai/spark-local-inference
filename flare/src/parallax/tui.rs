//! Parallax presentation uses the same terminal toolkit and palette as Flare serving.

use super::{Failure, Shared, View, read};
use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use flare_helios::server::view::{
    BACKGROUND, CYAN, GOLD, GREEN, MUTED, PURPLE, TEXT, crossterm, ratatui,
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Row, Table, Wrap},
};
use serde_json::Value;
use std::{
    io,
    sync::atomic::Ordering,
    thread,
    time::{Duration, Instant},
};

struct Screen;
impl Drop for Screen {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stderr(), LeaveAlternateScreen, Show);
    }
}
fn refresh(view: &Shared) {
    let path = view.lock().unwrap().output.join("capture/comparison.json");
    if let Ok(report) = read(&path) {
        view.lock().unwrap().report = Some(report);
    }
}
fn value(row: &Value, key: &str, scale: f64, suffix: &str) -> String {
    row[key]
        .as_f64()
        .map(|v| format!("{:.2}{suffix}", v * scale))
        .unwrap_or_else(|| "—".into())
}
fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .title(title)
        .border_style(Style::default().fg(MUTED))
}
fn draw(frame: &mut Frame<'_>, view: &View, pane: usize, elapsed: Duration, scroll: u16) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
        area,
    );
    if area.width < 78 || area.height < 21 {
        frame.render_widget(
            Paragraph::new(format!(
                "FLARE · PARALLAX\n{}\nResize to at least 78 × 21.\nq: drain and exit\n{}",
                view.phase,
                view.error.as_deref().unwrap_or("")
            ))
            .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(4),
        Constraint::Min(9),
        Constraint::Length(4),
        Constraint::Length(2),
    ])
    .split(area);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    " FLARE / PARALLAX ",
                    Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" SHADOW ", Style::default().fg(PURPLE)),
                Span::raw(format!(
                    "  {:.0}s  ·  {}",
                    elapsed.as_secs_f64(),
                    view.phase
                )),
            ]),
            Line::from(format!(
                " {}",
                if view.model.is_empty() {
                    "Loading package identity"
                } else {
                    &view.model
                }
            )),
        ]),
        areas[0],
    );
    let frames = view
        .report
        .as_ref()
        .map(|r| r["routing_frames"].to_string())
        .unwrap_or_else(|| "0".into());
    frame.render_widget(
        Paragraph::new(format!(
            "Frames evaluated: {frames}    Responses: {}    Physical moves: 0\nGPU: {}",
            view.responses.len(),
            view.gpu
        ))
        .block(panel(" Session ")),
        areas[1],
    );
    if let Some(error) = view
        .error
        .as_deref()
        .or_else(|| view.report.as_ref().and_then(|r| r["error"].as_str()))
    {
        frame.render_widget(
            Paragraph::new(error)
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(GOLD))
                .block(panel(" Capture failed · no completed result ")),
            areas[2],
        );
    } else if pane == 2 {
        let text = view
            .responses
            .iter()
            .enumerate()
            .map(|(i, s)| format!("Response {}\n{s}\n", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        frame.render_widget(
            Paragraph::new(if text.is_empty() {
                "Waiting for generated responses".into()
            } else {
                text
            })
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0))
            .block(panel(
                " Generated responses · ↑ ↓ scroll · saved in responses.json ",
            )),
            areas[2],
        );
    } else if let Some(report) = &view.report {
        let rows = report["contenders"]
            .as_object()
            .map(|r| {
                r.iter()
                    .map(|(name, row)| {
                        let cells = if pane == 0 {
                            vec![
                                name.clone(),
                                row["prediction_records"].to_string(),
                                value(row, "forecast_precision", 100., "%"),
                                value(row, "demand_coverage", 100., "%"),
                                value(row, "usable_lead_p50_ns", 1e-6, ""),
                                value(row, "hbm_hit_rate", 100., "%"),
                                row["promotion_count"].to_string(),
                                value(row, if report["evaluation"]["version"]==2 { "modeled_balance_ns" } else { "estimated_net_route_cost_saved_ns" }, 1e-6, ""),
                            ]
                        } else {
                            vec![
                                name.clone(),
                                value(row, "prediction_compute_p50_ns", 1e-6, ""),
                                value(row, "prediction_compute_p95_ns", 1e-6, ""),
                                value(row, "prediction_queue_p50_ns", 1e-6, ""),
                                value(row, "prediction_publication_p50_ns", 1e-6, ""),
                                value(row, "actionable_rate", 100., "%"),
                                row["late_predictions"].to_string(),
                                row["expired_predictions"].to_string(),
                            ]
                        };
                        Row::new(cells).style(Style::default().fg(TEXT))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let header = if pane == 0 {
            vec![
                "Predictor / policy",
                "N",
                if report["evaluation"]["version"]==2 { "P/layer" } else { "P@8" },
                "Coverage",
                "Lead ms",
                "HBM*",
                "Moves*",
                if report["evaluation"]["version"]==2 { "Balance*" } else { "Legacy ms*" },
            ]
        } else {
            vec![
                "Predictor / policy",
                "P50 ms",
                "P95 ms",
                "Queue ms",
                "Recv ms",
                "Actionable",
                "Late",
                "Expired",
            ]
        };
        let widths = [
            Constraint::Min(19),
            Constraint::Length(6),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(7),
            Constraint::Length(10),
        ];
        frame.render_widget(
            Table::new(rows, widths)
                .column_spacing(1)
                .header(Row::new(header).style(Style::default().fg(CYAN)))
                .block(panel(if pane == 0 {
                    " Prediction and modeled residency "
                } else {
                    " Predictor timing · measured wall latency "
                })),
            areas[2],
        );
    } else {
        let log = std::fs::read_to_string(view.output.join("server.log")).unwrap_or_default();
        let tail = log
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        frame.render_widget(
            Paragraph::new(format!("{}\n{tail}", view.phase))
                .wrap(Wrap { trim: false })
                .block(panel(" Native startup ")),
            areas[2],
        );
    }
    let profile = view
        .report
        .as_ref()
        .and_then(|r| r["hardware"]["identity"].as_str())
        .unwrap_or("Cost model will be shown when the native observatory starts");
    let counts = view
        .report
        .as_ref()
        .and_then(|r| r["workers"].as_object())
        .map(|workers| {
            workers
                .iter()
                .map(|(name, w)| {
                    format!(
                        "{name}: {} busy drops / {} failures / {} timeouts",
                        w["dropped_busy"], w["failures"], w["timeouts"]
                    )
                })
                .collect::<Vec<_>>()
                .join("   ")
        })
        .unwrap_or_default();
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "* Virtual placement; additive route-cost estimates, not serving speedups.",
                Style::default().fg(GOLD),
            )),
            Line::from(profile),
            Line::from(counts),
        ])
        .wrap(Wrap { trim: false }),
        areas[3],
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                if view.done {
                    " Tab / ← →: comparison · timing · responses    q / Enter: close"
                } else {
                    " Tab / ← →: comparison · timing · responses    q: stop requests and drain"
                },
                Style::default().fg(if view.done { GREEN } else { MUTED }),
            )),
            Line::from(format!(" {}", view.output.display())),
        ]),
        areas[4],
    );
}

pub fn run(view: &Shared) -> Result<(), Failure> {
    let _screen = Screen;
    enable_raw_mode()?;
    execute!(io::stderr(), EnterAlternateScreen, Hide)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stderr()))?;
    let start = Instant::now();
    let mut pane = 0;
    let mut scroll = 0u16;
    let mut quitting = false;
    loop {
        refresh(view);
        {
            let state = view.lock().unwrap();
            terminal.draw(|frame| draw(frame, &state, pane, start.elapsed(), scroll))?;
            if state.done && (quitting || crate::STOP.load(Ordering::Relaxed)) {
                break;
            }
        }
        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Up if pane == 2 => scroll = scroll.saturating_sub(1),
                    KeyCode::Down if pane == 2 => scroll = scroll.saturating_add(1),
                    KeyCode::Tab | KeyCode::Right => pane = (pane + 1) % 3,
                    KeyCode::Left => pane = (pane + 2) % 3,
                    KeyCode::Char('q') | KeyCode::Esc | KeyCode::Enter => {
                        if view.lock().unwrap().done {
                            break;
                        }
                        quitting = true;
                        crate::STOP.store(true, Ordering::Relaxed);
                    }
                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                        quitting = true;
                        crate::STOP.store(true, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}
pub fn plain(view: &Shared) -> Result<(), Failure> {
    let mut previous = String::new();
    loop {
        refresh(view);
        {
            let state = view.lock().unwrap();
            if state.phase != previous {
                eprintln!("Parallax: {}", state.phase);
                previous = state.phase.clone();
            }
            if state.done {
                if let Some(error) = &state.error {
                    eprintln!("{error}");
                }
                break;
            }
        }
        thread::sleep(Duration::from_millis(200));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    #[test]
    fn dashboard_keeps_failure_and_no_movement_scope_visible() {
        let mut terminal = Terminal::new(TestBackend::new(110, 28)).unwrap();
        let state = View {
            phase: "Incomplete".into(),
            error: Some("routing retention gap at sequence 353".into()),
            ..View::default()
        };
        terminal
            .draw(|f| draw(f, &state, 0, Duration::from_secs(1), 0))
            .unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(text.contains("routing retention gap"));
        assert!(text.contains("Physical moves: 0"));
        assert!(text.contains("not serving speedups"));
        let mut small = Terminal::new(TestBackend::new(60, 15)).unwrap();
        small
            .draw(|f| draw(f, &state, 0, Duration::ZERO, 0))
            .unwrap();
        assert!(
            small
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect::<String>()
                .contains("Resize")
        );
    }
}
