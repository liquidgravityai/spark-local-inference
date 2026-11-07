"""Terminal comparison of observed forecasts and explicitly modeled residency."""


def render(report):
    def number(value, scale=1, suffix=""):
        return "—" if value is None else f"{value * scale:.2f}{suffix}"

    lines = [f"PARALLAX SHADOW | {report['routing_frames']} routing frames | no physical expert movement",
             "Placement and net values are modeled estimates, NOT serving speedups."]
    if report.get("hardware"):
        lines.append(f"Cost profile: {report['hardware']['identity']}")
    if report.get("error"):
        lines.append(f"INCOMPLETE: {report['error']}")
    lines.append(f"{'Predictor / policy':24} {'N':>5} {'Raw P@8':>8} {'Coverage':>9} {'Pred ms':>8} {'Lead ms':>9} {'HBM hit':>8} {'Moves':>6} {'Net ms*':>11}")
    for name, row in sorted(report["contenders"].items()):
        lines.append(f"{name:24} {row['prediction_records']:5d} "
            f"{number(row.get('forecast_precision'), 100, '%'):>8} "
            f"{number(row['demand_coverage'], 100, '%'):>9} "
            f"{number(row['prediction_compute_p50_ns'], 1e-6):>8} "
            f"{number(row['usable_lead_p50_ns'], 1e-6):>9} "
            f"{number(row['hbm_hit_rate'], 100, '%'):>8} {row['promotion_count']:6d} "
            f"{number(row['estimated_net_route_cost_saved_ns'], 1e-6):>11}")
    lines.append("Raw P@8 includes late forecasts; coverage requires timely, unexpired forecasts.")
    lines.append("Prediction/lead are medians. *Net is an additive route-cost proxy including prediction delay.")
    for name, counts in sorted(report.get("workers", {}).items()):
        lines.append(f"{name}: busy drops={counts['dropped_busy']}, failures={counts['failures']}, timeouts={counts['timeouts']}")
    return "\n".join(lines)
