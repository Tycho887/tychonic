#!/usr/bin/env python3
"""Summarize phase_pass CSVs and save publication/export-ready plots.

Requires numpy and matplotlib. Usage: python3 tools/scripts/plot_phase_pass.py OUTPUT
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key])


def save(fig, output, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def compare_drift(output, comparison_path, config, rows):
    """Compare paired errors while holding all inputs except OU amplitude fixed."""
    other = json.loads((comparison_path / "config.json").read_text())
    for key in ("seed", "trials", "carrier_hz", "baseline_enu_m", "cadence_seconds", "tau_hours"):
        if config[key] != other[key]:
            raise ValueError(f"Paired comparison has different {key}")
    if (output / "priors.csv").read_bytes() != (comparison_path / "priors.csv").read_bytes():
        raise ValueError("Paired comparison has different initial priors")
    other_rows = read_csv(comparison_path / "results.csv")
    rng = np.random.default_rng(config["seed"])
    comparison = []
    for level in config["phase_noise_deg"]:
        for solver in ("batch", "ekf"):
            def by_trial(source):
                return {int(r["trial"]): r for r in source if r["solver"] == solver
                        and number(r, "phase_sigma_deg") == level and r["status"] == "ok"}
            nominal, negligible = by_trial(rows), by_trial(other_rows)
            paired = sorted(nominal.keys() & negligible.keys())
            if not paired:
                raise ValueError("No successful paired trials")
            item = {"phase_sigma_deg": level, "solver": solver, "paired_trials": len(paired)}
            for key in ("position_m", "velocity_m_s"):
                a = np.array([number(nominal[i], key) for i in paired])
                b = np.array([number(negligible[i], key) for i in paired])
                indices = rng.integers(len(paired), size=(2000, len(paired)))
                delta = np.sqrt(np.mean(a[indices]**2, axis=1)) - np.sqrt(np.mean(b[indices]**2, axis=1))
                item.update({f"nominal_{key}_rmse": np.sqrt(np.mean(a**2)),
                             f"negligible_{key}_rmse": np.sqrt(np.mean(b**2)),
                             f"difference_{key}_ci_low": np.percentile(delta, 2.5),
                             f"difference_{key}_ci_high": np.percentile(delta, 97.5)})
            comparison.append(item)
    with (output / "drift_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in zip(axes, ("position_m", "velocity_m_s"), ("Position RMSE (m)", "Velocity RMSE (m/s)")):
        for solver, color in (("batch", "#0072B2"), ("ekf", "#D55E00")):
            items = [r for r in comparison if r["solver"] == solver]
            for prefix, style, amplitude in (("nominal", "-", config["stationary_drift_deg"]),
                                             ("negligible", "--", other["stationary_drift_deg"])):
                ax.plot(config["phase_noise_deg"], [r[f"{prefix}_{key}_rmse"] for r in items], marker="o",
                        linestyle=style, color=color, label=f"{solver.upper()}, OU σ={amplitude:g}°")
        ax.set_xscale("log")
        ax.set_xticks(config["phase_noise_deg"], [f"{x:g}°" for x in config["phase_noise_deg"]])
        ax.set_xlabel("White phase measurement standard deviation")
        ax.set_ylabel(label)
        ax.legend()
    fig.suptitle("Paired drift-amplitude comparison; identical priors and standardized noise")
    fig.tight_layout()
    save(fig, output, "drift_comparison")
    lines = ["", "## Effect of offset drift", "",
             f"A paired {config['trials']}-trial run reduces stationary OU amplitude from "
             f"{config['stationary_drift_deg']:g}° to {other['stationary_drift_deg']:g}°, "
             "holding the correlation time, geometry, priors, hardware offsets and standardized noise fixed.", "",
             "| Phase 1σ | Batch RMSE with drift (m) | With negligible drift (m) | Paired 95% CI for RMSE increase (m) |",
             "|---|---:|---:|---:|"]
    for r in comparison:
        if r["solver"] == "batch":
            lines.append(f"| {r['phase_sigma_deg']:g}° | {r['nominal_position_m_rmse']:.2f} | "
                         f"{r['negligible_position_m_rmse']:.2f} | "
                         f"{r['difference_position_m_ci_low']:.2f} to {r['difference_position_m_ci_high']:.2f} |")
    return [*lines, "", "Drift limits the benefit of the smallest white phase uncertainty in this pass. "
            "At larger uncertainties its effect is smaller. Even negligible drift leaves uncertainty "
            "from Doppler noise, unknown offsets and single-pass geometry.", "",
            "![Drift comparison](drift_comparison.png)", ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare-negligible-drift", type=Path)
    args = parser.parse_args()
    output = args.output
    config = json.loads((output / "config.json").read_text())
    solver_config = json.loads((output / "solver_settings.json").read_text()) if (output / "solver_settings.json").exists() else {}
    representative_trial = solver_config.get("start_trial", 0)
    rows = read_csv(output / "results.csv")
    histories = read_csv(output / "representative.csv")
    levels = [0.0, *config["phase_noise_deg"]]
    labels = ["Doppler only", *[f"{x:g}°" for x in levels[1:]]]
    solvers = ["batch", "ekf"]
    colors = {"batch": "#0072B2", "ekf": "#D55E00", "rts": "#009E73"}
    summary = []
    rng = np.random.default_rng(config["seed"])
    for level in levels:
        for solver in solvers:
            selected = [r for r in rows if number(r, "phase_sigma_deg") == level and r["solver"] == solver]
            good = [r for r in selected if r["status"] == "ok"]
            item = {"phase_sigma_deg": level, "solver": solver,
                    "successes": len(good), "failures": len(selected) - len(good),
                    "verified_linearized_tolerance": sum(r.get("convergence") == "verified_linearized_tolerance" for r in good)}
            for key in ("position_m", "velocity_m_s"):
                values = np.array([number(r, key) for r in good])
                if len(values):
                    samples = rng.choice(values, size=(2000, len(values)), replace=True)
                    boot = np.sqrt(np.mean(samples**2, axis=1))
                    item.update({f"{key}_rmse": np.sqrt(np.mean(values**2)),
                                 f"{key}_median": np.median(values), f"{key}_p95": np.percentile(values, 95),
                                 f"{key}_rmse_ci_low": np.percentile(boot, 2.5),
                                 f"{key}_rmse_ci_high": np.percentile(boot, 97.5)})
                else:
                    for suffix in ("rmse", "median", "p95", "rmse_ci_low", "rmse_ci_high"):
                        item[f"{key}_{suffix}"] = float("nan")
            item["mean_nees"] = np.mean([number(r, "nees") for r in good]) if good else float("nan")
            item["coverage_95"] = np.mean([number(r, "nees") <= 12.591587 for r in good]) if good else float("nan")
            item["mean_nis"] = np.mean([number(r, "mean_nis") for r in good]) if good and solver == "ekf" else ""
            summary.append(item)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, unit in zip(axes, ("position_m", "velocity_m_s"), ("Position error (m)", "Velocity error (m/s)")):
        for solver in solvers:
            items = [r for r in summary if r["solver"] == solver]
            y = np.array([r[f"{key}_rmse"] for r in items])
            low = np.array([r[f"{key}_rmse_ci_low"] for r in items])
            high = np.array([r[f"{key}_rmse_ci_high"] for r in items])
            offset = -.06 if solver == "batch" else .06
            ax.errorbar(np.arange(len(levels)) + offset, y, yerr=[y-low, high-y],
                        marker="o", capsize=3, label=solver.upper(), color=colors[solver])
        ax.set_xticks(range(len(levels)), labels)
        ax.set_yscale("log")
        ax.set_ylabel(unit)
        ax.set_xlabel("Phase measurement standard deviation")
        ax.legend()
    fig.suptitle(f"Final-epoch RMSE; {config['trials']} paired trials (95% bootstrap intervals)")
    fig.tight_layout()
    save(fig, output, "accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, unit in zip(axes, ("position_m", "velocity_m_s"), ("Position error (m)", "Velocity error (m/s)")):
        for solver in solvers:
            groups = [[number(r, key) for r in rows if r["solver"] == solver and r["status"] == "ok"
                       and number(r, "phase_sigma_deg") == level] for level in levels]
            positions = np.arange(len(levels)) + (-.17 if solver == "batch" else .17)
            boxes = ax.boxplot(groups, positions=positions, widths=.28, patch_artist=True,
                               manage_ticks=False, showfliers=True)
            for box in boxes["boxes"]:
                box.set(facecolor=colors[solver], alpha=.5)
            ax.plot([], [], color=colors[solver], label=solver.upper(), linewidth=6)
        ax.set_xticks(range(len(levels)), labels)
        ax.set_yscale("log")
        ax.set_ylabel(unit)
        ax.legend()
    fig.suptitle("Final-epoch error distributions (quartiles, 1.5 IQR whiskers, individual outliers)")
    fig.tight_layout()
    save(fig, output, "distributions")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, level in zip(axes.flat, levels[1:]):
        for solver in ("batch", "ekf", "rts"):
            selected = sorted([r for r in histories if r["solver"] == solver
                               and number(r, "phase_sigma_deg") == level], key=lambda r: number(r, "seconds"))
            t = [number(r, "seconds") for r in selected]
            ax.plot(t, [number(r, "position_m") for r in selected], label=solver.upper(), color=colors[solver])
            ax.plot(t, [number(r, "position_sigma_m") for r in selected], color=colors[solver], linestyle=":", alpha=.7)
        ax.set_title(f"Phase σ = {level:g}°")
        ax.set_yscale("log")
        ax.set_ylabel("Position error (m)")
        ax.set_xlabel("Seconds from first observation")
        ax.legend()
    fig.suptitle(f"Representative trial {representative_trial}: orbit errors (solid), √trace(position covariance) (dotted)")
    fig.tight_layout()
    save(fig, output, "pass_errors")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, level in zip(axes.flat, levels[1:]):
        selected = [r for r in histories if r["solver"] == "batch" and number(r, "phase_sigma_deg") == level]
        t = [number(r, "seconds") for r in selected]
        ax.plot(t, np.rad2deg([number(r, "true_phase_offset_rad") for r in selected]), color="black", label="Truth")
        for solver in ("batch", "ekf", "rts"):
            selected = [r for r in histories if r["solver"] == solver and number(r, "phase_sigma_deg") == level]
            ax.plot([number(r, "seconds") for r in selected],
                    np.rad2deg([number(r, "estimated_phase_offset_rad") for r in selected]),
                    label=solver.upper(), color=colors[solver])
        ax.set_title(f"Phase σ = {level:g}°")
        ax.set_xlabel("Seconds from first observation")
        ax.set_ylabel("Hardware mean + OU drift (degrees)")
        ax.legend()
    fig.suptitle(f"Representative trial {representative_trial}: phase-offset recovery")
    fig.tight_layout()
    save(fig, output, "phase_offset")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for solver in solvers:
        items = [r for r in summary if r["solver"] == solver]
        axes[0].plot(range(len(levels)), [r["mean_nees"] for r in items], "o-", label=solver.upper(), color=colors[solver])
        axes[1].plot(range(len(levels)), [r["coverage_95"] for r in items], "o-", color=colors[solver])
    items = [r for r in summary if r["solver"] == "ekf"]
    axes[2].plot(range(len(levels)), [r["mean_nis"] for r in items], "o-", color=colors["ekf"])
    axes[0].axhline(6, color="black", linestyle="--")
    axes[1].axhline(.95, color="black", linestyle="--")
    axes[2].plot(range(len(levels)), [1, 2, 2, 2, 2], color="black", linestyle="--")
    for ax, label in zip(axes, ("Mean orbital NEES (expected 6)", "95% ellipsoid coverage", "Mean EKF NIS (expected dimension)")):
        ax.set_xticks(range(len(levels)), labels, rotation=25, ha="right")
        ax.set_ylabel(label)
    axes[0].legend()
    fig.tight_layout()
    save(fig, output, "consistency")

    lines = ["# Phase-difference pass study", "",
             f"{config['trials']} paired trials, {config['retained_samples']} observations over "
             f"{config['retained_duration_seconds']:g} s above {config['min_elevation_deg']:g}° elevation. "
             f"Carrier {config['carrier_hz']/1e9:g} GHz; baseline {config['baseline_enu_m'][0]:g} m east–west.", "",
             "| Phase 1σ | Batch position RMSE (m) | EKF position RMSE (m) | Batch velocity RMSE (m/s) | EKF velocity RMSE (m/s) | Failures (batch / EKF) |",
             "|---|---:|---:|---:|---:|---:|"]
    for level, label in zip(levels, labels):
        b, e = [next(r for r in summary if r["phase_sigma_deg"] == level and r["solver"] == s) for s in solvers]
        lines.append(f"| {label} | {b['position_m_rmse']:.2f} | {e['position_m_rmse']:.2f} | "
                     f"{b['velocity_m_s_rmse']:.4f} | {e['velocity_m_s_rmse']:.4f} | {b['failures']} / {e['failures']} |")
    drift_change = config['stationary_drift_deg'] * np.sqrt(2 * -np.expm1(-config['retained_duration_seconds']/(config['tau_hours']*3600)))
    rts_failures = sum(r["solver"] == "rts" and r["status"] != "ok" for r in rows)
    linearized_checks = sum(r.get("convergence") == "verified_linearized_tolerance" for r in rows)
    lines += ["", f"OU drift has a {config['tau_hours']:g}-hour correlation time and "
              f"{config['stationary_drift_deg']:g}° stationary standard deviation, giving "
              f"{drift_change:.3f}° RMS change across this observed pass. This is separate from white phase measurement noise.", "",
              f"RTS failures: {rts_failures}. The final RTS estimate equals the final EKF estimate; smoothing changes earlier epochs.", "",
              f"{linearized_checks} batch fits met the supplied tolerance through the independent undamped QR check after the numeris iteration limit. "
              "These are explicitly identified in results.csv; iteration-limit fits failing that check remain failures.", "",
              "`summary.csv` includes medians, 95th percentiles, RMSE confidence intervals, NEES, NIS and coverage. "
              "Error statistics use successful trials; failures are counted explicitly. "
              "The reported covariance includes uncertainty in receiver offsets. Dotted √trace curves are aggregate standard deviations, not scalar confidence bounds.", "",
              f"The nominal 95% orbital ellipsoid covers {100*min(r['coverage_95'] for r in summary):.0f}–"
              f"{100*max(r['coverage_95'] for r in summary):.0f}% of trials across the cases. "
              "The covariance does not achieve nominal coverage in every case; assess it alongside the empirical errors.", "",
              "These results describe one internally consistent synthetic geometry and dynamics model. "
              "Continuous phase, accurate surveyed geometry, known carrier, independent white sensor noise and fixed true drag are assumed. "
              "Absolute calibration, cycle slips, atmospheric effects and orbital force-model errors are not assessed.", "",
              "![Final accuracy](accuracy.png)", "", "![Pass errors](pass_errors.png)", "",
              "![Phase offset](phase_offset.png)", "", "![Covariance consistency](consistency.png)", ""]
    if args.compare_negligible_drift:
        lines.extend(compare_drift(output, args.compare_negligible_drift, config, rows))
    (output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"Saved summary, report and PNG/SVG/PDF figures in {output}")


if __name__ == "__main__":
    main()
