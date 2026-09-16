import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# =============================================================================
# Constants
# =============================================================================

WINNER_LIST = [
    "berniemorenoforohio-historical-stats.csv",
    "davemccormickpa-historical-stats.csv",
    "gallegoforaz-historical-stats.csv",
    "senslotkin-historical-stats.csv",
    "sentedcruz-historical-stats.csv",
    "tammybaldwinwi-historical-stats.csv",
]

LOSER_LIST = [
    "colinallred-historical-stats.csv",
    "erichovde-historical-stats.csv",
    "karilake-historical-stats.csv",
    "mikerogersformi-historical-stats.csv",
    "senbobcasey-historical-stats.csv",
    "sherrod-historical-stats.csv",
]

# Set DATA_ROOT to the folder holding the per-candidate historical-stats
# CSVs (see WINNER_LIST / LOSER_LIST above), e.g.
#   export DATA_ROOT=/path/to/senator_ig_engagement
ROOT = os.path.join(
    os.environ.get("DATA_ROOT", "./data/senator_ig_engagement"),
    "downloaded_historicalStats",
)
START_DATE = "2024-09-01"
END_DATE = "2025-03-03"


# =============================================================================
# Path helpers
# =============================================================================

def resolve_folder(root, input_list, end_date):
    if input_list is WINNER_LIST:
        tag = f"growth_winner_list_{end_date}"
    elif input_list is LOSER_LIST:
        tag = f"growth_loser_list_{end_date}"
    else:
        tag = f"growth_combined_both_{end_date}"
    return os.path.join(root, tag)


# =============================================================================
# Load helpers
# =============================================================================

def load_regression_df(folder, start_date, end_date):
    """
    Load and date-filter the merged regression CSV for a given folder.
    Expected file:
        {folder}/{basename(folder)}_regression.csv
    """
    path = os.path.join(folder, f"{os.path.basename(folder)}_regression.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[
        (df["date"] >= pd.Timestamp(start_date)) &
        (df["date"] <= pd.Timestamp(end_date))
    ].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded: {path} ({len(df)} rows)")
    return df


# =============================================================================
# Fitting helpers
# =============================================================================

def available_predictor_cols(df, preferred=("y_avg", "y_pred", "y_pred_gp")):
    return [col for col in preferred if col in df.columns]


def fit_affine_from_points(df, target_name, predictor_col, observations):
    """
    Fit: raw = a * predictor + b

    observations supports both:
    1) {"date": "...", "raw": 1234}  -> manual raw
    2) {"date": "..."}               -> auto read from df[target_name]

    Rules:
    - 1 point  -> proportional fallback (b = 0)
    - 2 points -> exact affine
    - >2 points -> OLS
    """
    rows = []

    for obs in observations:
        ts = pd.Timestamp(obs["date"])
        match = df[df["date"] == ts]

        if match.empty:
            print(f"Warning: observation date {ts.date()} not found, skipped.")
            continue

        x = float(match.iloc[0][predictor_col])

        if "raw" in obs and obs["raw"] is not None:
            y = float(obs["raw"])
            raw_source = "manual"
        else:
            if target_name not in df.columns:
                raise ValueError(
                    f"Target column '{target_name}' not found, cannot auto-read raw from CSV."
                )

            raw_val = match.iloc[0][target_name]
            if pd.isna(raw_val):
                raise ValueError(
                    f"No raw value found in CSV for '{target_name}' on {ts.date()}."
                )

            y = float(raw_val)
            raw_source = "csv"

        rows.append((ts, x, y, raw_source))

    if not rows:
        raise ValueError(f"No valid observations found for predictor '{predictor_col}'.")

    X = np.array([r[1] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows], dtype=float)

    if len(rows) == 1:
        x_ref = X[0]
        if x_ref == 0:
            a, b = 0.0, y[0]
        else:
            a, b = y[0] / x_ref, 0.0
        method = "single-point proportional"

    elif len(rows) == 2:
        x1, x2 = X
        y1, y2 = y

        if x2 == x1:
            raise ValueError(
                f"Cannot fit two-point affine map because predictor values are identical ({x1:.6f})."
            )

        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        method = "two-point exact affine"

    else:
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        a = float(model.coef_[0])
        b = float(model.intercept_)
        method = f"multi-point OLS ({len(rows)} pts)"

    diagnostics = pd.DataFrame(
        rows, columns=["date", "predictor", "raw", "raw_source"]
    )
    diagnostics["predictor_col"] = predictor_col
    diagnostics["fit_method"] = method
    diagnostics["fitted_raw"] = a * diagnostics["predictor"] + b
    diagnostics["abs_error"] = (diagnostics["raw"] - diagnostics["fitted_raw"]).abs()

    return a, b, method, diagnostics


def fit_affine_from_actual_period(df, target_name, predictor_col, cutoff_date):
    """
    Fit: raw = a * predictor + b
    using all available actual observations on/after cutoff_date.
    """
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in dataframe.")

    cutoff = pd.Timestamp(cutoff_date)
    fit_df = df.loc[
        (df["date"] >= cutoff) & df[target_name].notna(),
        ["date", target_name, predictor_col]
    ].copy()

    if len(fit_df) == 0:
        raise ValueError(
            f"No actual data found for '{target_name}' on/after cutoff {cutoff.date()}."
        )

    X = fit_df[[predictor_col]].values.astype(float)
    y = fit_df[target_name].values.astype(float)

    if len(fit_df) == 1:
        x_ref = float(X[0, 0])
        y_ref = float(y[0])

        if x_ref == 0:
            a, b = 0.0, y_ref
        else:
            a, b = y_ref / x_ref, 0.0

        method = "cutoff single-point proportional"
    else:
        model = LinearRegression()
        model.fit(X, y)
        a = float(model.coef_[0])
        b = float(model.intercept_)
        method = f"cutoff OLS ({len(fit_df)} pts)"

    fit_df["raw_source"] = "csv_actual_period"
    fit_df["predictor"] = fit_df[predictor_col]
    fit_df["raw"] = fit_df[target_name]
    fit_df["predictor_col"] = predictor_col
    fit_df["fit_method"] = method
    fit_df["fitted_raw"] = a * fit_df[predictor_col] + b
    fit_df["abs_error"] = (fit_df[target_name] - fit_df["fitted_raw"]).abs()

    diagnostics = fit_df[
        ["date", "predictor", "raw", "raw_source", "predictor_col", "fit_method", "fitted_raw", "abs_error"]
    ].copy()

    return a, b, method, diagnostics


# =============================================================================
# Backfill builders
# =============================================================================

def minmax_scale(series):
    s = pd.Series(series, dtype=float)
    vmin = s.min(skipna=True)
    vmax = s.max(skipna=True)

    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.nan, index=s.index)

    return (s - vmin) / (vmax - vmin)


def build_backfilled_series(df, target_name, predictor_col, a, b, cutoff_date=None):
    """
    Returns a dataframe with:
    - predicted_raw
    - actual_raw
    - final_raw
    - final_minmax
    - source

    If cutoff_date is provided:
      - date < cutoff: use predicted
      - date >= cutoff: use actual if available, else predicted

    If cutoff_date is None:
      - use actual when available, else predicted
    """
    out = df[["date", predictor_col]].copy()
    out["predicted_raw"] = a * out[predictor_col] + b
    out["predicted_raw"] = out["predicted_raw"].clip(lower=0)
    out["predicted_raw_rounded"] = np.rint(out["predicted_raw"]).astype(int)

    if target_name in df.columns:
        actual = df[target_name]
    else:
        actual = pd.Series(np.nan, index=df.index)

    out["actual_raw"] = actual

    if cutoff_date is None:
        out["final_raw"] = actual.combine_first(out["predicted_raw_rounded"])
        out["source"] = np.where(actual.notna(), "actual", "predicted")
    else:
        cutoff = pd.Timestamp(cutoff_date)
        before = out["date"] < cutoff
        after = ~before

        out["final_raw"] = out["predicted_raw_rounded"]
        out.loc[after & actual.notna(), "final_raw"] = actual[after & actual.notna()]

        out["source"] = np.where(
            before,
            "predicted",
            np.where(actual.notna(), "actual", "predicted")
        )

    out["final_minmax"] = minmax_scale(out["final_raw"])
    return out


def merge_three_backfills(df, target_name, backfill_dict):
    """
    Merge all predictor-specific backfill results into one dataframe.

    backfill_dict format:
        {
            "y_avg": backfill_df,
            "y_pred": backfill_df,
            "y_pred_gp": backfill_df,
        }
    """
    out = df.copy()

    for predictor_col, bf in backfill_dict.items():
        out[f"{target_name}_{predictor_col}_predicted_raw"] = bf["predicted_raw_rounded"].values
        out[f"{target_name}_{predictor_col}_final_raw"] = bf["final_raw"].values
        out[f"{target_name}_{predictor_col}_final_minmax"] = bf["final_minmax"].values
        out[f"{target_name}_{predictor_col}_source"] = bf["source"].values

    if target_name in df.columns:
        out[f"{target_name}_actual_raw"] = df[target_name].values
    else:
        out[f"{target_name}_actual_raw"] = np.nan

    return out


# =============================================================================
# Save / plot helpers
# =============================================================================

def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def plot_three_backfills(df, target_name, backfill_dict, fit_summary_df, folder, file_stub):
    """
    One figure:
    - Top panel: raw followers
    - Bottom panel: minmax curves
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    predictor_order = ["y_avg", "y_pred", "y_pred_gp"]
    predictor_order = [p for p in predictor_order if p in backfill_dict]

    # Top: raw
    for predictor_col in predictor_order:
        bf = backfill_dict[predictor_col]
        axes[0].plot(
            df["date"],
            bf["predicted_raw_rounded"],
            linewidth=2,
            label=f"{predictor_col} predicted"
        )
        axes[0].plot(
            df["date"],
            bf["final_raw"],
            linewidth=2,
            linestyle="--",
            label=f"{predictor_col} final"
        )

    if target_name in df.columns:
        axes[0].scatter(
            df["date"],
            df[target_name],
            s=18,
            label="Actual raw"
        )

    axes[0].set_ylabel("Followers")
    axes[0].set_title(f"{target_name} — three backfill predictors")
    axes[0].legend(loc="upper left", ncol=2)

    # Bottom: minmax
    for predictor_col in predictor_order:
        bf = backfill_dict[predictor_col]
        axes[1].plot(
            df["date"],
            bf["final_minmax"],
            linewidth=2,
            label=f"{predictor_col} final minmax"
        )

    axes[1].set_ylabel("Scaled value")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")

    # Fit summary as text box
    summary_lines = []
    for _, row in fit_summary_df.iterrows():
        mae = row["mean_abs_error"]
        mae_text = "nan" if pd.isna(mae) else f"{mae:.2f}"
        summary_lines.append(
            f"{row['predictor_col']}: {row['fit_method']} | MAE={mae_text}"
        )

    if summary_lines:
        axes[0].text(
            1.01, 0.98,
            "\n".join(summary_lines),
            transform=axes[0].transAxes,
            va="top",
            fontsize=9
        )

    plt.xticks(rotation=45)
    fig.tight_layout()

    path = os.path.join(folder, f"{file_stub}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Chart saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # both
    input_list = LOSER_LIST + WINNER_LIST
    folder = resolve_folder(ROOT, input_list, END_DATE)
    base = os.path.basename(folder)

    # ---------------------------------------------------------------------
    # Config
    #
    # Mode 1: observations mode
    #   - no cutoff_date
    #   - use 1~N points
    #
    # Mode 2: cutoff mode
    #   - provide cutoff_date
    #   - use all actual values on/after cutoff
    #
    # observations supports both:
    #   {"date": "...", "raw": 1234}  -> manual
    #   {"date": "..."}               -> auto read from CSV
    # ---------------------------------------------------------------------
    candidates_config = {
        "Eric Hovde": {
            "predictor_cols": ["y_avg", "y_pred", "y_pred_gp"],
            "observations": [
                {"date": "2025-02-24"},
                {"date": "2025-03-03"},
            ],
        },
        #  "Eric Hovde": {
        #     "predictor_cols": ["y_avg", "y_pred", "y_pred_gp"],
        #     "observations": [
        #         {"date": "2024-12-24", "raw": 2895},
        #         {"date": "2025-02-24"},
        #     ],

        # },
        # "Mike Rogers": {
        #     "predictor_cols": ["y_avg", "y_pred", "y_pred_gp"],
        #     "cutoff_date": "2024-11-13",
        # },


        # Example: mixed manual + csv
        # "Kari Lake": {
        #     "predictor_cols": ["y_avg", "y_pred", "y_pred_gp"],
        #     "observations": [
        #         {"date": "2024-12-24"},
        #         {"date": "2025-02-24", "raw": 1500},
        #     ],
        # },

        # Example: cutoff mode
        # "Some Candidate": {
        #     "predictor_cols": ["y_avg", "y_pred", "y_pred_gp"],
        #     "cutoff_date": "2024-10-22",
        # },
    }

    # ---------------------------------------------------------------------
    # Load regression data
    # ---------------------------------------------------------------------
    df = load_regression_df(folder, START_DATE, END_DATE)

    # ---------------------------------------------------------------------
    # Per-candidate
    # ---------------------------------------------------------------------
    for target_name, cfg in candidates_config.items():
        print("\n" + "=" * 80)
        print(target_name)
        print("=" * 80)

        predictor_cols = cfg.get("predictor_cols", ["y_avg", "y_pred", "y_pred_gp"])
        predictor_cols = [col for col in predictor_cols if col in df.columns]

        if not predictor_cols:
            predictor_cols = available_predictor_cols(df)

        cutoff_date = cfg.get("cutoff_date")
        observations = cfg.get("observations", [])

        all_diagnostics = []
        fit_summary_rows = []
        backfill_dict = {}

        for predictor_col in predictor_cols:
            print("-" * 60)
            print(f"Predictor: {predictor_col}")

            try:
                if cutoff_date is not None:
                    a, b, fit_method, diagnostics = fit_affine_from_actual_period(
                        df=df,
                        target_name=target_name,
                        predictor_col=predictor_col,
                        cutoff_date=cutoff_date,
                    )
                else:
                    a, b, fit_method, diagnostics = fit_affine_from_points(
                        df=df,
                        target_name=target_name,
                        predictor_col=predictor_col,
                        observations=observations,
                    )
            except ValueError as e:
                print(f"Skipped {predictor_col} — {e}")
                continue

            print(f"Fit method: {fit_method}")
            print(f"Affine map: raw = {a:.6f} * {predictor_col} + {b:.6f}")

            backfill_df = build_backfilled_series(
                df=df,
                target_name=target_name,
                predictor_col=predictor_col,
                a=a,
                b=b,
                cutoff_date=cutoff_date,
            )

            backfill_dict[predictor_col] = backfill_df
            all_diagnostics.append(diagnostics)

            fit_summary_rows.append({
                "target_name": target_name,
                "predictor_col": predictor_col,
                "fit_method": fit_method,
                "a": a,
                "b": b,
                "n_fit_points": len(diagnostics),
                "mean_abs_error": diagnostics["abs_error"].mean() if "abs_error" in diagnostics.columns else np.nan,
                "max_abs_error": diagnostics["abs_error"].max() if "abs_error" in diagnostics.columns else np.nan,
            })

        if not backfill_dict:
            print(f"No valid backfill results for {target_name}.")
            continue

        diagnostics_df = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
        fit_summary_df = pd.DataFrame(fit_summary_rows)

        combined_output_df = merge_three_backfills(
            df=df,
            target_name=target_name,
            backfill_dict=backfill_dict,
        )

        # Save one diagnostics CSV
        save_csv(
            diagnostics_df,
            os.path.join(folder, f"{base}_{target_name}_all_predictors_fit_diagnostics.csv"),
        )

        # Save one fit summary CSV
        save_csv(
            fit_summary_df,
            os.path.join(folder, f"{base}_{target_name}_all_predictors_fit_summary.csv"),
        )

        # Save one combined backfill CSV
        save_csv(
            combined_output_df,
            os.path.join(folder, f"{base}_{target_name}_all_predictors_backfilled.csv"),
        )

        # Save one plot with all three predictors
        plot_three_backfills(
            df=df,
            target_name=target_name,
            backfill_dict=backfill_dict,
            fit_summary_df=fit_summary_df,
            folder=folder,
            file_stub=f"{base}_{target_name}_all_predictors_backfill",
        )


if __name__ == "__main__":
    main()