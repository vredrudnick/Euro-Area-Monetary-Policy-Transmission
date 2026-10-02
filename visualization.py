import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def output_plot(
    model,
    model_name,
    indicators=None,
    interactions=None,
    table_statistics=None,
    model_statistics=None,
    config=None,
    save_dir=".",
):
    """
    Reproduces the exact notebook Layer 4 visualization:
      - Figure 1: Regression Results Table + Model Statistics
      - Figure 2: Coefficient Plot (95% CI) + Interaction Effects
    Saves both figures to disk in `save_dir`.
    """
    # Build indicator_names dictionary using config safely
    indicator_names = {}
    if config is not None and isinstance(config, dict) and "indicators" in config:
        indicator_names = {
            internal_name: real_name
            for real_name, (internal_name, unit) in config["indicators"].items()
        }

    indicator_names["HICPX_DEMAND"] = "Demand-driven"
    indicator_names["HICPX_SUPPLY"] = "Supply-driven"

    def display_name(term):
        parts = term.split(":")
        return " × ".join(
            indicator_names.get(part, part) for part in parts
        )

    interactions = interactions if interactions is not None else []

    table_statistics = table_statistics if table_statistics is not None else [
        "coef",
        "std_err",
        "t_value",
        "p_value",
        "conf_int",
    ]

    model_statistics = model_statistics if model_statistics is not None else [
        "nobs",
        "rsquared",
        "rsquared_adj",
        "fvalue",
        "f_pvalue",
    ]

    params = model.params
    conf_int = model.conf_int()
    pvalues = model.pvalues

    # 1. Regression results table

    statistic_labels = {
        "coef": "Coefficient",
        "std_err": "Std. Error",
        "t_value": "t-value",
        "p_value": "p-value",
        "conf_int": "95% CI",
    }

    table_data = []

    for term in params.index:
        row = [display_name(term)]

        for statistic in table_statistics:
            if statistic == "coef":
                row.append(f"{params[term]:.4f}")

            elif statistic == "std_err":
                row.append(f"{model.bse[term]:.4f}")

            elif statistic == "t_value":
                row.append(f"{model.tvalues[term]:.3f}")

            elif statistic == "p_value":
                if pvalues[term] < 0.001:
                    row.append("<0.001")
                else:
                    row.append(f"{pvalues[term]:.3f}")

            elif statistic == "conf_int":
                row.append(
                    f"[{conf_int.loc[term, 0]:.4f}, "
                    f"{conf_int.loc[term, 1]:.4f}]"
                )

        table_data.append(row)

    column_labels = ["Variable"] + [statistic_labels[s] for s in table_statistics]

    # 2. Coefficient + 95% CI plot

    plot_terms = [
        term for term in params.index if term.lower() not in ("intercept", "const")
    ]

    coefficients = np.array([params[term] for term in plot_terms])

    lower = np.array([conf_int.loc[term, 0] for term in plot_terms])

    upper = np.array([conf_int.loc[term, 1] for term in plot_terms])

    errors = np.array([coefficients - lower, upper - coefficients])

    # 3. Interaction-effect plot

    interaction_labels = []
    interaction_coefficients = []

    for interaction in interactions:
        interaction_term = ":".join(interaction)

        if interaction_term in params.index:
            interaction_labels.append(
                display_name(f"{interaction[0]}:{interaction[1]}")
            )
            val = params[interaction_term]
            interaction_coefficients.append(float(val) if hasattr(val, "item") else val)

    # 4. Model statistics

    statistic_values = {}

    for statistic in model_statistics:
        if statistic == "nobs":
            statistic_values["Observations"] = int(model.nobs)

        elif statistic == "rsquared":
            statistic_values["R²"] = f"{model.rsquared:.4f}"

        elif statistic == "rsquared_adj":
            statistic_values["Adjusted R²"] = f"{model.rsquared_adj:.4f}"

        elif statistic == "fvalue":
            statistic_values["F-statistic"] = f"{model.fvalue:.4f}"

        elif statistic == "f_pvalue":
            statistic_values["F p-value"] = f"{model.f_pvalue:.4g}"

    statistics_text = "   ".join(
        f"{name}: {value}" for name, value in statistic_values.items()
    )

    # Figure 1: Regression table + model statistics

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [4, 1]}
    )

    # Regression table

    axes[0].axis("off")

    table = axes[0].table(
        cellText=table_data, colLabels=column_labels, cellLoc="center", loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.8)

    axes[0].set_title("Regression Results", fontweight="bold")

    # Highlight significant coefficients

    for row_index, term in enumerate(params.index, start=1):
        if pvalues[term] < 0.05:
            for column_index in range(len(column_labels)):
                table[row_index, column_index].set_text_props(fontweight="bold")

    # Model statistics

    axes[1].axis("off")

    axes[1].text(0.5, 0.5, statistics_text, ha="center", va="center", fontsize=11)

    axes[1].set_title("Model Statistics", fontweight="bold")

    fig.suptitle(f"{model_name}", fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_table_path = f"{save_dir}/{model_name}_regression_table.png"
    plt.savefig(fig_table_path, dpi=300)
    print(f"Saved figure: {fig_table_path}")
    plt.show()

    # Figure 2: Graphical results

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Coefficient plot

    y_positions = np.arange(len(plot_terms))

    axes[0].errorbar(coefficients, y_positions, xerr=errors, fmt="o", capsize=4)

    axes[0].axvline(0, linestyle="--", linewidth=1)

    axes[0].set_yticks(y_positions)

    axes[0].set_yticklabels([display_name(term) for term in plot_terms])

    axes[0].invert_yaxis()

    axes[0].set_xlabel("Coefficient")

    axes[0].set_title("Coefficients with 95% CI", fontweight="bold")

    # Interaction plot

    if len(interaction_coefficients) > 0:
        axes[1].barh(interaction_labels, interaction_coefficients)

        axes[1].axvline(0, linestyle="--", linewidth=1)

        axes[1].set_xlabel("Coefficient")

        axes[1].set_title("Interaction Effects", fontweight="bold")

    else:
        axes[1].text(
            0.5, 0.5, "No interaction terms specified", ha="center", va="center"
        )

        axes[1].axis("off")

    fig.suptitle(
        f"{model_name} — Graphical Results", fontsize=16, fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_graph_path = f"{save_dir}/{model_name}_graphical_results.png"
    plt.savefig(fig_graph_path, dpi=300)
    print(f"Saved figure: {fig_graph_path}")
    plt.show()


def run_visualization_pipeline(
    model_result,
    model_name="model_1.0",
    model_indicators=None,
    config=None,
    save_dir=".",
):
    """
    Executes Layer 4 validation and calls output_plot.
    """
    if not hasattr(model_result, "params"):
        raise ValueError(
            "Layer 4 validation failed. Model result is not a valid "
            "statsmodels regression result."
        )

    if model_name == "model_0.1":
        interactions = [("MPS", "HICPX_DEMAND"), ("MPS", "HICPX_SUPPLY")]
    elif model_name == "model_1.0":
        interactions = [("MPT", "HICPX_DEMAND"), ("MPT", "HICPX_SUPPLY")]
    else:
        interactions = []

    output_plot(
        model=model_result,
        model_name=model_name,
        indicators=model_indicators,
        interactions=interactions,
        table_statistics=["coef", "std_err", "t_value", "p_value", "conf_int"],
        model_statistics=["nobs", "rsquared", "rsquared_adj", "fvalue", "f_pvalue"],
        config=config,
        save_dir=save_dir,
    )