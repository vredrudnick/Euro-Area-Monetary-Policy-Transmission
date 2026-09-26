import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf


def run_model_10(df, time_span=("2017-02-15", "2022-10-16")):
    df = df[
        (df["Date"] >= pd.to_datetime(time_span[0]).date())
        & (df["Date"] <= pd.to_datetime(time_span[1]).date())
    ].copy()

    df["AAA_1Y_change"] = df["AAA_1Y"].diff()

    mp_threshold = df["MP_SHOCK"].dropna().quantile(0.75)
    df["MPT"] = (
        df["MP_SHOCK"].notna() & (df["MP_SHOCK"] >= mp_threshold)
    ).astype(int)

    numeric_columns = df.select_dtypes(include="number").columns
    numeric_columns = numeric_columns.drop(
        ["MP_SHOCK", "MPT"], errors="ignore"
    )
    df[numeric_columns] = df[numeric_columns].ffill()

    model = smf.ols(
        "AAA_1Y_change ~ MPT * HICPX_DEMAND + MPT * HICPX_SUPPLY", data=df
    ).fit()

    return model, df


def plot_model_results(model_result, model_df):
    plt.figure(figsize=(10, 5))
    plt.plot(model_df["Date"], model_result.fittedvalues, label="Fitted Values")
    plt.plot(
        model_df["Date"],
        model_df["AAA_1Y_change"],
        label="Actual AAA 1Y Change",
        alpha=0.5,
    )
    plt.xlabel("Date")
    plt.ylabel("Yield Change")
    plt.title("Model 1.0: OLS Fit vs Actual Bond Yield Changes")
    plt.legend()
    plt.tight_layout()
    plt.show()