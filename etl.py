import re
import pandas as pd

DEFAULT_LAG_RULES = {
    30: ["EURIBOR_3M"],
    45: ["HICP", "HICPX_SUPPLY", "HICPX_DEMAND"],
    60: [],
    120: [],
}


def streamline_dates_fixed_lag(df, release_lag_rules=None):
    if release_lag_rules is None:
        release_lag_rules = DEFAULT_LAG_RULES

    df = df.copy()
    lag_lookup = {
        indicator: lag
        for lag, indicators in release_lag_rules.items()
        for indicator in indicators
    }

    def transform_date(value, indicator):
        lag = lag_lookup.get(indicator, 0)
        if isinstance(value, pd.Timestamp):
            return value.normalize()
        if not isinstance(value, str):
            return pd.Timestamp(value).normalize()

        value = value.strip()
        if re.fullmatch(r"\d{4}-Q[1-4]", value, re.IGNORECASE):
            year, quarter = int(value[:4]), int(value[-1])
            date = pd.Timestamp(year=year, month=(quarter - 1) * 3 + 1, day=1)
            return date + pd.Timedelta(days=lag)
        if re.fullmatch(r"\d{4}-\d{2}", value):
            date = pd.to_datetime(value, format="%Y-%m")
            return date + pd.Timedelta(days=lag)
        if re.fullmatch(r"\d{4}", value):
            date = pd.to_datetime(value, format="%Y")
            return date + pd.Timedelta(days=lag)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return pd.to_datetime(value, format="%Y-%m-%d")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value):
            return pd.to_datetime(value).normalize()

        raise ValueError(f"Unrecognized Time format: {value}")

    df["Time"] = [
        transform_date(time, indicator)
        for time, indicator in zip(df["Time"], df["Indicator"])
    ]
    df = df.rename(columns={"Time": "Date"})

    expanded_data = []
    for indicator, group in df.groupby("Indicator"):
        date_range = pd.date_range(
            start=min(group["Date"]), end=max(group["Date"]), freq="D"
        )
        group = (
            group.set_index("Date")
            .reindex(date_range)
            .rename_axis("Date")
            .reset_index()
        )
        group["Indicator"] = indicator
        for column in ["Source_Unit", "Canonical_Unit", "Source"]:
            group[column] = group[column].ffill().bfill()
        expanded_data.append(group)

    df = pd.concat(expanded_data, ignore_index=True)
    df["Date"] = df["Date"].dt.date
    return df


def prepare_and_harmonize_data(working_df, release_lag_rules=None):
    df_streamlined = streamline_dates_fixed_lag(working_df, release_lag_rules)
    model_df = (
        df_streamlined.pivot(index="Date", columns="Indicator", values="Value")
        .reset_index()
    )
    model_df.columns.name = None
    return model_df