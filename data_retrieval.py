import io
import json
import pandas as pd
import requests


def validate_config(config, model_name="model_1.0"):
    required_sections = [
        "indicators",
        "sources",
        "source_mappings",
        "model_definitions",
    ]

    validation_checks = {
        "top_level_structure": all(
            section in config for section in required_sections
        )
        and all(
            isinstance(config[section], dict) for section in required_sections
        ),
        "model_definitions": (
            model_name in config["model_definitions"]
            and isinstance(config["model_definitions"][model_name], list)
            and all(
                indicator
                in [
                    definition[0]
                    for definition in config["indicators"].values()
                ]
                for indicator in config["model_definitions"][model_name]
            )
        ),
        "indicators": (
            all(
                isinstance(name, str) and name.strip() != ""
                for name in config["indicators"]
            )
            and all(
                isinstance(indicator, list)
                and len(indicator) >= 2
                and isinstance(indicator[0], str)
                and indicator[0].strip() != ""
                and isinstance(indicator[1], str)
                and indicator[1].strip() != ""
                for indicator in config["indicators"].values()
            )
            and len(config["indicators"].values())
            == len(
                set(
                    indicator[0] for indicator in config["indicators"].values()
                )
            )
        ),
        "sources": (
            all(
                indicator in config["sources"]
                for indicator in config["model_definitions"][model_name]
            )
            and all(
                isinstance(config["sources"][indicator], list)
                for indicator in config["model_definitions"][model_name]
            )
            and all(
                len(config["sources"][indicator]) > 0
                for indicator in config["model_definitions"][model_name]
            )
        ),
        "source_mappings": (
            all(
                source in config["source_mappings"]
                for indicator in config["model_definitions"][model_name]
                for source in config["sources"][indicator]
            )
            and all(
                indicator in config["source_mappings"][source]
                for indicator in config["model_definitions"][model_name]
                for source in config["sources"][indicator]
            )
            and all(
                isinstance(config["source_mappings"][source][indicator], list)
                and len(config["source_mappings"][source][indicator]) >= 2
                and isinstance(
                    config["source_mappings"][source][indicator][-1], str
                )
                and config["source_mappings"][source][indicator][
                    -1
                ].strip()
                != ""
                for indicator in config["model_definitions"][model_name]
                for source in config["sources"][indicator]
            )
        ),
    }

    failed_checks = [
        name for name, valid in validation_checks.items() if not valid
    ]
    if failed_checks:
        raise ValueError(
            f"Configuration validation failed: {', '.join(failed_checks)}"
        )


def retrieve_ecb_data(indicator, config):
    dataflow, series_key, source_unit = config["source_mappings"]["ECB_API"][
        indicator
    ]
    canonical_unit = next(
        definition[1]
        for definition in config["indicators"].values()
        if definition[0] == indicator
    )
    url = f"https://data-api.ecb.europa.eu/service/data/{dataflow}/{series_key}?format=jsondata"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        series = next(iter(data["dataSets"][0]["series"].values()))
        time_periods = data["structure"]["dimensions"]["observation"][0][
            "values"
        ]
        observations = series["observations"]

        rows = []
        for index, observation in observations.items():
            rows.append({
                "Time": time_periods[int(index)]["id"],
                "Indicator": indicator,
                "Value": observation[0],
                "Source_Unit": source_unit,
                "Canonical_Unit": canonical_unit,
                "Source": "ECB_API",
            })
        return True, pd.DataFrame(rows)
    except Exception as e:
        print(f"ECB_API failed for {indicator}: {e}")
        return False, None


def retrieve_offline_data(indicator, config):
    file_name, column_name, source_unit = config["source_mappings"]["OFFLINE"][
        indicator
    ]
    canonical_unit = next(
        definition[1]
        for definition in config["indicators"].values()
        if definition[0] == indicator
    )

    try:
        if file_name.lower().endswith(".csv"):
            data = pd.read_csv(file_name)
        elif file_name.lower().endswith(".xlsx"):
            data = pd.read_excel(file_name)
        else:
            return False, None

        time_column = next(
            (
                column
                for column in data.columns
                if column.lower() in ["time", "date"]
            ),
            None,
        )
        if time_column is None or column_name not in data.columns:
            return False, None

        data = data.rename(columns={time_column: "Time"})
        data = data[data["Time"].notna()][["Time", column_name]].rename(
            columns={column_name: "Value"}
        )
        data.insert(1, "Indicator", indicator)
        data["Source_Unit"] = source_unit
        data["Canonical_Unit"] = canonical_unit
        data["Source"] = "OFFLINE"
        return True, data
    except Exception as e:
        print(f"OFFLINE failed for {indicator}: {e}")
        return False, None


def retrieve_JKshock_Github(indicator, config):
    owner, repository, file_name, column_name, source_unit = config[
        "source_mappings"
    ]["GITHUB_JK"][indicator]
    canonical_unit = next(
        definition[1]
        for definition in config["indicators"].values()
        if definition[0] == indicator
    )

    try:
        branches_url = (
            f"https://api.github.com/repos/{owner}/{repository}/branches"
        )
        response = requests.get(
            branches_url, params={"per_page": 100}, timeout=30
        )
        response.raise_for_status()
        branches = response.json()

        file_data = None
        matched_branch = None
        for branch in branches:
            branch_name = branch["name"]
            file_url = f"https://api.github.com/repos/{owner}/{repository}/contents/{file_name}"
            response = requests.get(
                file_url, params={"ref": branch_name}, timeout=30
            )
            if response.status_code == 200:
                file_data = response.json()
                matched_branch = branch_name
                break

        if file_data is None:
            return False, None

        download_url = file_data.get("download_url")
        response = requests.get(
            download_url, params={"ref": matched_branch}, timeout=30
        )
        response.raise_for_status()

        data = pd.read_csv(io.StringIO(response.text))
        if "date" not in data.columns or column_name not in data.columns:
            return False, None

        data = data[data["date"].notna()][["date", column_name]].rename(
            columns={"date": "Time", column_name: "Value"}
        )
        data.insert(1, "Indicator", indicator)
        data["Source_Unit"] = source_unit
        data["Canonical_Unit"] = canonical_unit
        data["Source"] = "GITHUB_JK"
        return True, data
    except Exception as e:
        print(f"GITHUB_JK failed for {indicator}: {e}")
        return False, None


def run_data_retrieval(config_path="core_mappings.json", model_name="model_1.0"):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    validate_config(config, model_name)
    model_indicators = config["model_definitions"][model_name]

    working_data = []
    retrieval_report = []

    for indicator in model_indicators:
        indicator_retrieved = False
        for source in config["sources"][indicator]:
            if source == "ECB_API":
                success, data = retrieve_ecb_data(indicator, config)
            elif source == "OFFLINE":
                success, data = retrieve_offline_data(indicator, config)
            elif source == "GITHUB_JK":
                success, data = retrieve_JKshock_Github(indicator, config)
            else:
                success, data = False, None

            if success:
                working_data.append(data)
                retrieval_report.append(
                    {"Indicator": indicator, "Source": source, "Status": "Retrieved"}
                )
                indicator_retrieved = True
                break

        if not indicator_retrieved:
            retrieval_report.append(
                {"Indicator": indicator, "Source": None, "Status": "Unavailable"}
            )

    working_df = (
        pd.concat(working_data, ignore_index=True)
        if working_data
        else pd.DataFrame(
            columns=[
                "Time",
                "Indicator",
                "Value",
                "Source_Unit",
                "Canonical_Unit",
                "Source",
            ]
        )
    )

    return working_df, retrieval_report