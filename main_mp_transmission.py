import json
from data_retrieval import run_data_retrieval
from etl import prepare_and_harmonize_data
from model_implementation import run_model_10
from visualization import run_visualization_pipeline


def main():
    config_path = "core_mappings.json"
    model_name = "model_1.0"

    print("--- Running Layer 1: Data Retrieval ---")
    working_df, report = run_data_retrieval(
        config_path=config_path, model_name=model_name
    )
    print("Retrieval Report:", report)

    print("\n--- Running Layer 2: Data Preparation & Harmonization ---")
    model_df = prepare_and_harmonize_data(working_df)

    print("\n--- Running Layer 3: Model Construction ---")
    time_span = ("2017-02-15", "2022-10-16")
    model_result, model_df = run_model_10(model_df, time_span=time_span)

    print("\n--- Running Layer 4: Visualization ---")
    config = None
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception:
        pass

    run_visualization_pipeline(
        model_result=model_result,
        model_name=model_name,
        model_indicators=None,
        config=config,
        save_dir=".",
    )


if __name__ == "__main__":
    main()