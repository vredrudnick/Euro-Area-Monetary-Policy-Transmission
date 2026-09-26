from data_retrieval import run_data_retrieval
from etl import prepare_and_harmonize_data
from model_implementation import plot_model_results, run_model_10


def main():
    print("--- Running Layer 1: Data Retrieval ---")
    working_df, report = run_data_retrieval(
        config_path="core_mappings.json", model_name="model_1.0"
    )
    print("Retrieval Report:", report)

    print("\n--- Running Layer 2: Data Preparation & Harmonization ---")
    model_df = prepare_and_harmonize_data(working_df)

    print("\n--- Running Layer 3: Model Construction ---")
    time_span = ("2017-02-15", "2022-10-16")
    model_result, model_df = run_model_10(model_df, time_span=time_span)
    print(model_result.summary())

    print("\n--- Running Layer 4: Results & Visualization ---")
    plot_model_results(model_result, model_df)


if __name__ == "__main__":
    main()