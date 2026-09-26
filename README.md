# Monetary Policy Transmission and Inflation Composition

This project examines whether the transmission of monetary policy in the Eurozone varies with the composition of inflation, distinguishing between demand-driven and supply-driven pressures.

The empirical framework builds on the approach of Najjar and Shapiro (2026) and adapts it to Eurozone data.

For a brief overview of the research question, methodology, and results, see:

**[Project Overview](MP_Transmission_overview.pdf)**

## Repository

- **`main_mp_transmission.py`**: Executes the end-to-end pipeline.
- **`data_retrieval.py`**: Manages Layer 1 data ingestion and source validation.
- **`etl.py`**: Handles Layer 2 data cleaning, lag adjustments, and matrix construction.
- **`model_implementation.py`**: Runs Layer 3 OLS regressions and shock classifications.
