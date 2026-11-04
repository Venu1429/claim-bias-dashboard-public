# Insurance claim bias dashboard
Streamlit app testing whether repudiation of death claims differs between groups after allowing for legitimate underwriting factors.
It reads only `aggregates.json` (summary tables). The raw claims file is not part of this repository.
To rebuild the summaries from your own claims file: `python build_aggregates.py your_claims.csv`
Rebuilding also needs `scikit-learn` (`pip install scikit-learn`).
Run the app: `pip install -r requirements.txt` then `streamlit run app.py`
