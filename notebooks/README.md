This folder contains a generator script that creates one benchmark notebook per model.

Usage

1. From the project root run:

```bash
python notebooks/generate_model_notebooks.py
```

2. The script writes notebooks to `notebooks/generated/`.

Notebook template features

- Device detection (CPU/GPU)
- Dataset discovery (looks for `feeds_cleaned_15T.csv`, `feeds_cleaned.csv`, `feeds.csv` at project root)
- Minimal preprocessing stub
- A cell that invokes the repository benchmark harness (`data/kaggle_dataset/kaggle_benchmarking_suite.py`) with the model name

Next steps

- Review notebooks under `notebooks/generated/` and adjust model-specific preprocessing/training cells as needed.
- If you want, I can run the generator now and then run one notebook end-to-end for a chosen model/city.
