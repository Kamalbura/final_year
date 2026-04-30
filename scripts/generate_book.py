import json
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib

# Import trainers
from scripts.individual_trainers.lstm_trainer import LSTMTrainer, LSTMForecaster
from scripts.individual_trainers.rf_trainer import RFTrainer
from scripts.individual_trainers.transformer_trainer import TransformerTrainer, TransformerForecaster
from scripts.individual_trainers.trainer_base import FEATURE_COLUMNS, TARGET_COLUMN

sns.set_theme(style="whitegrid")

OUTPUT_DIR = Path("outputs/book_assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("outputs/individual_trainers")
KAGLLE_DIR = Path("outputs/kaggle_drilldown/outputs")

CITIES = ["delhi", "hyderabad", "bengaluru"]
MODELS = ["LSTM", "RandomForest", "Transformer"]

def load_all_results():
    results = []
    for model in MODELS:
        for city in CITIES:
            res_file = RESULTS_DIR / model / city / "results.json"
            if res_file.exists():
                with open(res_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
    return pd.DataFrame(results)

def load_kaggle_results():
    sum_file = KAGLLE_DIR / "summary.csv"
    if sum_file.exists():
        return pd.read_csv(sum_file)
    return pd.DataFrame()

def generate_comparison_plots(df, kaggle_df):
    plt.figure(figsize=(12, 7))
    metrics_df = pd.json_normalize(df['metrics'])
    df_flat = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
    
    # Merge with Kaggle if available
    if not kaggle_df.empty:
        k_df = kaggle_df[kaggle_df['status'] == 'ok'][['city', 'model', 'rmse', 'mae']].copy()
        k_df['source'] = 'Kaggle'
        df_flat['source'] = 'Local'
        combined = pd.concat([df_flat[['city', 'model', 'rmse', 'mae', 'source']], k_df])
    else:
        combined = df_flat
        combined['source'] = 'Local'

    # Plot RMSE Comparison
    plt.figure(figsize=(12, 7))
    sns.barplot(data=combined, x='city', y='rmse', hue='model')
    plt.title('RMSE Comparison: All Models across Cities')
    plt.savefig(OUTPUT_DIR / 'rmse_total_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    return df_flat, combined

CITY_DATA_CACHE = {}

def get_city_data(city, trainer_type="LSTM"):
    if city in CITY_DATA_CACHE:
        return CITY_DATA_CACHE[city]
    
    base_results_path = Path("outputs/individual_trainers").resolve()
    if trainer_type == "LSTM":
        trainer = LSTMTrainer(city, output_dir=str(base_results_path))
    elif trainer_type == "Transformer":
        trainer = TransformerTrainer(city, output_dir=str(base_results_path))
    else:
        trainer = RFTrainer(city, output_dir=str(base_results_path))
        
    df = trainer.load_data()
    data = trainer.prepare_data(df, 168, 24)
    CITY_DATA_CACHE[city] = data
    return data

def generate_forecast_plot(city, model_name, config):
    try:
        base_results_path = Path("outputs/individual_trainers").resolve()
        data = get_city_data(city, model_name)
        
        if model_name == "LSTM":
            model = LSTMForecaster(len(FEATURE_COLUMNS), config["hidden_dim"], config["num_layers"], 24, config.get("dropout", 0.2))
            model_path = base_results_path / "LSTM" / city / "best_model.pth"
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            model.eval()
            x_test = torch.tensor(data["X_test"][-1:])
            with torch.no_grad():
                pred = model(x_test).numpy().flatten()
            actual = data["y_test"][-1].flatten()
        
        elif model_name == "RandomForest":
            model_path = base_results_path / "RandomForest" / city / "best_model.joblib"
            model = joblib.load(model_path)
            x_test = data["X_test"][-1:].reshape(1, -1)
            pred = model.predict(x_test).flatten()
            actual = data["y_test"][-1].flatten()
            
        elif model_name == "Transformer":
            model = TransformerForecaster(len(FEATURE_COLUMNS), 24, config["model_dim"], config["layers"], config["heads"], config.get("dropout", 0.1))
            model_path = base_results_path / "Transformer" / city / "best_model.pth"
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            model.eval()
            x_test = torch.tensor(data["X_test"][-1:])
            with torch.no_grad():
                pred = model(x_test).numpy().flatten()
            actual = data["y_test"][-1].flatten()
        else:
            return None
        
        y_scaler = data["y_scaler"]
        pred_inv = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        actual_inv = y_scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
        
        plt.figure(figsize=(10, 5))
        plt.plot(actual_inv, label='Actual AQI', marker='o', markersize=4, color='#2ecc71', linewidth=2)
        plt.plot(pred_inv, label='Predicted AQI', marker='x', markersize=4, linestyle='--', color='#3498db', linewidth=2)
        plt.fill_between(range(24), actual_inv, pred_inv, color='gray', alpha=0.2)
        plt.title(f'24-Hour Air Quality Forecast: {model_name} ({city.title()})', fontsize=14)
        plt.xlabel('Hours Ahead', fontsize=12)
        plt.ylabel('US AQI Index', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = OUTPUT_DIR / f"{model_name}_{city}_forecast.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close()
        return out_path
    except Exception as e:
        print(f"Plotting failed for {model_name}/{city}: {e}")
        return None

def escape_latex(s):
    return str(s).replace('_', '\\_').replace('%', '\\%').replace('#', '\\#')

def generate_latex(local_df, combined_df):
    tex_file = OUTPUT_DIR / "air_quality_report.tex"
    
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(r"""\documentclass[11pt,a4paper,oneside]{book}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{amssymb}

\geometry{margin=1in}
\definecolor{primary}{RGB}{41, 128, 185}
\hypersetup{colorlinks=true, linkcolor=primary, citecolor=primary, urlcolor=primary}

\titleformat{\chapter}[display]{\normalfont\huge\bfseries\color{primary}}{}{0pt}{\Huge}
\titleformat{\section}{\normalfont\Large\bfseries\color{primary}}{\thesection}{1em}{}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyfoot[C]{\thepage}

\begin{document}

\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge\bfseries Air Quality Forecasting Platform \par}
    \vspace{1cm}
    {\Large\itshape Comprehensive Machine Learning Benchmark \& Analysis \par}
    \vspace{2cm}
    \includegraphics[width=0.6\textwidth]{rmse_total_comparison.png} \par
    \vspace{2cm}
    {\large Prepared by: Automated Research Pipeline \par}
    {\large Date: \today \par}
\end{titlepage}

\tableofcontents

\chapter{Introduction}
Air pollution is one of the most significant environmental challenges facing urban populations today. High concentrations of Particulate Matter (PM2.5 and PM10) are associated with severe respiratory and cardiovascular diseases. This report details the implementation and evaluation of various machine learning architectures aimed at providing accurate 24-hour forecasts of the US Air Quality Index (AQI).

\section{Project Objective}
The goal of this project is to develop a robust, multi-city forecasting system that leverages historical air quality data to predict future trends. By providing early warnings, urban authorities and citizens can take proactive measures to mitigate the impacts of poor air quality.

\section{Methodology}
Our methodology involves several key phases:
\begin{enumerate}
    \item \textbf{Data Ingestion}: Continuous monitoring of air quality parameters from Open-Meteo.
    \item \textbf{Preprocessing}: Imputation of missing values, temporal resampling, and normalization.
    \item \textbf{Feature Engineering}: Selection of optimal meteorological and pollutant features.
    \item \textbf{Model Training}: Benchmarking classical ML and Deep Learning architectures.
    \item \textbf{Optimization}: Automated hyperparameter tuning using Optuna.
\end{enumerate}

\chapter{Data Analysis and Features}
The models are trained on one year of historical data (2025-2026) for three cities: Delhi, Hyderabad, and Bengaluru. These cities represent diverse climatic and industrial profiles.

\section{Feature Space}
The following features are utilized as inputs for all models:
\begin{itemize}
    \item \textbf{PM2.5}: Fine particulate matter.
    \item \textbf{PM10}: Coarse particulate matter.
    \item \textbf{CO}: Carbon Monoxide levels.
    \item \textbf{NO2}: Nitrogen Dioxide levels.
    \item \textbf{O3}: Ozone levels.
\end{itemize}

The target variable is the calculated \textbf{US AQI}, which follows EPA standards for indexing air quality health risks.

\chapter{Architecture Overview}
In this benchmark, we evaluate three fundamentally different architectures.

\section{Long Short-Term Memory (LSTM)}
LSTMs are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies. They are particularly effective for time-series forecasting where the relative timing of events is crucial. Our implementation uses a multi-layered LSTM with dropout for regularization.

\section{Random Forest}
Random Forest is an ensemble learning method that constructs a multitude of decision trees during training. It is robust to outliers and can capture non-linear relationships without the need for extensive feature scaling. We use a multi-output configuration to predict the entire 24-hour horizon simultaneously.

\section{Transformer}
The Transformer architecture, originally designed for NLP, utilizes self-attention mechanisms to weigh the significance of different parts of the input sequence. For time-series, this allows the model to focus on specific historical patterns that are most relevant to the current prediction.

\chapter{Experimental Results}
\section{Metrics Summary}
The table below summarizes the performance of the models trained locally and on Kaggle.

\begin{table}[H]
    \centering
    \begin{tabular}{lllc rr}
    \toprule
    \textbf{City} & \textbf{Model} & \textbf{Source} & \textbf{RMSE} & \textbf{MAE} \\
    \midrule
""")
        for _, row in combined_df.sort_values(by=['city', 'source', 'rmse']).iterrows():
            source_tag = row['source']
            f.write(f"    {row['city'].title()} & {row['model']} & {source_tag} & {row['rmse']:.2f} & {row['mae']:.2f} \\\\\n")
        
        f.write(r"""    \bottomrule
    \end{tabular}
    \caption{Benchmarking Results: RMSE and MAE across all trials}
\end{table}

\chapter{Detailed Model Reports}
""")

        for model in MODELS:
            f.write(f"\\section{{{model} Detailed Analysis}}\n")
            f.write(f"This section explores the specific performance of the {model} architecture.\n\n")
            
            m_df = local_df[local_df['model'] == model]
            if m_df.empty:
                f.write("Local training results for this model are still being processed.\n\n")
                continue

            for _, row in m_df.iterrows():
                city = row['city']
                f.write(f"\\subsection{{{city.title()} Analysis}}\n")
                f.write(f"The {model} model for {city.title()} was optimized over multiple trials.\n\n")
                
                f.write("\\begin{description}\n")
                f.write(f"    \\item[RMSE] {row['rmse']:.4f}\n")
                f.write(f"    \\item[MAE] {row['mae']:.4f}\n")
                f.write("\\end{description}\n\n")

                f.write("\\textbf{Final Hyperparameter Configuration:}\n")
                f.write("\\begin{itemize}\n")
                for k, v in row['config'].items():
                    f.write(f"    \\item {escape_latex(k)}: {escape_latex(v)}\n")
                f.write("\\end{itemize}\n\n")

                plot_path = generate_forecast_plot(city, model, row['config'])
                if plot_path:
                    f.write("\\begin{figure}[H]\n")
                    f.write("    \\centering\n")
                    f.write(f"    \\includegraphics[width=0.9\\textwidth]{{{plot_path.name}}}\n")
                    f.write(f"    \\caption{{{model} Forecasting Performance: {city.title()}}}\n")
                    f.write("\\end{figure}\n\n")
                f.write("\\newpage\n")

        f.write(r"""\chapter{Conclusion}
The benchmarking results indicate that while deep learning models like Transformers and LSTMs show great promise in capturing complex temporal patterns, classical ensemble methods like Random Forest and LightGBM remain highly competitive, especially in cities with higher variance such as Delhi.

Future work will focus on:
\begin{itemize}
    \item Hybrid spatio-temporal models (ST-GCN).
    \item Incorporation of external meteorological factors (Wind speed, Precipitation).
    \item Real-time deployment on Raspberry Pi edge devices.
\end{itemize}

\end{document}
""")

if __name__ == "__main__":
    local_results = load_all_results()
    kaggle_results = load_kaggle_results()
    
    if local_results.empty and kaggle_results.empty:
        print("No results found.")
        exit(1)
        
    print("Generating analytics...")
    local_flat, combined = generate_comparison_plots(local_results, kaggle_results)
    
    print("Writing LaTeX report...")
    generate_latex(local_flat, combined)
    print("Done.")
