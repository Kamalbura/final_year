# Script to append chapters to thesis
content = '''

% Chapter 2: System Analysis
\\chapter{System Analysis}

\\section{Introduction}

System analysis is the process of gathering and interpreting facts, diagnosing problems, and using the information to recommend improvements in the system. This chapter provides a comprehensive analysis of the air quality forecasting system.

\\section{System Requirements}

\\subsection{Functional Requirements}

\\begin{enumerate}
    \\item Data Collection: Fetch hourly air quality data from Open-Meteo API
    \\item Data Preprocessing: Handle missing values and normalize features
    \\item Model Training: Support 17 different forecasting models
    \\item Model Evaluation: Calculate RMSE, MAE, and R\\textsuperscript{2} metrics
    \\item Edge Deployment: Deploy on Raspberry Pi with ONNX
\\end{enumerate}

\\section{Hardware Requirements}

\\begin{table}[h]
\\centering
\\caption{Hardware Requirements}
\\begin{tabular}{lll}
\\hline
\\textbf{Component} & \\textbf{Development} & \\textbf{Edge} \\\\
\\hline
CPU & Intel i5-12450H & ARM Cortex-A72 \\\\
GPU & RTX 2050 (4GB) & None \\\\
RAM & 16 GB & 8 GB \\\\
Storage & 512 GB SSD & 64 GB microSD \\\\
\\hline
\\end{tabular}
\\end{table}

% Chapter 3: System Design  
\\chapter{System Design}

\\section{Detailed Model Architectures}

\\subsection{Random Forest Architecture}

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.

\\textbf{Mathematical Formulation}:
\\begin{equation}
    \\hat{y} = \\frac{1}{B}\\sum_{b=1}^{B} T_b(x)
\\end{equation}

where $B$ is the number of trees and $T_b$ is the $b$-th decision tree.

\\textbf{Key Parameters}:
\\begin{itemize}
    \\item n\\_estimators: 200 trees
    \\item max\\_depth: 20 levels
    \\item min\\_samples\\_split: 5
\\end{itemize}

\\subsection{LSTM Architecture}

Long Short-Term Memory networks use gating mechanisms to control information flow:

\\begin{align}
    f_t &= \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f) \\\\
    i_t &= \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) \\\\
    C_t &= f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t
\\end{align}

\\subsection{Transformer Architecture}

The Transformer uses multi-head self-attention:

\\begin{equation}
    \\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
\\end{equation}

% Chapter 4: Results
\\chapter{Results}

\\section{Model Performance}

Table shows the complete results:

\\begin{table}[h]
\\centering
\\caption{Model Performance}
\\begin{tabular}{lccc}
\\hline
\\textbf{Model} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{R\\textsuperscript{2}} \\\\
\\hline
Random Forest & 15.58 & 11.79 & 0.572 \\\\
Transformer & 17.34 & 12.50 & 0.471 \\\\
GRU & 17.62 & 13.26 & 0.452 \\\\
\\hline
\\end{tabular}
\\end{table}

% Chapter 5: Conclusion
\\chapter{Conclusion}

This project successfully benchmarked 17 models and deployed on Raspberry Pi 4.

% References
\\begin{thebibliography}{99}
\\bibitem{rf} L. Breiman, ``Random Forests,'' Machine Learning, 2001.
\\bibitem{lstm} S. Hochreiter, ``Long Short-Term Memory,'' Neural Computation, 1997.
\\bibitem{transformer} A. Vaswani, ``Attention Is All You Need,'' NIPS, 2017.
\\end{thebibliography}

\\end{document}
'''

with open('final_report/main_v2.tex', 'a') as f:
    f.write(content)

print("Chapters appended successfully!")
