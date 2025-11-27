# SlopeStability-ANN-MealPy

ANN-based Slope Stability Factor (SRF) prediction optimized using 12 powerful metaheuristic algorithms from the MealPy library.

Switch between optimizers with just one line - perfect for comparing performance on geotechnical regression problems.

Features
- Clean, well-commented, single-file script
- 12 ready-to-use optimizers (change one line):
  - Swarm-based: GWO, WOA, PSO, ABC, HHO, MFO
  - Evolutionary: GA, DE
  - Human-based: TLBO
  - Physics-based: SA, MVO, EO
- Automatic hyperparameter search: neurons, dropout, learning rate, L2 regularization, batch size
- Smart preprocessing: log transform, correlation filtering, sample weighting (penalizes low SRF)
- Final model training with early stopping
- Evaluation: R², RMSE, MAE, MAPE, VAF
- Linear Regression baseline comparison
- Saves: model (.h5), predictions, hyperparameters, convergence, plots

Requirements
pandas
numpy
scikit-learn
tensorflow (>=2.10)
mealpy (>=3.0)
matplotlib
seaborn
openpyxl

Install with:
pip install pandas numpy scikit-learn tensorflow mealpy matplotlib seaborn openpyxl

How to Use

1. Place your dataset as rs2_srf.xlsx in the repository root
   (Must contain columns: SRF, Slp_Angl, Chsn, Sl_thick, Unit_Wt_Sl, mb, s, Slp_Ht, Ang_Frn, Yng_Sl)

2. Open the script and change one line:
   Optimizer = OriginalGWO   # Change to OriginalWOA, OriginalHHO, OriginalEO, etc.

3. Run:
   python slope_ann_mealpy.py

4. Results (plots, CSVs, model) will be saved automatically.

Output Files
- model_[Optimizer].h5 - Trained ANN
- best_hyperparameters_[Optimizer].csv
- predictions_[Optimizer].csv
- convergence_[Optimizer].png
- actual_vs_predicted_[Optimizer].png
- and more...

Reference
If you use this code in research, please cite:
Your Name. (2025). SlopeStability-ANN-MealPy: Metaheuristic-Optimized ANN for Slope Stability Prediction. GitHub Repository. https://github.com/yourusername/SlopeStability-ANN-MealPy

One script. 12 optimizers. Instant comparison.
