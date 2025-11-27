#bash 
pip install mealpy tensorflow pandas numpy matplotlib seaborn scikit-learn openpyxl
# Slope Stability Prediction - ANN + MealPy Optimizer
# Clean, readable version with clear optimizer menu

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from mealpy import FloatVar
from mealpy.swarm_based import OriginalGWO, OriginalWOA, OriginalPSO, OriginalABC, OriginalHHO, OriginalMFO
from mealpy.evolutionary_based import BaseGA, OriginalDE
from mealpy.human_based import OriginalTLO
from mealpy.physics_based import OriginalSA, OriginalMVO, OriginalEO

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Config
DATA_FILE = "rs2_srf.xlsx"
TARGET = "SRF"
POP_SIZE = 8
MAX_ITER = 20
EPOCHS_SEARCH = 100
EPOCHS_FINAL = 500

# Search bounds: [n1, n2, n3, dropout, lr, l2, batch]
lb = [32,  64,  32, 0.1, 0.0001, 0.0001, 16]
ub = [256, 512, 256, 0.5, 0.01,   0.01,   64]

# Load and preprocess data
df = pd.read_excel(DATA_FILE)
df['Pk_Tn_Str'] = pd.to_numeric(df['Pk_Tn_Str'], errors='coerce').fillna(df['Pk_Tn_Str'].median())
df.dropna(subset=['Slp_Angl','Chsn','Sl_thick','Unit_Wt_Sl','mb','s','Slp_Ht','Ang_Frn','SRF','Yng_Sl'], inplace=True)

df['Sl_thick'] = np.log1p(df['Sl_thick'] + 0.01)
df['SRF'] = np.log1p(df['SRF'] + 0.05)

features = ['Slp_Angl','Chsn','Sl_thick','Unit_Wt_Sl','mb','s','Slp_Ht','Ang_Frn','Yng_Sl']
optional = ['Psns_Rck','Yng_Mdls_Rck','Cmp_Str']
features += [f for f in optional if f in df.columns]

# Remove highly correlated features
corr = df[features].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
to_drop = [col for col in corr.columns if any(corr.where(mask)[col] > 0.8)]
features = [f for f in features if f not in to_drop]

X = StandardScaler().fit_transform(df[features])
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Sample weights: higher penalty for low SRF values
sample_weights = np.where(y < 1.0, 15.0, np.where(y <= 3.0, 1.5, 1.0))

# CHANGE ONLY THIS LINE TO SWITCH OPTIMIZER
Optimizer = OriginalGWO   # ← Replace with any from the menu above😊😊😊change HERE!!!!!!!! from the MENu below
# === OPTIMIZER MENU - Choose one below ===
# Swarm-based:
#   OriginalGWO   - Grey Wolf Optimizer
#   OriginalWOA   - Whale Optimization Algorithm
#   OriginalPSO   - Particle Swarm Optimization
#   OriginalABC   - Artificial Bee Colony
#   OriginalHHO   - Harris Hawks Optimization
#   OriginalMFO   - Moth-Flame Optimization
#
# Evolutionary:
#   BaseGA        - Genetic Algorithm
#   OriginalDE    - Differential Evolution
#
# Human-based:
#   OriginalTLO   - Teaching-Learning-Based Optimization
#
# Physics-based:
#   OriginalSA    - Simulated Annealing
#   OriginalMVO   - Multi-Verse Optimizer
#   OriginalEO    - Equilibrium Optimizer

# Model builder
def build_model(params):
    n1, n2, n3 = int(params[0]), int(params[1]), int(params[2])
    drop, lr, l2_reg = params[3], params[4], params[5]
    batch = int(params[6])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), input_dim=X.shape[1]),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(n2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(n3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(1)
    ])
    model.compile(tf.keras.optimizers.Adam(lr), loss='mse')
    return model, batch

# Fitness function
def fitness(solution):
    model, batch = build_model(solution)
    model.fit(X_tr, y_tr,
              sample_weight=sample_weights[:len(y_tr)],
              validation_data=(X_val, y_val),
              epochs=EPOCHS_SEARCH, batch_size=batch, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    pred = model.predict(X_val, verbose=0).flatten()
    tf.keras.backend.clear_session()
    return mean_squared_error(y_val, pred)

# Run optimization
problem = {"bounds": FloatVar(lb=lb, ub=ub), "minmax": "min", "obj_func": fitness}
opt = Optimizer(epoch=MAX_ITER, pop_size=POP_SIZE)
best = opt.solve(problem)

# Results
best_params = best.solution
print(f"\nOptimizer: {Optimizer.__name__}")
print(f"Best Val MSE: {best.target.fitness:.6f}")
print(f"Architecture: {int(best_params[0])}-{int(best_params[1])}-{int(best_params[2])}")
print(f"Dropout: {best_params[3]:.3f} | LR: {best_params[4]:.6f} | Batch: {int(best_params[6])}")

# Final training
final_model, final_batch = build_model(best_params)
final_model.fit(X_train, y_train,
                sample_weight=sample_weights[len(y_test):],
                validation_data=(X_test, y_test),
                epochs=EPOCHS_FINAL, batch_size=final_batch,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
                verbose=1)

pred_test = final_model.predict(X_test).flatten()
print(f"\nFinal Test R²: {r2_score(y_test, pred_test):.4f}")
print(f"Final Test RMSE: {np.sqrt(mean_squared_error(y_test, pred_test)):.4f}")

# Baseline
lr_r2 = r2_score(y_test, LinearRegression().fit(X_train, y_train).predict(X_test))
print(f"Linear Regression R²: {lr_r2:.4f}")

print("\nTo try another optimizer, change the 'Optimizer =' line and rerun.")
