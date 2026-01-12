import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# =========================
# Page Config
# =========================
st.set_page_config(page_title="GP Traffic Light Optimization", layout="wide")
st.title("🚦 Traffic Light Optimization using Genetic Programming (GP)")
st.markdown("JIE 42903 - Evolutionary Computing (Lab Report and Project)")

# =========================
# 1. Case Study Selection: Load Dataset
# =========================
st.subheader("Traffic Dataset Preview")
data = pd.read_csv("traffic_dataset.csv")

# Encode categorical features
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].astype("category").cat.codes

st.dataframe(data.head())

# Features and target
feature_names = list(data.drop(columns=["vehicle_count"]).columns)
X = data.drop(columns=["vehicle_count"]).values
y = data["vehicle_count"].values

# =========================
# Sidebar: GP Parameters
# =========================
st.sidebar.subheader("GP Parameters")
population_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 5, 100, 30)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
coef_range = st.sidebar.slider("Coefficient Range (±)", 0.5, 5.0, 2.0)
bias_range = st.sidebar.slider("Bias Range (±)", 1.0, 10.0, 5.0)
optimization_mode = st.sidebar.radio("Optimization Mode", ["Single Objective", "Multi Objective"])
complexity_weight = 0.0
if optimization_mode == "Multi Objective":
    complexity_weight = st.sidebar.slider("Complexity Weight", 0.0, 1.0, 0.2, help="Penalty for complex coefficients")

# =========================
# 2. Genetic Programming Functions
# =========================
def random_individual():
    feature_idx = random.randint(0, len(feature_names)-1)
    coef = random.uniform(-coef_range, coef_range)
    bias = random.uniform(-bias_range, bias_range)
    return (coef, feature_idx, bias)

def predict(expr, X):
    coef, feature_idx, bias = expr
    return coef * X[:, feature_idx] + bias

def fitness(expr, X, y):
    y_pred = predict(expr, X)
    mse = np.mean((y - y_pred)**2)
    if optimization_mode == "Single Objective":
        return mse
    else:
        return mse + complexity_weight * abs(expr[0])

def mutate(expr):
    coef, feature_idx, bias = expr
    if random.random() < mutation_rate:
        feature_idx = random.randint(0, len(feature_names)-1)
    coef += random.uniform(-0.2*coef_range, 0.2*coef_range)
    bias += random.uniform(-0.2*bias_range, 0.2*bias_range)
    return (coef, feature_idx, bias)

# =========================
# 5. Streamlit: Run GP
# =========================
st.subheader("Run Genetic Programming Optimization")
if st.button("Run GP"):
    start_time = time.time()
    population = [random_individual() for _ in range(population_size)]
    fitness_history = []

    for gen in range(generations):
        # Evaluate fitness
        scored = [(ind, fitness(ind, X, y)) for ind in population]
        scored.sort(key=lambda x: x[1])
        fitness_history.append(scored[0][1])

        # Selection: top 50%
        population = [ind for ind, _ in scored[:population_size//2]]

        # Reproduction & Mutation
        while len(population) < population_size:
            parent = random.choice(population)
            population.append(mutate(parent))

    # Best individual
    best_expr = min(population, key=lambda e: fitness(e, X, y))
    best_coef, best_feature_idx, best_bias = best_expr
    best_feature_name = feature_names[best_feature_idx]
    best_fitness = fitness(best_expr, X, y)
    y_pred = predict(best_expr, X)
    exec_time = time.time() - start_time

    # =========================
    # 3. Performance Analysis
    # =========================
    st.success("GP Optimization Completed")
    st.metric("Execution Time (s)", f"{exec_time:.4f}")
    st.metric("Best Fitness", f"{best_fitness:.4f}")

    st.subheader("Best Mathematical Model")
    st.code(f"vehicle_count = {best_coef:.2f} × {best_feature_name} + {best_bias:.2f}")

    # =========================
    # Visualization: Convergence & Accuracy
    # =========================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📈 **Convergence Curve**")
        st.line_chart(pd.DataFrame({"Best Fitness": fitness_history}))
    with col2:
        st.markdown("📊 **Actual vs Predicted Vehicle Count**")
        st.scatter_chart(pd.DataFrame({"Actual": y, "Predicted": y_pred}))

    # =========================
    # 4. Extended Analysis
    # =========================
    st.subheader("Extended Analysis")
    st.markdown(f"""
    - Optimization Mode: **{optimization_mode}**
    - Multi-objective optimization penalizes high coefficient values to produce simpler models.
    - The best feature affecting vehicle count: **{best_feature_name}**
    - Convergence curve shows early rapid improvement followed by slower refinement, typical of GP.
    """)

    # =========================
    # Conclusion
    # =========================
    st.subheader("Conclusion")
    st.markdown("""
    The Genetic Programming model successfully predicts vehicle count for traffic light optimization.
    Multi-objective GP balances prediction accuracy and interpretability. The system identifies the
    most influential traffic feature and generates an easy-to-understand linear model suitable for real-world traffic management.
    """)
