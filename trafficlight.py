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
# Load Dataset
# =========================
data = pd.read_csv("traffic_dataset.csv")

# Encode categorical features if needed
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].astype("category").cat.codes

st.subheader("Traffic Dataset Preview")
st.dataframe(data.head())

# =========================
# Features & Target
# =========================
feature_names = list(data.drop(columns=["vehicle_count"]).columns)
X = data.drop(columns=["vehicle_count"]).values
y = data["vehicle_count"].values

# =========================
# Sidebar Parameters
# =========================
population_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 5, 50, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
coef_range = st.sidebar.slider("Coefficient Range (±)", 0.5, 5.0, 2.0)
bias_range = st.sidebar.slider("Bias Range (±)", 1.0, 10.0, 5.0)

# =========================
# GP Helper Functions
# =========================
def random_individual():
    # Individual = (coef, feature_idx, bias)
    feature_idx = random.randint(0, len(feature_names) - 1)
    coef = random.uniform(-coef_range, coef_range)
    bias = random.uniform(-bias_range, bias_range)
    return (coef, feature_idx, bias)

def predict(expr, X):
    coef, feature_idx, bias = expr
    return coef * X[:, feature_idx] + bias

def fitness(expr, X, y):
    y_pred = predict(expr, X)
    return np.mean((y - y_pred)**2)

def mutate(expr):
    coef, feature_idx, bias = expr
    if random.random() < mutation_rate:
        feature_idx = random.randint(0, len(feature_names) - 1)
    coef += random.uniform(-0.2*coef_range, 0.2*coef_range)
    bias += random.uniform(-0.2*bias_range, 0.2*bias_range)
    return (coef, feature_idx, bias)

# =========================
# Run GP
# =========================
st.subheader("Genetic Programming Optimization Results")

if st.button("Run GP"):
    start_time = time.time()

    # Initialize population
    population = [random_individual() for _ in range(population_size)]
    fitness_history = []

    for gen in range(generations):
        scored = [(ind, fitness(ind, X, y)) for ind in population]
        scored.sort(key=lambda x: x[1])
        fitness_history.append(scored[0][1])

        # Selection: top 50%
        population = [ind for ind, _ in scored[:population_size // 2]]

        # Reproduction & Mutation
        while len(population) < population_size:
            parent = random.choice(population)
            population.append(mutate(parent))

    # Best individual
    best_expr = min(population, key=lambda e: fitness(e, X, y))
    best_coef, best_feature_idx, best_bias = best_expr
    best_feature_name = feature_names[best_feature_idx]
    best_mse = fitness(best_expr, X, y)
    y_pred = predict(best_expr, X)
    exec_time = time.time() - start_time

    # =========================
    # Results
    # =========================
    st.success("GP Optimization Completed")
    st.metric("Execution Time (s)", f"{exec_time:.4f}")
    st.metric("Best MSE", f"{best_mse:.4f}")

    st.subheader("Best Interpretable Model")
    st.code(f"vehicle_count = {best_coef:.2f} × {best_feature_name} + {best_bias:.2f}")

    # =========================
    # Visualizations
    # =========================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📈 **Convergence Curve**")
        st.line_chart(pd.DataFrame({"Best MSE": fitness_history}))
    with col2:
        st.markdown("📊 **Actual vs Predicted Vehicle Count**")
        st.scatter_chart(pd.DataFrame({"Actual": y, "Predicted": y_pred}))

    # =========================
    # Conclusion
    # =========================
    st.subheader("Conclusion")
    st.markdown(
        "This GP model generates a simple linear formula linking vehicle count to the most relevant traffic feature. "
        "It provides interpretable output suitable for traffic light optimization and decision-making in intelligent traffic systems."
    )
