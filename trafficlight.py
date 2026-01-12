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

# Encode categorical features
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

# =========================
# GP Functions
# =========================
def random_individual():
    # Individual = selected feature index
    return random.randint(0, len(feature_names) - 1)

def predict(feature_idx, X):
    return X[:, feature_idx]

def fitness(feature_idx, X, y):
    y_pred = predict(feature_idx, X)
    return np.mean((y - y_pred) ** 2)

def mutate(feature_idx):
    if random.random() < mutation_rate:
        return random.randint(0, len(feature_names) - 1)
    return feature_idx

# =========================
# Run GP
# =========================
st.subheader("Genetic Programming Optimization Results")

if st.button("Run GP"):
    start_time = time.time()

    # Initialize population
    population = [random_individual() for _ in range(population_size)]
    fitness_history = []

    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness
        scored = [(ind, fitness(ind, X, y)) for ind in population]
        scored.sort(key=lambda x: x[1])
        fitness_history.append(scored[0][1])

        # Selection: top 50%
        population = [ind for ind, _ in scored[:population_size // 2]]

        # Reproduction & Mutation
        while len(population) < population_size:
            parent = random.choice(population)
            population.append(mutate(parent))

    # Extract best individual
    best_feature_idx = min(population, key=lambda i: fitness(i, X, y))
    best_feature_name = feature_names[best_feature_idx]
    best_mse = fitness(best_feature_idx, X, y)
    y_pred = predict(best_feature_idx, X)

    exec_time = time.time() - start_time

    # =========================
    # Output Results
    # =========================
    st.success("GP Optimization Completed")
    st.metric("Execution Time (s)", f"{exec_time:.4f}")
    st.metric("Best MSE", f"{best_mse:.4f}")

    st.subheader("Best GP Model")
    st.code(f"vehicle_count ≈ {best_feature_name}")

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
        "This simplified Genetic Programming model selects the traffic feature most strongly associated "
        "with vehicle count. It provides a clear and interpretable model for traffic light optimization, "
        "supports real-time decision-making, and aligns with intelligent traffic management objectives."
    )
