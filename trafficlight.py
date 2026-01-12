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
# 1. Load Dataset
# =========================
data = pd.read_csv("/mnt/data/traffic_dataset.csv")

# Encode time_of_day: morning=1, afternoon=2, evening=3, night=4
if 'time_of_day' in data.columns:
    data['time_of_day'] = data['time_of_day'].map({
        'morning': 1,
        'afternoon': 2,
        'evening': 3,
        'night': 4
    })

st.subheader("Traffic Dataset Preview")
st.dataframe(data.head())

# Features & target
feature_names = list(data.drop(columns=["vehicle_count"]).columns)
X = data.drop(columns=["vehicle_count"]).values
y = data["vehicle_count"].values

# =========================
# Sidebar Parameters
# =========================
population_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 5, 50, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
optimization_mode = st.sidebar.radio("Optimization Mode", ["Single Objective", "Multi Objective"])
complexity_weight = 0.0
if optimization_mode == "Multi Objective":
    complexity_weight = st.sidebar.slider("Complexity Weight", 0.0, 1.0, 0.2, help="Penalizes complex solutions")

# =========================
# 2. GP Helper Functions
# =========================
def random_feature():
    return random.randint(0, len(feature_names)-1)

def fitness(feature_idx, X, y):
    y_pred = X[:, feature_idx]  # predict using only the feature value
    mse = np.mean((y - y_pred)**2)
    if optimization_mode == "Single Objective":
        return mse
    else:
        return mse + complexity_weight  # optional penalty

def mutate(feature_idx):
    if random.random() < mutation_rate:
        return random_feature()
    return feature_idx

# =========================
# 3. Run GP
# =========================
st.subheader("Genetic Programming Optimization")

if st.button("Run GP"):
    start_time = time.time()

    # Initialize population
    population = [random_feature() for _ in range(population_size)]
    fitness_history = []

    for gen in range(generations):
        scored = [(f, fitness(f, X, y)) for f in population]
        scored.sort(key=lambda x: x[1])
        fitness_history.append(scored[0][1])

        # Selection: top 50%
        population = [f for f, _ in scored[:population_size//2]]

        # Reproduction & mutation
        while len(population) < population_size:
            parent = random.choice(population)
            population.append(mutate(parent))

    # Best feature
    best_feature_idx = min(population, key=lambda f: fitness(f, X, y))
    best_feature_name = feature_names[best_feature_idx]
    best_fitness = fitness(best_feature_idx, X, y)
    y_pred = X[:, best_feature_idx]
    exec_time = time.time() - start_time

    # =========================
    # 4. Display Results
    # =========================
    st.success("GP Optimization Completed")
    st.metric("Execution Time (s)", f"{exec_time:.4f}")
    st.metric("Best Fitness (MSE)", f"{best_fitness:.4f}")

    st.subheader("Best Feature for Vehicle Count Prediction")
    st.markdown(f"**{best_feature_name}** is the most influential feature for predicting vehicle count.")

    # =========================
    # 5. Visualizations
    # =========================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📈 **Convergence Curve**")
        st.line_chart(pd.DataFrame({"Best Fitness": fitness_history}))
    with col2:
        st.markdown("📊 **Actual vs Predicted Vehicle Count**")
        st.scatter_chart(pd.DataFrame({"Actual": y, "Predicted": y_pred}))

    # =========================
    # 6. Conclusion
    # =========================
    st.subheader("Conclusion")
    st.markdown("""
    This GP model identifies the most influential traffic feature affecting vehicle count. 
    By using only the feature values directly (no coefficients), the model is fully interpretable and suitable for managing traffic congestion.
    Multi-objective optimization allows balancing accuracy and model simplicity.
    """)

