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
st.markdown("**Computational Evolution Case Study**")

# =========================
# Load Dataset
# =========================
st.subheader("Traffic Dataset")

data = pd.read_csv("traffic_dataset.csv")

# =========================
# Encode categorical column (time_of_day)
# =========================
if data["time_of_day"].dtype == object:
    data["time_of_day"] = data["time_of_day"].map({
        "morning": 0,
        "afternoon": 1,
        "evening": 2,
        "night": 3
    })

st.markdown("**Encoded Dataset Preview (After Preprocessing):**")
st.dataframe(data.head())

# =========================
# Feature & Target
# =========================
X = data.drop(columns=["waiting_time"]).astype(float).values
y = data["waiting_time"].astype(float).values

feature_names = list(data.drop(columns=["waiting_time"]).columns)

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("GP Parameters")

population_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 5, 200, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.50, 0.10)
coef_range = st.sidebar.slider("Coefficient Range (±)", 0.5, 5.0, 2.0)
bias_range = st.sidebar.slider("Bias Range (±)", 1.0, 10.0, 5.0)

feature_mode = st.sidebar.radio(
    "Feature Usage",
    ["Single Best Feature (Default)", "Random Feature Selection"]
)

# =========================
# GP Helper Functions
# =========================
def random_expression(n_features):
    feature = random.randint(0, n_features - 1)
    coef = random.uniform(-coef_range, coef_range)
    bias = random.uniform(-bias_range, bias_range)
    return (coef, feature, bias)

def predict(expr, X):
    coef, feature, bias = expr
    return coef * X[:, feature] + bias

def fitness(expr, X, y):
    coef, feature, bias = expr
    y_pred = predict(expr, X)
    mse = np.mean((y - y_pred) ** 2)
    complexity_penalty = 0.01 * abs(coef)
    return mse + complexity_penalty

def mutate(expr):
    coef, feature, bias = expr
    coef += random.uniform(-0.2 * coef_range, 0.2 * coef_range)
    bias += random.uniform(-0.2 * bias_range, 0.2 * bias_range)
    return (coef, feature, bias)

if feature_mode == "Single Best Feature (Default)":
    feature = random.randint(0, n_features - 1)
else:
    feature = random.choice(range(n_features))

# =========================
# Run GP Optimization
# =========================
st.subheader("Optimization Results of Genetic Programming ")

if st.button("Run Genetic Programming (GP)"):
    start_time = time.time()

    with st.spinner("Running GP evolution..."):
        population = [random_expression(X.shape[1]) for _ in range(population_size)]
        fitness_history = []

        for gen in range(generations):
            scored = [(expr, fitness(expr, X, y)) for expr in population]
            scored.sort(key=lambda x: x[1])

            best_fitness_gen = scored[0][1]
            fitness_history.append(best_fitness_gen)

            # Selection (top 50%)
            population = [expr for expr, _ in scored[:population_size // 2]]

            # Reproduction
            while len(population) < population_size:
                parent = random.choice(population)
                if random.random() < mutation_rate:
                    population.append(mutate(parent))
                else:
                    population.append(parent)

        best_expr = min(population, key=lambda e: fitness(e, X, y))
        best_fitness = fitness(best_expr, X, y)

    exec_time = time.time() - start_time

    st.success("Result Found")
if st.button("Run Genetic Programming (GP)"):
    start_time = time.time()

    with st.spinner("Running GP evolution..."):
        population = [random_expression(X.shape[1]) for _ in range(population_size)]
        fitness_history = []

        for gen in range(generations):
            scored = [(expr, fitness(expr, X, y)) for expr in population]
            scored.sort(key=lambda x: x[1])

            best_fitness_gen = scored[0][1]
            fitness_history.append(best_fitness_gen)

            population = [expr for expr, _ in scored[:population_size // 2]]

            while len(population) < population_size:
                parent = random.choice(population)
                if random.random() < mutation_rate:
                    population.append(mutate(parent))
                else:
                    population.append(parent)

        best_expr = min(population, key=lambda e: fitness(e, X, y))
        best_fitness = fitness(best_expr, X, y)

    exec_time = time.time() - start_time

    st.success("Result Found")

    coef, feature, bias = best_expr
    feature_name = feature_names[feature]

    st.markdown("**Best Interpretable Mathematical Model Generated:**")
    st.code(f"waiting_time = {coef:.3f} × {feature_name} + {bias:.3f}")

    st.write(f"⏱ Execution Time: {exec_time:.4f} seconds")
    st.write(f"📉 Best Fitness (MSE): {best_fitness:.4f}")

    st.subheader("Conclusion")
    st.markdown(
        "This Streamlit-based Genetic Programming system demonstrates how evolutionary computation "
        "can automatically generate interpretable mathematical models for predicting traffic waiting time."
    )

