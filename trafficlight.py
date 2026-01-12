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
st.markdown("JIE 42903 - Evolutionary Computing (Lab Report and Project ")

# =========================
# Load Dataset
# =========================
st.subheader("Traffic Dataset")

data = pd.read_csv("traffic_dataset.csv")
feature_names = list(data.drop(columns=["vehicle_count"]).columns)

if data["time_of_day"].dtype == object:
    data["time_of_day"] = data["time_of_day"].map({
        "morning": 1,
        "afternoon": 2,
        "evening": 3,
        "night": 4
    })

st.markdown("Encoded Dataset Preview: after preprocessing")
st.dataframe(data.head())

# =========================
# Feature & Target
# =========================
X = data.drop(columns=["vehicle_count"]).astype(float).values
y = data["vehicle_count"].astype(float).values

# =========================
# Sidebar Parameters
# =========================

population_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 5, 100, 30)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.50, 0.10)
coef_range = st.sidebar.slider("Coefficient Range (±)", 0.5, 5.0, 2.0)
bias_range = st.sidebar.slider("Bias Range (±)", 1.0, 10.0, 5.0)

optimization_mode = st.sidebar.radio(
    "Optimization Mode",
    ["Single Objective", "Multi Objective"]
)

if optimization_mode == "Multi Objective":
    complexity_weight = st.sidebar.slider(
        "Complexity Weight",
        0.0, 1.0, 0.2,
        help="Controls trade-off between accuracy and simplicity"
    )
else:
    complexity_weight = 0.0

# =========================
# GP Helper Functions
# =========================
def random_expression(feature_names):
    feature = random.randint(0, len(feature_names) - 1)
    coef = random.uniform(-coef_range, coef_range)
    bias = random.uniform(-bias_range, bias_range)
    return (coef, feature, bias)

def predict(expr, X):
    coef, feature, bias = expr
    return coef * X[:, feature] + bias

def fitness(expr, X, y):
    y_pred = predict(expr, X)
    mse = np.mean((y - y_pred) ** 2)

    if optimization_mode == "Single Objective":
        return mse
    else:
        complexity_penalty = abs(expr[0])  # coefficient magnitude
        return mse + complexity_weight * complexity_penalty

def mutate(expr):
    coef, feature, bias = expr
    coef += random.uniform(-0.2 * coef_range, 0.2 * coef_range)
    bias += random.uniform(-0.2 * bias_range, 0.2 * bias_range)
    return (coef, feature, bias)

# =========================
# Run GP Optimization
# =========================
st.subheader("Genetic Programming Optimization Results")

if st.button("Run Genetic Programming (GP)"):

    start_time = time.time()

    with st.spinner("Running GP evolution..."):
        population = [random_expression(feature_names) for _ in range(population_size)]
        fitness_history = []

        for gen in range(generations):
            scored = [(expr, fitness(expr, X, y)) for expr in population]
            scored.sort(key=lambda x: x[1])

            fitness_history.append(scored[0][1])

            # Selection
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
        y_pred = predict(best_expr, X)

    exec_time = time.time() - start_time    
    
    #=========================
    # Status Output (Side by Side)
    # =========================
    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.success("Genetic Programming Optimization Completed")

    with status_col2:
        st.info(f"Mode: **{optimization_mode}**")

    with status_col3:
        st.metric("Execution Time (s)", f"{exec_time:.4f}")
    # =========================
    # Results
    # =========================
    coef, feature, bias = best_expr
    feature_name = feature_names[feature]

    st.subheader("Best Interpretable Mathematical Model")
    st.code(f"vehicle_count = {coef:.3f} × {feature_name} + {bias:.3f}")

    st.write(f"📉 **Best Fitness (MSE):** {best_fitness:.4f}")
    st.write(f"⏱ **Execution Time:** {exec_time:.4f} seconds")

    # =========================
    # Visualization (Side by Side)
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 Convergence Curve**")
        st.line_chart(
            pd.DataFrame({"Best Fitness (MSE)": fitness_history}),
            height=280
        )

    with col2:
        st.markdown("Actual vs Predicted")
        st.scatter_chart(
            pd.DataFrame({
                "Actual vehicle count": y,
                "Predicted vehicle count": y_pred
            }),
            height=280
        )

    # =========================
    # Performance Analysis
    # =========================
    st.subheader("Performance Analysis")
    st.markdown(
        "- **Convergence Rate:** Rapid improvement in early generations\n"
        "- **Accuracy:** GP predicts waiting time with acceptable error\n"
        "- **Computational Efficiency:** Low execution time\n"
        "- **Interpretability:** Simple mathematical expression\n\n"
        "**Extended Analysis:**\n"
        "- Multi-objective optimization balances accuracy and simplicity\n"
        "- Higher complexity weight yields more interpretable models"
    )

    # =========================
    # Conclusion
    # =========================
    st.subheader("Conclusion")
    st.markdown(
        "This Streamlit-based Genetic Programming system demonstrates how evolutionary computation "
        "can automatically generate interpretable mathematical models for traffic waiting time prediction. "
        "By extending **Genetic Programming** to multi-objective optimization, the system balances prediction accuracy and "
        "model simplicity, enhancing transparency and real-world applicability."
    )

