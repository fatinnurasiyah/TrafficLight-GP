import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Traffic Optimization with GP", layout="wide")
st.title("🚦 Traffic Light Optimization using Genetic Programming (GP)")
st.markdown("Predict vehicle count and optimize traffic flow using GP with linear models.")

# =========================
# Upload Dataset
# =========================
st.subheader("Upload Traffic Dataset (CSV)")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Encode time_of_day if exists
    if 'time_of_day' in data.columns:
        data['time_of_day'] = data['time_of_day'].map({
            'morning': 1,
            'afternoon': 2,
            'evening': 3,
            'night': 4
        })

    st.subheader("Traffic Dataset Preview")
    st.dataframe(data.head())

    # -------------------------
    # Select Target Variable
    # -------------------------
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    target_column = st.selectbox("Select target variable (numerical)", numeric_cols)
    st.markdown(f"**Selected target:** `{target_column}`")

    # Features & target
    feature_names = [col for col in numeric_cols if col != target_column]
    X = data[feature_names].values
    y = data[target_column].values

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
    # GP Helper Functions (same as before)
    # =========================
    def random_feature():
        return random.randint(0, len(feature_names)-1)

    def random_a_b():
        a = random.uniform(-5, 5)
        b = random.uniform(-50, 50)
        return a, b

    def fitness(individual, X, y):
        feature_idx, a, b = individual
        y_pred = a * X[:, feature_idx] + b
        mse = np.mean((y - y_pred) ** 2)
        if optimization_mode == "Single Objective":
            return mse
        else:
            return mse + complexity_weight * abs(a)

    def mutate(individual):
        feature_idx, a, b = individual
        if random.random() < mutation_rate:
            feature_idx = random_feature()
        if random.random() < mutation_rate:
            a += random.uniform(-0.5, 0.5)
        if random.random() < mutation_rate:
            b += random.uniform(-5, 5)
        return [feature_idx, a, b]

    # =========================
    # Run GP
    # =========================
    st.subheader("Genetic Programming Optimization")

    if st.button("Run GP"):
        start_time = time.time()

        # Initialize population: [feature_idx, a, b]
        population = [[random_feature(), *random_a_b()] for _ in range(population_size)]
        fitness_history = []

        for gen in range(generations):
            scored = [(ind, fitness(ind, X, y)) for ind in population]
            scored.sort(key=lambda x: x[1])
            fitness_history.append(scored[0][1])

            # Selection: top 50%
            population = [ind for ind, _ in scored[:population_size // 2]]

            # Reproduction & mutation
            while len(population) < population_size:
                parent = random.choice(population)
                population.append(mutate(parent))

        # Best individual
        best_individual = min(population, key=lambda ind: fitness(ind, X, y))
        feature_idx, a, b = best_individual
        best_feature_name = feature_names[feature_idx]
        best_fitness = fitness(best_individual, X, y)
        y_pred = a * X[:, feature_idx] + b
        exec_time = time.time() - start_time

        # =========================
        # Results
        # =========================
        st.success("GP Optimization Completed")
        st.metric("Execution Time (s)", f"{exec_time:.4f}")
        st.metric("Best Fitness (MSE)", f"{best_fitness:.4f}")

        st.subheader(f"Best Feature for Predicting `{target_column}`")
        st.markdown(f"**{best_feature_name}** is the most influential feature.")
        st.markdown(f"Mathematical Model: ` {target_column} = {a:.2f} × {best_feature_name} + {b:.2f} `")

        # =========================
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("📈 **Convergence Curve**")
            st.line_chart(pd.DataFrame({"Best Fitness": fitness_history}))
        with col2:
            st.markdown("📊 **Actual vs Predicted**")
            st.scatter_chart(pd.DataFrame({"Actual": y, "Predicted": y_pred}))

        # =========================
        # Conclusion
        st.subheader("Conclusion")
        st.markdown(f"""
        This GP model identifies the most influential numerical feature affecting `{target_column}` 
        and generates a simple linear model: `{target_column} = a * feature + b`.
        Multi-objective optimization balances accuracy and simplicity.
        """)
else:
    st.info("Please upload a CSV file to start GP optimization.")

