import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from supertree import SuperTree  # SuperTree expects the model object and extracts what it needs

def train_model(max_depth):
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    clf.fit(X, y)
    return clf, X, y, iris.feature_names, list(iris.target_names), iris

def inject_decision_path_css(html_content, clf, sample):
    """
    Compute the decision path for the test sample and inject CSS rules into the HTML
    to highlight nodes along the path. We assume that nodes in the HTML are identified by
    an id of the form "nodeX" where X is the node id.
    """
    # Ensure sample is 2D.
    sample = np.array(sample).reshape(1, -1)
    # Compute the decision path; nonzero returns (row_indices, col_indices).
    dp = clf.decision_path(sample)
    node_ids = dp.nonzero()[1]  # get the column indices (node ids)
    # Create CSS rules that highlight each node.
    # Adjust the CSS as needed; here we set an orange border.
    css_rules = "\n".join([f"#node{node} {{ border: 2px solid orange !important; }}" for node in node_ids])
    css_block = f"<style>{css_rules}</style>"
    # Prepend the CSS block to the HTML content.
    return css_block + html_content

def generate_viz(clf, X, y, feature_names, class_names, sample=None):
    # Create a SuperTree object.
    super_tree = SuperTree(clf, X, y, feature_names, class_names)
    html_filename = "supertree.html"
    # If a sample is provided, attempt to highlight the decision path.
    if sample is not None:
        # NOTE: The 'show_sample' parameter is premium, so we avoid it.
        # Instead we just generate the default visualization.
        super_tree.save_html(html_filename)
    else:
        super_tree.save_html(html_filename)
    with open(html_filename, "r", encoding="utf-8") as f:
        html_content = f.read()
    # If a sample is provided, inject CSS to highlight the decision path.
    if sample is not None:
        html_content = inject_decision_path_css(html_content, clf, sample)
    return html_content

def show_data_visualization(df, feature_names):
    st.subheader("Initial Data Visualization")
    # Create an interactive scatter matrix using Plotly Express.
    fig = px.scatter_matrix(
        df,
        dimensions=feature_names,
        color="target_name",
        hover_data=df.columns
    )
    # Adjust layout to minimize overlap.
    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        autosize=True
    )
    # Rotate x-axis tick labels in each subplot.
    for axis in fig.layout:
        if axis.startswith("xaxis"):
            fig.layout[axis].tickangle = -45
    st.plotly_chart(fig, use_container_width=True)

def main():
    HORIZONTAL = "logo.png"
    ICON = "icon.png"

    st.logo(HORIZONTAL, icon_image=ICON)

    st.set_page_config(
        page_title="Krisolis Supervised Learning Demonstration",
        page_icon=ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': """# Krisolis Supervised Learning Demonstration. 
            Created by Eoghan Staunton @Krisolis
            Using scikit-learn and supertree"""
        })

    st.title("🌳 Supervised Learning Demonstration with Decision Trees")
    st.markdown("This app demonstrates a Supervised Machine Learning Model called a **Decision Tree Classifier** on the Iris dataset.", 
             help="""The Iris dataset is a classic and widely utilized resource in the fields of statistics and machine learning. 
             Introduced by British statistician and biologist Ronald Fisher in his 1936 paper 
             "The use of multiple measurements in taxonomic problems," the dataset comprises 150 samples of iris flowers, divided equally among three species: 
             Iris setosa, Iris versicolor, and Iris virginica. Each sample includes four features: sepal length, sepal width, petal length, and petal width, 
             all measured in centimeters.
             
             Due to its simplicity and well-structured nature, the Iris dataset has become a standard test case for various statistical classification techniques and machine learning algorithms. """)
    st.markdown("Use the controls in the sidebar to train a model. Once it is trained you can use it to make predictions.")

    
    
    # Load the Iris dataset and create a DataFrame.
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    mapping = {i: name for i, name in enumerate(iris.target_names)}
    df["target_name"] = df["target"].map(mapping)
    
    # Sidebar: all user inputs.
    st.sidebar.subheader("Model Parameters")
    max_depth = st.sidebar.slider("Select the number of levels (max_depth)",
                                  min_value=1, max_value=10, value=3, step=1)
    
    # Button to train the model.
    if st.sidebar.button("Train and Visualize Model"):
        st.write(f"Training a Decision Tree with max_depth = {max_depth} ...")
        clf, X, y, feature_names, class_names, iris_obj = train_model(max_depth)
        st.session_state["trained_model"] = clf
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["feature_names"] = feature_names
        st.session_state["class_names"] = class_names
        st.session_state["iris_obj"] = iris_obj
        # Initially, generate visualization without a test sample.
        st.session_state["viz_html"] = generate_viz(clf, X, y, feature_names, class_names)
        st.session_state["model_trained"] = True

    # Before the model is trained, show the interactive dataset visualization.
    if not st.session_state.get("model_trained", False):
        show_data_visualization(df, iris.feature_names)
        st.sidebar.info("Train a model to see the decision tree visualization and to input a new observation.")
    else:
        # Once the model is trained, display the SuperTree visualization.
        st.subheader("Trained Decision Tree Visualization")
        st.components.v1.html(st.session_state["viz_html"], height=800, scrolling=True)
        
        # In the sidebar, display new observation inputs.
        st.sidebar.subheader("Test New Observation")
        obs_sepal_length = st.sidebar.number_input("Sepal Length (cm)", value=5.8, key="obs_sepal_length")
        obs_sepal_width  = st.sidebar.number_input("Sepal Width (cm)",  value=3.0, key="obs_sepal_width")
        obs_petal_length = st.sidebar.number_input("Petal Length (cm)", value=4.35, key="obs_petal_length")
        obs_petal_width  = st.sidebar.number_input("Petal Width (cm)",  value=1.3, key="obs_petal_width")
        
        # Only allow prediction if the model is trained.
        if st.sidebar.button("Predict Observation", key="predict_obs"):
            test_sample = np.array([obs_sepal_length, obs_sepal_width, obs_petal_length, obs_petal_width]).reshape(1, -1)
            clf = st.session_state["trained_model"]
            pred = clf.predict(test_sample)[0]
            prob = clf.predict_proba(test_sample)[0]
            st.subheader("🔮 Prediction Result")
            st.write(f"**Prediction:** {st.session_state['class_names'][pred]}")
            st.write("**Class Probabilities:**")
            for cls, p in zip(st.session_state["class_names"], prob):
                st.write(f"- {cls}: {p:.2f}")
            # Regenerate the visualization with the test sample to highlight the decision path.
            st.session_state["viz_html"] = generate_viz(
                clf, st.session_state["X"], st.session_state["y"],
                st.session_state["feature_names"], st.session_state["class_names"],
                sample=list(test_sample[0])
            )
            st.subheader("Trained Decision Tree Decision Path Visualization")
            st.components.v1.html(st.session_state["viz_html"], height=800, scrolling=True)

if __name__ == "__main__":
    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    main()
