import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, plot_model, pull
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pycaret.classification import *
from imblearn.combine import *


def main():
    st.set_page_config(
    page_title="Crash Severity AutoML",
    page_icon="🖥️",
    layout="centered",
    initial_sidebar_state="expanded")
    st.subheader("**Crash Severity Analysis using Automated ML method**")
    #st.sidebar.title("xxxxx")
    # Initialize session state for best_model
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None


    data_file = st.file_uploader("Upload your CSV", type=["csv"])
    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Data Preview:")
        st.dataframe(data, use_container_width=True)

        all_columns = data.columns.tolist()
        selected_features = st.multiselect("Select Features", all_columns)
        target = st.selectbox("Select Target", all_columns)
        imbalance_method = st.checkbox("Apply Imbalance Handling (SMOTE)")

        if st.button("Run Model Comparison"):
            X = data[selected_features]
            y = data[target]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2023)

            if imbalance_method:
                smote = SMOTE(random_state=2023)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            clf1 = setup(pd.concat([X_train, y_train], axis=1), target=target, session_id=123, 
                         normalize=True, transformation=True, fix_imbalance=imbalance_method)

            st.session_state.best_model = compare_models(sort='F1')
            results = pull()
            results = results.set_index('Model')
            highlighted_df = results.style.highlight_max(axis=0, subset=results.columns[0:])
            st.write("Comparison Results:")
            st.dataframe(highlighted_df, use_container_width=True)
            st.session_state.model = create_model(st.session_state.best_model)
            img = plot_model(st.session_state.model, plot="feature", display_format="streamlit", save=True)
            st.image(img)
            img2 = plot_model(st.session_state.model, plot='confusion_matrix',  plot_kwargs = {'percent' : True}, display_format="streamlit", save=True)
            st.image(img2)
            img3 = plot_model(st.session_state.model, plot='auc', display_format="streamlit", save=True )
            st.image(img3)
            img4 = plot_model(st.session_state.model, plot='pr',display_format="streamlit", save=True )
            st.image(img4)
            img5 = plot_model(st.session_state.model, plot='error',display_format="streamlit", save=True )
            st.image(img5)
            img6 = plot_model(st.session_state.model, plot='class_report',display_format="streamlit", save=True )
            st.image(img6)
            img7 = plot_model(st.session_state.model, plot='learning',display_format="streamlit", save=True )
            st.image(img7)
            img8 = plot_model(st.session_state.model, plot='vc',display_format="streamlit", save=True )
            st.image(img8)

if __name__ == "__main__":
    main()



