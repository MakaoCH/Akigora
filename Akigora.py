import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import json


def main():
    st.image("logo_rvb_horizontal_petit.png")
    st.title("Dashboard")
    st.subheader("Auteur : Mickaël Brothier")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_excel('CollectionExperts.xlsx')
        return data
    #Affichage des données
    df = load_data()
    # Assuming 'non_numeric_column' is the column with non-numeric values

    df = df.dropna(subset=['isFake'])
    if st.sidebar.checkbox("Afficher les données 'Collection Profile'", False):
        st.subheader("Jeu de données 'Collection Profile' : Echantillon")
        st.write(df)

    seed = 123

    #train/test split
    def split(df):
        y = df['isFake']
        X = df.drop("isFake", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=seed
        )

        return X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = split(df)

    classifier = st.sidebar.selectbox(
        "Classificateur",
        ("Random forest", "SVM", "Regression Logistic")
    )

    if classifier == "Random forest":
        st.sidebar.subheader("Hyperparamètres du modèle")
        n_estimators = st.sidebar.number_input("Choisir le nombre de d'arbres",
                                               1, 20, step=1
                                               )
        max_depth = st.sidebar.number_input(
            "Profondeur maximale d'un arbre",
            1, 20, step=1
        )

        bootstrap_str = st.sidebar.radio("Echantillon lors de la création", ("True", "False"))
        bootstrap = bool(bootstrap_str)

        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Random Forest resultats ")
            
            model = RandomForestClassifier(
                n_estimators = n_estimators,
                max_depth = max_depth,
                bootstrap = bootstrap
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            st.write("Accuracy", accuracy)
            st.write("Precision", precision)
            st.write("Recall", recall)


if __name__ == '__main__':
    main()