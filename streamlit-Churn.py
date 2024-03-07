import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import streamlit as st
import joblib
import time
import io
import requests
from io import BytesIO
import base64

st.set_page_config(layout="wide")

# Titre de l'application
st.markdown("<h1 style='text-align:center; color: black;'>Churn Contol</h1>", unsafe_allow_html=True)

# Fonction pour l'ETL
def perform_etl(data):

    # Convertir les colonnes de date en type Timestamp
    date_columns = ['DATEEFFE', 'DATEDEBU', 'DATE_FIN', 'DATE_MEC', 'DATNAICO', 'DATDELPE']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], format='%Y/%m/%d', errors='coerce')

    df_copy = data[['IDENPOLI', 'IDENRISQ', 'REDUSAHA', 'CODEPROD', 'CODEINTE', 'DATEEFFE', 'DATEDEBU', 'DATE_FIN', 'TYPECONT', 
                    'CLASVEHI', 'DATDELPE', 'VILLASSU', 'RC', 'DR', 'VL', 'IC', 'DC', 'PTA', 'TC', 'BG', 'NOMBSINI', 'NOMBGARA', 'TYPEMOTE',
                    'PUISVEHI', 'DATNAICO', 'TYPEIMMA', 'PRIMNETT', 'DATE_MEC', 'SEXECOND']]
    
    df = df_copy.copy()

    # Imputation par le mode
    mode_VILLASSU = df['VILLASSU'].mode().iloc[0]
    df['VILLASSU'].fillna(mode_VILLASSU, inplace=True)

    mode_SEXE = df['SEXECOND'].mode().iloc[0]
    df['SEXECOND'].fillna(mode_SEXE, inplace=True)

    mode_DATE_MEC = df['DATE_MEC'].mode().iloc[0]
    df['DATE_MEC'].fillna(mode_DATE_MEC, inplace=True)

    mode_DATE_MEC = df['DATEEFFE'].mode().iloc[0]
    df['DATEEFFE'].fillna(mode_DATE_MEC, inplace=True)

    mode_DATNAICO = df['DATNAICO'].mode().iloc[0]
    df['DATNAICO'].fillna(mode_DATNAICO, inplace=True)

    mode = df['DATDELPE'].mode().iloc[0]
    df['DATDELPE'].fillna(mode, inplace=True)

    # Nous utilisons la moyenne car la variables est quantitative discrète
    median_puisvehi = df['PUISVEHI'].median()
    df['PUISVEHI'].fillna(median_puisvehi, inplace=True)

    # Nous utilisons la moyenne car la variables est quantitative continue
    median_puisvehi = df['PRIMNETT'].median()
    df['PRIMNETT'].fillna(median_puisvehi, inplace=True)      

    df['VILLASSU'] = df['VILLASSU'].str.replace('.', 'INDEFINI')

    # Convertir les colonnes de date en datetime
    date_columns = ['DATE_MEC', 'DATNAICO']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%Y/%m/%d', errors='coerce')

    mode_DATE_MEC = df['DATE_MEC'].mode().iloc[0]
    df['DATE_MEC'].fillna(mode_DATE_MEC, inplace=True)

    mode_DATNAICO = df['DATNAICO'].mode().iloc[0]
    df['DATNAICO'].fillna(mode_DATNAICO, inplace=True)

    mode = df['DATDELPE'].mode().iloc[0]
    df['DATDELPE'] = df['DATDELPE'].fillna(mode)

    # Calcul de la différence en mois pour 'DURCOUV'
    df['DURCOUV'] = ((df['DATE_FIN'] - df['DATEDEBU']).dt.days / 30).astype(int)

    # Calcul de l'âge du véhicule en années
    df['AGEVEHI'] = ((df['DATEEFFE'] - df['DATE_MEC']).dt.days / 365).astype(int)

    # Calcul de l'âge du conducteur en années
    df['AGECOND'] = ((df['DATEEFFE'] - df['DATNAICO']).dt.days / 365).astype(int)

    # Calcul de la durée du permis en années
    df['LICENSE_DURATION'] = ((df['DATEEFFE'] - df['DATDELPE']).dt.days / 365).astype(int)

    prefixes_a_supprimer = ['.', '-']

    # Supprimer les préfixes dans la colonne "MARQVEHI"
    for prefix in prefixes_a_supprimer:
        df['VILLASSU'] = df['VILLASSU'].str.replace(prefix, '')

    df_pred = df[['IDENPOLI', 'IDENRISQ', 'REDUSAHA', 'CODEPROD', 'CODEINTE', 'VILLASSU', 'CLASVEHI', 'TYPEIMMA',
                'RC', 'DR', 'DC', 'PTA', 'TC', 'VL', 'IC', 'BG', 'TYPEMOTE', 'TYPECONT', 'SEXECOND',
                'NOMBSINI', 'NOMBGARA', 'PUISVEHI', 'PRIMNETT', 'DURCOUV', 'AGEVEHI', 'AGECOND', 'LICENSE_DURATION']]

    # Remplacer les valeurs qui ne sont pas comprises entre 0 et 40 par NaN
    df_pred.loc[~((df_pred['AGEVEHI'] >= 0) & (df_pred['AGEVEHI'] <= 40)), 'AGEVEHI'] = np.nan

    df_pred.loc[~((df_pred['AGECOND'] >= 18) & (df_pred['AGECOND'] <= 70)), 'AGECOND'] = np.nan

    df_pred.loc[~((df_pred['LICENSE_DURATION'] >= 0) & (df_pred['LICENSE_DURATION'] <= 50)), 'LICENSE_DURATION'] = np.nan

    # Nous utilisons la moyenne car la variables est quantitative discrète
    median_puisvehi = df_pred['AGEVEHI'].median()
    df_pred['AGEVEHI'].fillna(median_puisvehi, inplace=True)

    # Nous utilisons la moyenne car la variables est quantitative continue
    median_puisvehi = df_pred['AGECOND'].median()
    df_pred['AGECOND'].fillna(median_puisvehi, inplace=True)

    # Nous utilisons la moyenne car la variables est quantitative continue
    median_puisvehi = df_pred['LICENSE_DURATION'].median()
    df_pred['LICENSE_DURATION'].fillna(median_puisvehi, inplace=True)

    # Convertir les colonnes en types entiers
    df_pred['PUISVEHI'] = df_pred['PUISVEHI'].astype('Int64')
    df_pred['AGEVEHI'] = df_pred['AGEVEHI'].astype('Int64')
    df_pred['AGECOND'] = df_pred['AGECOND'].astype('Int64')
    df_pred['LICENSE_DURATION'] = df_pred['LICENSE_DURATION'].astype('Int64')

    return df_pred

# Fonction pour effectuer la prédiction
def perform_prediction(df_pred):

    data = df_pred.copy()

    # Télécharger le modèle depuis GitHub
    #model_url = "https://raw.githubusercontent.com/VotreNom/VotreRepo/main/model.joblib"
    model_url = "https://github.com/youssouph5/Churn_Control/blob/main/Model_Churn151223.joblib"
    response = requests.get(model_url)
    model_file = BytesIO(response.content)

    XGB_model = joblib.load(model_file)

    # Effectuer la prédiction
    data['PREDICTION'] = XGB_model.predict(data.iloc[:, 3:])
    data['PREDICTION'] = data['PREDICTION'].apply(lambda x: "N" if x == 1 else "O")

    data['PROBA_CHURN%'] = (XGB_model.predict_proba(data.iloc[:, 3:])[:, 1] * 100).round(2)

    return data

# Appliquer la logique pour déterminer le contrat renouvelé
def determiner_contrat(proba, seuil):
    if proba < seuil:
        return "O"  # Contrat renouvelé
    else:
        return "N"  # Contrat non renouvelé

# Fonction pour créer un lien de téléchargement vers un DataFrame au format Excel
def get_binary_file_downloader_html(df):
    # Convertir le DataFrame en un fichier Excel en mémoire
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    excel_file = output.getvalue()
    # Générer le lien de téléchargement
    b64 = base64.b64encode(excel_file)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="pred_dt.xlsx">Télécharger le fichier Excel</a>'

# Streamlit app
def main():

    # Ajouter un curseur de sélection de seuil dans la barre latérale
    seuil_proba = st.sidebar.slider("Seuil de probabilité (%)", min_value=50, max_value=100, value=65, step=1)

    # Uploader de fichier
    uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel", type=["xlsx", "xls"])

    # Initialisation de data avec None
    data = None

    if uploaded_file is not None:
        try:
            # Lecture du fichier Excel
            dat = pd.read_excel(uploaded_file)

            # Bouton pour exécuter l'ETL
            etl_button = st.button("PRÉDICTION DES DONNÉES")

            if etl_button:
                # Afficher une barre de progression
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                
                # Exécuter l'ETL
                data = perform_etl(dat)

                # Afficher un message de succès
                st.success("ETL terminé avec succès!") 
                
                st.write("Les 5 premières lignes du fichier de la prédiction:")
                st.write(data.head(10))

                st.write("Prédiction en cours...")

                # Afficher une barre de progression
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)

                # Exécuter la prédiction
                pred_dt = perform_prediction(data)

                # Appliquer la logique de détermination du contrat renouvelé
                pred_dt[f"PREDICTION_{seuil_proba}%"] = pred_dt['PROBA_CHURN%'].apply(lambda x: determiner_contrat(x, seuil_proba))

                # Affichage des 5 premières lignes
                #st.write("Les 5 premières lignes du fichier de la prédiction:")
                #st.write(pred_dt.head(10))

                # Affichage des dimensions du fichier
                st.write(f"Le fichier comporte : {pred_dt.shape[0]} lignes, et {pred_dt.shape[1]} colonnes")

                # Créer une rangée pour les graphes
                button_col1, button_col2 = st.columns(2)

                # Filtrer les lignes où PROBA_CHURN% est supérieur à 65 %
                high_prob_churn = pred_dt[pred_dt['PROBA_CHURN%'] < 65]

                # Calculer la distribution des prédictions dans ce sous-ensemble
                churn_distribution = high_prob_churn['PREDICTION'].value_counts()

                # Tracer le diagramme circulaire pour cette distribution
                fig, ax = plt.subplots(figsize=(13, 7))
                ax.pie(churn_distribution, labels=churn_distribution.index, autopct='%1.1f%%', startangle=120)
                ax.set_title("Distribution du Churn (PROBA_CHURN% < 65%)")
                ax.axis('equal')

                # Afficher chaque graphique dans une colonne
                with button_col1:
                    # Afficher le diagramme dans Streamlit
                    st.pyplot(fig)

                # Définir une palette de couleurs personnalisée
                custom_palette = ['red', 'green']

                # Définir le style seaborn
                sns.set_style("whitegrid")

                # Graphique en barres pour la fréquence de chaque catégorie
                fig1, ax1 = plt.subplots(figsize=(13, 9))
                sns.countplot(x="PREDICTION", data=pred_dt, palette=custom_palette, ax=ax1)
                plt.title("Distribution du churn : renouvellement contrat ou pas")

                    # Ajouter le nombre de chaque catégorie sur les barres
                for p in ax1.patches:
                    ax1.annotate(format(p.get_height(), ".0f"), (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

                # Afficher le graphique dans Streamlit
                with button_col2:
                    st.pyplot(fig1)
                
                # Créer un lien de téléchargement pour le DataFrame au format Excel
                st.sidebar.markdown("### Télécharger le fichier Excel")
                st.sidebar.markdown(get_binary_file_downloader_html(pred_dt), unsafe_allow_html=True)

        except Exception as e:
            # Gérer les erreurs
            st.error("Une erreur s'est produite lors de la lecture du fichier. Veuillez réessayer.")
            print(e)

if __name__ == "__main__":
    main()
