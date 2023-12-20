import streamlit as st
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.image("logo_rvb_horizontal_petit.png")
st.title("Dashboard")
st.set_option('deprecation.showPyplotGlobalUse', False)

CE = pd.read_excel('CollectionExperts.xlsx', engine='openpyxl')
CI = pd.read_excel('CollectionInterventions.xlsx', engine='openpyxl')
CU = pd.read_excel('CollectionUser.xlsx', engine='openpyxl')
CN = pd.read_excel('CollectionNewsletter.xlsx', engine='openpyxl')
CR = pd.read_excel('CollectionRecommandation.xlsx', engine='openpyxl')
CC = pd.read_excel('CollectionCompany.xlsx', engine='openpyxl')
CN['createdAt'] = pd.to_datetime(CN['createdAt']).dt.date

CE['createdAt'] = CE['createdAt'].str.replace("010","10")
CE['createdAt'] = pd.to_datetime(CE['createdAt'], dayfirst=True)

CC['createdAt'] = CC['createdAt'].str.replace("010","10")
CC['createdAt'] = pd.to_datetime(CC['createdAt'], dayfirst=True)

def extract_coordinates(geo_data):
    try:
        if pd.notna(geo_data):
            geo_data = json.loads(geo_data.replace("'", "\""))
            coordinates = geo_data["location"]["coordinates"]
            return pd.Series({'LAT': coordinates[1], 'LON': coordinates[0]})
        else:
            return pd.Series({'LAT': None, 'LON': None})
    except (KeyError, json.JSONDecodeError):
        return pd.Series({'LAT': None, 'LON': None})

# Appliquer la fonction pour créer deux nouvelles colonnes 'LAT' et 'LON'
CE[['LAT', 'LON']] = CE['geo'].apply(extract_coordinates)

def extract_values(row, key):
    if pd.isna(row) or row == '[]' or isinstance(row, bool):
        return None
    try:
        references_list = json.loads(row)
        values = [item.get(key, "") for item in references_list if key in item]
        return values if any(values) else None
    except json.JSONDecodeError:
        return None

# Appliquez la fonction à chaque clé pour créer de nouvelles colonnes
keys_to_extract = ["name", "company_name", "company_function", "phone", "mail"]

for key in keys_to_extract:
    CE[key] = CE['references'].apply(lambda x: extract_values(x, key)).apply(lambda x: ' '.join(x) if x else None)

def remplacer(df, colonne, valeur_a_remplacer, par):
    df[colonne] = df[colonne].replace(valeur_a_remplacer, par)
    return df

CI = remplacer(CI, 'hours_planned', ["35 Heures"], 35)
CI = remplacer(CI, 'hours_planned', ["18 Heures"], 18)
CI = remplacer(CI, 'hours_planned', ["46 Heures"], 46)
CI = remplacer(CI, 'hours_planned', ["7h"], 7)
CI = remplacer(CI, 'hours_planned', ["30h par semaine pendant 3 semaines"], 90)
CI = remplacer(CI, 'hours_planned', ["10h"], 10)
CI = remplacer(CI, 'hours_planned', ["30h réparties sur 2 semaines"], 30)
CI = remplacer(CI, 'hours_planned', ["14 Heures"], 14)
CI = remplacer(CI, 'hours_planned', ["112h sur l'année (environ 4h/semaine)"], 112)
CI = remplacer(CI, 'hours_planned', ["8 Heures"], 8)
CI = remplacer(CI, 'hours_planned', ["15h ( créneau de 1h30 séparé de 15 min)"], 15)
CI = remplacer(CI, 'hours_planned', ["18 heures x2 classes"], 36)
CI = remplacer(CI, 'hours_planned', ["28h réparties comme suit : 7/10 PM 21/11 PM 12/12 PM 15/12 PM 10/01 AM 12/01 PM 1/02 AM 3/02 AM"], 28)
CI = remplacer(CI, 'hours_planned', ["23 Heures"], 23)
CI = remplacer(CI, 'hours_planned', ["Module de 30h réparties comme suit : 10/10 PM 3h 10/01 PM 3.5h 11/01 Day 7h 14/03 AM 3h 04/04 PM 3h 05/04 Day 7h 07/04 AM 3.5h"], 30)
CI = remplacer(CI, 'hours_planned', ["14h"], 14)
CI = remplacer(CI, 'hours_planned', ["68h annuelle"], 68)
CI = remplacer(CI, 'hours_planned', ["17h30"], 17.5)
CI = remplacer(CI, 'hours_planned', ["24h30"], 24.5)
CI = remplacer(CI, 'hours_planned', ["3 Heures"], 3)
CI = remplacer(CI, 'hours_planned', ["24,5 Heures"], 24.5)
CI = remplacer(CI, 'hours_planned', ["14 heures"], 14)
CI = remplacer(CI, 'hours_planned', ["4 (8h30-12h30)"], 4)
CI = remplacer(CI, 'hours_planned', ["18h"], 18)
CI = remplacer(CI, 'hours_planned', ["18 heures x 2 classes"], 36)
CI = remplacer(CI, 'hours_planned', ["60h sur l'année"], 60)
CI = remplacer(CI, 'hours_planned', ["70 Heures"], 70)
CI = remplacer(CI, 'hours_planned', ["21 Heures"], 21)
CI = remplacer(CI, 'hours_planned', ["28 Heures"], 28)
CI = remplacer(CI, 'hours_planned', ["60 Heures"], 60)
CI = remplacer(CI, 'hours_planned', ["30h"], 30)
CI = remplacer(CI, 'hours_planned', ["18 heures"], 18)
CI = remplacer(CI, 'hours_planned', ["7,5h/j, sur 10 jours, sur 2 semaines"], 75)
CI = remplacer(CI, 'hours_planned', ["ezfefz"], np.nan)
CI = remplacer(CI, 'hours_planned', ["7 Heures/Jour","4 heures par semaine","40 Heures/Groupe"], np.nan)
CI['hours_planned'] = pd.to_numeric(CI['hours_planned'], errors='coerce')

CE = remplacer(CE, 'experienceTime', ["mois de 10 ans"], ['moins de 10 ans'])
CE['experienceTimeOne'] = np.where(
    (CE['experienceTime'] == '10 à 20 ans') 
    | (CE['experienceTime'] == '20 à 30 ans') 
    | (CE['experienceTime'] == '+ de 30 ans') 
    | (CE['experienceTime'] == 'moins de 10 ans'),
    CE['experienceTime'], np.nan)



CE['experienceTimeSecond'] = np.where(
    (CE['experienceTime'] == 'Entre 5 et 10 ans') 
    | (CE['experienceTime'] == 'Entre 10 et 15 ans') 
    | (CE['experienceTime'] == 'Entre 15 et 25 ans') 
    | (CE['experienceTime'] == '+ de 25 ans'),
    CE['experienceTime'], np.nan)



CE["location"] = CE["location"].str.split(',').str[0]
CE["location"] = CE["location"].replace(["Bordeaux et périphérie"], "Bordeaux")
CE["location"] = CE["location"].replace(["Paris et périphérie"], "Paris")
CE["location"] = CE["location"].replace(["Île-de-France"], "Paris")
CE["location"] = CE["location"].replace(["Lyon et périphérie"], "Lyon")
CE["location"] = CE["location"].replace(["Lille et périphérie"], "Lille")
CE["location"] = CE["location"].replace(["Toulouse et périphérie"], "Toulouse")
CE["location"] = CE["location"].replace(["Montpellier et périphérie"], "Montpellier")
CE["location"] = CE["location"].replace(["Nantes et périphérie"], "Nantes")
CE["location"] = CE["location"].replace(["Nouvelle-Aquitaine"], "Bordeaux")
CE["location"] = CE["location"].replace(["Bordeaux Métropole"], "Bordeaux")
CE["location"] = CE["location"].replace(["Bordeaux "], "Bordeaux")
CE["location"] = CE["location"].replace("France", pd.NA)
CE["location"] = CE["location"].replace(["BORDEAUX"], "Bordeaux")
CE["location"] = CE["location"].replace(["Gironde"], "Bordeaux")
CE["location"] = CE["location"].replace(["Aquitaine"], "Bordeaux")

CE = remplacer(CE, 'studyLevel', ["Bac +5"], ['Bac5'])
CE = remplacer(CE, 'studyLevel', ["Bac + 8"], ['Bac8'])
CE = remplacer(CE, 'studyLevel', ["Bac + 3"], ['Bac3']) 

CE["location"] = CE["location"].str.split(',').str[0]
CE["location"] = CE["location"].replace(["Bordeaux et périphérie"], "Bordeaux")
CE["location"] = CE["location"].replace(["Paris et périphérie"], "Paris")
CE["location"] = CE["location"].replace(["Île-de-France"], "Paris")
CE["location"] = CE["location"].replace(["Lyon et périphérie"], "Lyon")
CE["location"] = CE["location"].replace(["Lille et périphérie"], "Lille")
CE["location"] = CE["location"].replace(["Toulouse et périphérie"], "Toulouse")
CE["location"] = CE["location"].replace(["Montpellier et périphérie"], "Montpellier")
CE["location"] = CE["location"].replace(["Nantes et périphérie"], "Nantes")
CE["location"] = CE["location"].replace(["Nouvelle-Aquitaine"], "Bordeaux")
CE["location"] = CE["location"].replace(["Bordeaux Métropole"], "Bordeaux")
CE["location"] = CE["location"].replace(["Bordeaux "], "Bordeaux")
CE["location"] = CE["location"].replace("France", pd.NA)
CE["location"] = CE["location"].replace(["BORDEAUX"], "Bordeaux")
CE["location"] = CE["location"].replace(["Gironde"], "Bordeaux")
CE["location"] = CE["location"].replace(["Aquitaine"], "Bordeaux")


merged_df = pd.merge(CE, CI, left_on='userId', right_on='expert_userId', how='left')
merged_df2 = pd.merge(CU, CI, left_on='_id', right_on='expert_userId', how='inner')
merged_df3 = pd.merge(CE, CR, left_on='_id', right_on='expertId', how='inner')
#final_merged_df = pd.merge(merged_df, merged_df2, left_on='expert_userId', right_on='expert_userId', how='inner')


if st.sidebar.checkbox("Afficher les données 'Collection Interventions'", False):
    st.subheader("Jeu de données")
    st.write(CI)
if st.sidebar.checkbox("Afficher les données 'Collection Profile'", False):
    st.subheader("Jeu de données")
    st.write(CE)
if st.sidebar.checkbox("Afficher les données 'Collection User'", False):
    st.subheader("Jeu de données")
    st.write(CU)
if st.sidebar.checkbox("Afficher les données 'Collection Newsletter'", False):
    st.subheader("Jeu de données")
    st.write(CN)
if st.sidebar.checkbox("Afficher les données 'Collection Recommandation'", False):
    st.subheader("Jeu de données")
    st.write(CR)
if st.sidebar.checkbox("Afficher les données 'Collection Company'", False):
    st.subheader("Jeu de données")
    st.write(CC)


tab1, tab2, tab3 = st.tabs(["Département RH", "Département commerce", "Département marketing"])

toutes_les_années = [2018, 2019, 2020, 2021, 2022, 2023]

with tab1:
    st.title("Experts incrits sur la plateforme")

    choix_drh = st.multiselect("Sélection indicateur(s)", ["Total","Carte", "Par période", "Par domaine", "Par ville" , "Avec ou sans entretien","Avec ou sans références","Note","Recommandation", "How we met"], default=["Total", "Carte", "Par période", "Par domaine", "Par ville", "Avec ou sans entretien", "Avec ou sans références","Note","Recommandation","How we met"])

    année = st.multiselect("Choisir l'année", toutes_les_années, default=toutes_les_années)
    filtered_data = CE[CE['createdAt'].dt.year.isin(année)] 
 

    nbre_tot_exp = filtered_data['type'].count()
    if "Total" in choix_drh:
        st.markdown(f"<h1 style='text-align: left; color: black; font-size: 2em;'>Total : {nbre_tot_exp}</h1>", unsafe_allow_html=True)

    if "Carte" in choix_drh:
        CELL = filtered_data.dropna(subset=['LON'])
        CELL = filtered_data.dropna(subset=['LAT'])
        st.map(CELL)

    if "Par période" in choix_drh:
        st.subheader("Par période")
        result = CE.pivot_table(index='createdAt', values='userId', aggfunc='count').reset_index()

        sns.histplot(filtered_data['createdAt'], kde=True, bins=65, color="#bcded0")
        plt.xlabel('Période')
        plt.ylabel("Nombre d'inscriptions")
        plt.xticks(rotation=45)
        st.pyplot()


    if "Par domaine" in choix_drh:
        st.subheader("% d'experts par domaine")
        nbre_domaine = st.number_input("Choisir le nombre de domaine à afficher", 1, 50, value=10, step=1)                                               
        result2 = filtered_data.pivot_table(index='domains', values='userId', aggfunc='count').reset_index()
        result2['pct'] = round((result2['userId'] / result2['userId'].sum()) * 100, 1)
        result2 = result2.sort_values(by='pct', ascending=False).head(nbre_domaine)

        # Création du graphique à l'aide de Seaborn
        plt.figure(figsize=(6, 8))
        sns.barplot(x='pct', y='domains', data=result2, color="#bcded0", orient='h')

        plt.ylabel('Domains')
        plt.xlabel('Pourcentage')
        plt.title('Pourcentage par domaine')
        st.pyplot()
        
    
    if "Par ville" in choix_drh:
        st.subheader("% d'experts par ville")
        nbre_ville = st.number_input("Choisir le nombre de ville à afficher (Par défaut 10)", 1, 15, value=10, step=1) 
        experts_par_ville = filtered_data["location"].value_counts().nlargest(nbre_ville)
        labels = experts_par_ville
        num_colors = 20
        colors = sns.color_palette("light:#03bc93", n_colors=num_colors)
        plt.figure(figsize=(10, 10))
        plt.pie(experts_par_ville, autopct='%.2f%%', labels=labels.index, pctdistance=0.9, colors=colors)
        st.pyplot()

    if "Avec ou sans entretien" in choix_drh:
        st.subheader("% d'entretiens passés")
        filtered_data["done"] = filtered_data["done"].replace(np.nan, "Non")
        filtered_data["done"] = filtered_data["done"].replace(1.0, "Oui")
        filtered_data["done"] = filtered_data["done"].replace(0.0, "Non")
        result3 = filtered_data["done"].value_counts()
        labels = ['Avec entretien','Sans entretien']
        colors = sns.color_palette(["#03bc93", "#bcded0"])
        plt.figure(figsize=(8, 8))
        plt.pie(result3, autopct='%.2f%%',  pctdistance=0.9, colors=colors)
        plt.legend(labels)
        st.pyplot()

    if "Avec ou sans références" in choix_drh:
        st.subheader("% d'experts avec ou sans 'Références'")
        # Fonction générique pour extraire les valeurs des clés
        ref = filtered_data['name'].isnull().value_counts()
        labels = ['Non complété','Complété']

        #explode = [0.1, 0.1]
        colors = sns.color_palette(["#03bc93", "#bcded0"])
        plt.figure(figsize=(8, 8))
        plt.pie(ref, autopct='%.2f%%', colors=colors)
        plt.legend(labels)
        st.pyplot()

        liste_ss_ref = filtered_data[CE['name'].isnull()]
        st.subheader("Liste des experts sans 'Références'")
        st.write(liste_ss_ref)

    if "Note" in choix_drh:
        st.subheader("Note d'évaluation par expert")
        merged_df['name'] = merged_df['name'].fillna(merged_df['userId'])
        note = merged_df.pivot_table(index=['name' or 'userId'], values=['note_communication','note_quality','note_level'])
        st.write(note)
    
    if "Recommandation" in choix_drh:
        st.subheader("Pourcentage d'experts recommandés")
        merged_df3['name'] = merged_df3['name'].fillna(merged_df3['userId'])
        #drop_name = merged_df3.dropna(subset=['name'])
        reco = merged_df3['name'].drop_duplicates().reset_index()
        nbre_reco = len(merged_df3['expertId'].unique())
        pct_exp_reco = round((( nbre_reco/ nbre_tot_exp ) * 100), 2)
        container = st.container(border=True)   
        container.write(f"Pourcentage d'experts recommandés : {pct_exp_reco} %")
        st.subheader("Liste des experts recommandés")
        st.write(reco)

    if "How we met" in choix_drh:
        st.subheader("How we met")
        CC = CC.dropna(subset=['howWeMet'])
        howWeMet_pivot = CC.pivot_table(index=['companyOrSchool', 'howWeMet'], values='_id', aggfunc='count').reset_index()
        howWeMet_pivot['pct.howWeMet.pivot'] = round((howWeMet_pivot['_id'] / howWeMet_pivot['_id'].sum()) * 100, 2)
        howWeMet_pivot = howWeMet_pivot.sort_values(by='pct.howWeMet.pivot', ascending=False)

        colors = {'company': '#bb8df7', 'school': '#7d43c8'}

        plt.figure(figsize=(8, 8))
        ax = sns.barplot(y='howWeMet', x='pct.howWeMet.pivot', hue='companyOrSchool' ,data=howWeMet_pivot, palette=colors,  orient='h' )
        plt.xlabel("Pourcentage")
        plt.ylabel("")
        st.pyplot()
        st.write()

with tab2:
    choix_dc = st.multiselect("Sélection indicateur(s)", ["Missions","Taux", "Clients"], default=["Missions","Taux", "Clients"])
    #Sélectionner tous les experts par défaut
    unique_experts = merged_df['name'].unique()
    unique_experts = ['Tous les experts'] + [str(expert) for expert in unique_experts if expert is not None]
    selected_expert = st.selectbox("Sélectionnez un expert (avec références)", unique_experts)

    # Filtrer par expert
    if selected_expert != 'Tous les experts':
        filtered_df = merged_df[merged_df['name'] == selected_expert]
    else:
        filtered_df = merged_df

    if "Missions" in choix_dc:

        nbre_tot = filtered_df['expert_userId'].count()
        nbre_tot_h = filtered_df['hours_planned'].sum()
        duree_mean = round(filtered_df['hours_planned'].mean(), 2)

        # Utilisation de st.columns pour créer trois colonnes
        col1, col2, col3 = st.columns(3)

        # Dans la première colonne
        with col1:
            container = st.container(border=True)
            container.write(f"Nombre total de missions : {nbre_tot}")

        # Dans la deuxième colonne
        with col2:
            container = st.container(border=True)
            container.write(f"Nombre total d'heures de missions : {nbre_tot_h} h")

        # Dans la troisième colonne
        with col3:
            container = st.container(border=True)
            container.write(f"Durée moyenne des missions : {duree_mean} h")

    if "Taux" in choix_dc:
        taux_sup_10 = filtered_df[(filtered_df['daily_hourly_prices.hourly_price_min'] >= 10) & (filtered_df['daily_hourly_prices.hourly_price_min'] < 10000)]
        taux_min = taux_sup_10['daily_hourly_prices.hourly_price_min'].min()

        taux_inf_10000 = filtered_df[(filtered_df['daily_hourly_prices.hourly_price_max'] >= 10) & (filtered_df['daily_hourly_prices.hourly_price_max'] <= 10000)]
        taux_inf = taux_inf_10000['daily_hourly_prices.hourly_price_max'].max()

        taux_sup_10_mean = round(taux_sup_10['daily_hourly_prices.hourly_price_min'].mean(), 2)
        taux_inf_10000_mean = round(taux_inf_10000['daily_hourly_prices.hourly_price_max'].mean(), 2)
        moyenne_total = round((taux_sup_10_mean + taux_inf_10000_mean) / 2, 2)

        col1, col2, col3 = st.columns(3)  
        with col1:
            container = st.container(border=True)       
            container.write(f"Taux journalier minimum : {taux_min} €")

        with col2:
            container = st.container(border=True)   
            container.write(f"Taux journalier maximum : {taux_inf} €")

        with col3:
            container = st.container(border=True)   
            container.write(f"Taux horaire moyen : {moyenne_total} €")

    if "Clients" in choix_dc:
        st.subheader("Répartition clients 'Company Or School'")

        CU.dropna(subset=['companyOrSchool'])

        companyOrSchool = CU['companyOrSchool'].value_counts()

        labels = companyOrSchool
        colors = sns.color_palette(["#03bc93", "#bcded0"])
        plt.figure(figsize=(8, 8))
        plt.pie(companyOrSchool, labels=companyOrSchool, colors=colors)
        plt.legend(companyOrSchool.index)
        st.pyplot()
        

with tab3:
    choix_dm = st.multiselect("Sélection indicateur(s)", ["Statut juridique","Nombre d'années d'expérience (1er catégorie)","Nombre d'années d'expérience (2ème catégorie)", "Diplôme", "Newsletter"],
                              default=["Statut juridique","Nombre d'années d'expérience (1er catégorie)","Nombre d'années d'expérience (2ème catégorie)", "Diplôme", "Newsletter"])

    if "Statut juridique" in choix_dm:
        dc = CU.dropna(subset=['civility'])
        hf = dc['civility'].unique()
        selected_hf = st.multiselect("Choix Homme/Femme", hf, default=hf)
        filtered_hf = CU[CU['civility'].isin(selected_hf)]
        
        st.subheader('Statut juridique en %')
        companyType = filtered_hf.dropna(subset=['company.type'])
        table_companyType = companyType.pivot_table(index=['company.type','civility'], values='_id', aggfunc='count').reset_index()
        table_companyType['pct.company.type'] = round((table_companyType['_id'] / table_companyType['_id'].sum()) * 100, 2)
        table_companyType = table_companyType.sort_values(by='pct.company.type', ascending=False)

        colors = {'male': '#03bc93', 'female': '#7d43c8'}

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='company.type', y='pct.company.type', hue='civility' ,data=table_companyType, palette=colors )
        plt.xlabel("Statut juridique")
        plt.ylabel("")
        plt.xticks(rotation=35)

        st.pyplot()
        st.write()

    dom = CE.dropna(subset=['domains'])
    dom_unique = dom['domains'].unique()
    dom_unique = ['Tous les domaines'] + [str(domaine) for domaine in dom_unique if domaine is not None]
    selected_dom = st.selectbox("Choix du domaines", dom_unique)

    if selected_dom == 'Tous les domaines':
        filtre_dom = CE
    else:
        filtre_dom = CE[CE['domains'].astype(str) == selected_dom]

    if "Nombre d'années d'expérience (1er catégorie)" in choix_dm:
        st.subheader("Nombre d'années d'expérience en %")
        ordered_categories_ets = ['Entre 5 et 10 ans', 'Entre 10 et 15 ans', 'Entre 15 et 25 ans', '+ de 25 ans']
        experienceTimeSecond = filtre_dom.dropna(subset=['experienceTimeSecond'])
        count_ets = experienceTimeSecond['experienceTimeSecond'].value_counts()
        count_ets = count_ets.reindex(ordered_categories_ets)
        labels = count_ets
        colors = sns.color_palette("flare")
        plt.figure(figsize=(7, 7))
        plt.pie(count_ets, autopct='%.2f%%', colors=colors, wedgeprops=dict(width=0.3), pctdistance=0.85)
        plt.legend(labels.index, loc="upper right", title="Catégories", bbox_to_anchor=(1.2, 1))

        st.pyplot()
        st.write()

    if "Nombre d'années d'expérience (2ème catégorie)" in choix_dm:
        st.subheader("Nombre d'années d'expérience en %")
        ordered_categories_eto = ['moins de 10 ans', '10 à 20 ans', '20 à 30 ans', '+ de 30 ans']
        experienceTimeOne = filtre_dom.dropna(subset=['experienceTimeOne'])
        count_eto = experienceTimeOne['experienceTimeOne'].value_counts()
        count_eto = count_eto.reindex(ordered_categories_eto)
        labels = count_eto
        colors = sns.color_palette("flare")
        plt.figure(figsize=(7, 7))
        plt.pie(count_eto, autopct='%.2f%%', colors=colors, wedgeprops=dict(width=0.3), pctdistance=0.85)
        plt.legend(labels.index, loc="upper right", title="Catégories", bbox_to_anchor=(1.2, 1))
        st.pyplot()
        st.write()


    if "Diplôme" in choix_dm:
        st.subheader("Diplôme obtenu en %")
        studyLevel = filtre_dom.dropna(subset=['studyLevel'])   
        pctStudyLevel = studyLevel ['studyLevel'].value_counts()
        labels = pctStudyLevel
        num_colors = 7
        colors = sns.color_palette("light:#7d43c8", n_colors=num_colors)
        plt.figure(figsize=(7, 7))
        plt.pie(pctStudyLevel, autopct='%.2f%%', colors=colors, pctdistance=1.2)
        plt.legend(labels.index, loc="upper right", title="Study Level", bbox_to_anchor=(1.2, 1))

        st.pyplot()
        st.write()

    if "Newsletter" in choix_dm:
        st.subheader("Newsletter")  

        pcttype = CN['type'].value_counts()
        labels = pcttype.index  # Use index to get the labels
        num_colors = 2
        colors = sns.color_palette("light:#7d43c8", n_colors=num_colors)

        plt.figure(figsize=(7, 7))
        sns.barplot(x=pcttype.index, y=pcttype.values, palette=colors)
        st.pyplot()
        st.write()

st.subheader("Auteur : Mickaël Brothier")
            
