##########################################
# Projet : FER - extraction de plusieurs images et analyse : 25fps - moyenne - mode - variance
# Auteur : Stéphane Meurisse
# Contact : stephane.meurisse@example.com
# Site Web : https://www.codeandcortex.fr
# LinkedIn : https://www.linkedin.com/in/st%C3%A9phane-meurisse-27339055/
# Date : 15 octobre 2024
##########################################

# pip install opencv-python-headless fer pandas matplotlib altair xlsxwriter scikit-learn numpy streamlit tensorflow yt_dlp
# pip install tensorflow-metal -> pour Mac M2
# pip install vl-convert-python
# FFmpeg -> attention sous Mac la procédure d'installation sous MAC nécessite "Homebrew"


import streamlit as st
import subprocess
import os
import pandas as pd
import numpy as np
from collections import Counter
from fer import FER
import cv2
from yt_dlp import YoutubeDL
import altair as alt

# Fonction pour vider le cache
def vider_cache():
    st.cache_data.clear()
    st.write("Cache vidé systématiquement au lancement du script")

# Appeler la fonction de vidage du cache au début du script
vider_cache()

# Fonction pour définir le répertoire de travail
def definir_repertoire_travail():
    repertoire = st.text_input("Définir le répertoire de travail", "", key="repertoire_travail")
    if not repertoire:
        st.write("Veuillez spécifier un chemin valide.")
        return ""
    repertoire = repertoire.strip()
    repertoire = os.path.abspath(repertoire)
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
        st.write(f"Le répertoire a été créé : {repertoire}")
    else:
        st.write(f"Le répertoire existe déjà : {repertoire}")
    return repertoire

# Fonction pour télécharger la vidéo avec ytdlp
def telecharger_video(url, repertoire):
    video_path = os.path.join(repertoire, 'video.mp4')
    if os.path.exists(video_path):
        st.write(f"La vidéo est déjà présente dans le répertoire : {video_path}")
        return video_path
    st.write(f"Téléchargement de la vidéo à partir de {url}...")
    ydl_opts = {'outtmpl': video_path, 'format': 'best'}
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    st.write(f"Téléchargement terminé : {video_path}")
    return video_path

# Fonction pour extraire des images à 25fps avec FFmpeg
def extraire_images_25fps_ffmpeg(video_path, repertoire, seconde):
    images_extraites = []
    for frame in range(25):
        image_path = os.path.join(repertoire, f"image_25fps_{seconde}_{frame}.jpg")
        if os.path.exists(image_path):
            images_extraites.append(image_path)
            continue
        time = seconde + frame * (1 / 25)
        cmd = ['ffmpeg', '-ss', str(time), '-i', video_path, '-frames:v', '1', '-q:v', '2', image_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.write(f"Erreur FFmpeg à {time} seconde : {result.stderr.decode('utf-8')}")
            break
        images_extraites.append(image_path)
    return images_extraites

# Fonction d'analyse d'émotion et d'annotation d'une image
def analyser_et_annoter_image(image_path, detector):
    if image_path is None:
        return {}
    image = cv2.imread(image_path)
    if image is None:
        return {}
    resultats = detector.detect_emotions(image)
    if resultats:
        return resultats[0]['emotions']
    return {}

# Calcul de l'émotion dominante par moyenne des scores
def emotion_dominante_par_moyenne(emotions_list):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    moyenne_emotions = {emotion: np.mean([emo.get(emotion, 0) for emo in emotions_list]) for emotion in emotions}
    emotion_dominante = max(moyenne_emotions, key=moyenne_emotions.get)
    return moyenne_emotions, emotion_dominante

# Calcul de l'émotion dominante par mode
def emotion_dominante_par_mode(emotions_list):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotions_dominantes = Counter({emotion: 0 for emotion in emotions})
    for emotion_dict in emotions_list:
        if emotion_dict:
            emotion_max = max(emotion_dict, key=emotion_dict.get)
            emotions_dominantes[emotion_max] += 1
    return emotions_dominantes, emotions_dominantes.most_common(1)[0][0]

# Calcul de la moyenne et de la variance des émotions
def moyenne_et_variance_par_emotion(emotions_25fps_list):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    resultats = {}
    for emotion in emotions:
        emotion_scores = [emotion_dict.get(emotion, 0) for emotion_dict in emotions_25fps_list]
        moyenne = np.mean(emotion_scores)
        variance = np.var(emotion_scores)
        resultats[emotion] = {'moyenne': moyenne, 'variance': variance}
    return resultats

####
# Fonction principale pour gérer le processus
def analyser_video(video_url, start_time, end_time, repertoire_travail):
    st.write(f"Analyse de la vidéo entre {start_time} et {end_time} seconde(s)")
    repertoire_25fps = os.path.join(repertoire_travail, "images_annotées_25fps")
    os.makedirs(repertoire_25fps, exist_ok=True)
    video_path = telecharger_video(video_url, repertoire_travail)
    detector = FER(mtcnn=True)

    results_25fps = []
    emotion_dominante_moyenne_results = []
    emotion_dominante_mode_results = []

    for seconde in range(start_time, end_time + 1):
        images_25fps = extraire_images_25fps_ffmpeg(video_path, repertoire_25fps, seconde)
        emotions_25fps_list = [analyser_et_annoter_image(image_path, detector) for image_path in images_25fps]

        if emotions_25fps_list:
            for idx, emotions in enumerate(emotions_25fps_list):
                results_25fps.append({'Seconde': seconde, 'Frame': f'25fps_{seconde * 25 + idx}', **emotions})

        # Calcul de la moyenne des émotions sur les 25 frames
        # Calcul de la moyenne des émotions pour chaque seconde
        moyenne_emotions, emotion_dominante_moyenne = emotion_dominante_par_moyenne(emotions_25fps_list)
        emotion_dominante_moyenne_results.append({
            'Seconde': seconde,
            **moyenne_emotions  # Ajoute la moyenne des émotions à chaque seconde
        })

        # Calcul du mode des émotions sur les 25 frames
        mode_emotions, emotion_dominante_mode = emotion_dominante_par_mode(emotions_25fps_list)
        emotion_dominante_mode_results.append({
            'Seconde': seconde,
            'Emotion_dominante_25fps_mode': emotion_dominante_mode,
            **mode_emotions
        })

    # DataFrame pour les résultats des 25 fps (pour chaque frame)
    df_emotions = pd.DataFrame(results_25fps)
    st.write("#### Analyse des émotions image par image (25fps)")
    st.dataframe(df_emotions)

    # DataFrame pour les résultats de la moyenne des émotions par seconde
    df_emotion_dominante_moyenne = pd.DataFrame(emotion_dominante_moyenne_results)
    st.write("#### Résultat de l'émotion dominante par moyenne (25fps)")
    st.dataframe(df_emotion_dominante_moyenne)

    # DataFrame pour les résultats du mode des émotions par seconde
    df_emotion_dominante_mode = pd.DataFrame(emotion_dominante_mode_results)
    st.write("#### Résultat de l'émotion dominante par mode (25fps)")
    st.dataframe(df_emotion_dominante_mode)
#####
    # Assurez-vous que les données de df_emotions sont correctes et que la colonne Frame_Index est bien créée
    df_emotions['Frame_Index'] = df_emotions.apply(lambda x: x['Seconde'] * 25 + int(x['Frame'].split('_')[1]), axis=1)

    # Création du DataFrame df_streamgraph en transformant df_emotions pour les graphiques
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # Préparation des données pour le Streamgraph et le Line chart (avec les moyennes)
    df_streamgraph = df_emotion_dominante_moyenne.melt(
        id_vars=['Seconde'],  # Colonne à conserver
        value_vars=emotions,  # Colonnes des émotions
        var_name='Emotion',  # Nom de la colonne pour les émotions
        value_name='Score'  # Nom de la colonne pour les scores des émotions
    )

    # Streamgraph
    # Streamgraph pour les moyennes des émotions par seconde
    streamgraph = alt.Chart(df_streamgraph).mark_area().encode(
        x=alt.X('Seconde:Q', title=f'Secondes (de {start_time} à {end_time})'),
        y=alt.Y('Score:Q', title='Score des émotions', stack='center'),
        color=alt.Color('Emotion:N', title='Émotion'),
        tooltip=['Seconde', 'Emotion', 'Score']
    ).properties(
        title='Streamgraph des émotions par seconde',
        width=800,
        height=400
    )
    st.write("#### Streamgraph des émotions par seconde")
    st.altair_chart(streamgraph, use_container_width=True)

    # Line chart pour les moyennes des émotions par seconde
    line_chart = alt.Chart(df_streamgraph).mark_line().encode(
        x=alt.X('Seconde:Q', title=f'Secondes (de {start_time} à {end_time})'),
        y=alt.Y('Score:Q', title='Score des émotions'),
        color=alt.Color('Emotion:N', title='Émotion'),
        tooltip=['Seconde', 'Emotion', 'Score']
    ).properties(
        title="Évolution des scores d'émotions par seconde",
        width=800,
        height=400
    )
    st.write("#### Line chart des émotions par seconde")
    st.altair_chart(line_chart, use_container_width=True)

    # Calcul et affichage des moyennes et variances sur toutes les secondes
    stats_par_emotion = moyenne_et_variance_par_emotion(results_25fps)
    if stats_par_emotion:
        df_stats = pd.DataFrame(stats_par_emotion).T.reset_index()
        df_stats.columns = ['Emotion', 'Moyenne', 'Variance']
        st.write("#### Tableau des moyennes et variances des émotions sur toutes les frames")
        st.dataframe(df_stats)

        moyenne_bar = alt.Chart(df_stats).mark_bar().encode(
            x=alt.X('Emotion:N', title='Émotion'),
            y=alt.Y('Moyenne:Q', title='Moyenne des probabilités'),
            color=alt.Color('Emotion:N', legend=None)
        )
        variance_point = alt.Chart(df_stats).mark_circle(size=100, color='red').encode(
            x=alt.X('Emotion:N', title='Émotion'),
            y=alt.Y('Variance:Q', title='Variance des probabilités'),
            tooltip=['Emotion', 'Variance']
        )
        graphique_combine = alt.layer(moyenne_bar, variance_point).resolve_scale(
            y='independent'
        ).properties(
            width=600,
            height=400,
        )
        st.altair_chart(graphique_combine, use_container_width=True)

        # Ajout du fichier Excel avec un tableau pour chaque seconde
        with pd.ExcelWriter(os.path.join(repertoire_travail, "resultats_emotions_par_seconde.xlsx")) as writer:
            # Pour chaque seconde analysée, créer une feuille dans le fichier Excel
            for seconde in range(start_time, end_time + 1):
                # Filtrer les résultats pour la seconde actuelle
                df_seconde = df_emotions[df_emotions['Seconde'] == seconde]

                # Créer une feuille Excel pour chaque seconde avec toutes les frames et les scores des émotions
                df_seconde.to_excel(writer, sheet_name=f"Seconde_{seconde}", index=False)

        st.write("Les résultats pour chaque seconde ont été exportés dans un fichier Excel.")

# Interface Streamlit
st.title("Analyse des émotions : 25 fps - moyenne - mode - variance")
st.markdown("<h6 style='text-align: center;'>www.codeandcortex.fr</h5>", unsafe_allow_html=True)

# Utilisation dans Streamlit
st.subheader("Définir le répertoire de travail")
repertoire_travail = definir_repertoire_travail()

video_url = st.text_input("URL de la vidéo à analyser", "", key="video_url")
start_time = st.number_input("Temps de départ de l'analyse (en secondes)", min_value=0, value=0, key="start_time")
end_time = st.number_input("Temps d'arrivée de l'analyse (en secondes)", min_value=start_time, value=start_time + 1, key="end_time")

if st.button("Lancer l'analyse"):
    if video_url and repertoire_travail:
        vider_cache()
        analyser_video(video_url, start_time, end_time, repertoire_travail)
    else:
        st.write("Veuillez définir le répertoire de travail et l'URL de la vidéo.")

st.markdown("""
    ### Explication des différents résultats :

    - **Analyse des émotions image par image (25fps)** : Analyse des émotions pour chaque image extraite à une fréquence de 25 images par seconde.
    - **Émotion dominante par détection automatique (FER `detect`)** : La fonction native de FER `detect` extrait l'émotion avec le score le plus élevé pour chaque seconde analysée. Ce score est calculé en prenant la valeur la plus élevée parmi les 25 images extraites pour chaque seconde.
    - **Émotion dominante par moyenne (25fps)** : La moyenne des scores des émotions est calculée pour chaque seconde en prenant en compte les 25 images. L'émotion avec la moyenne la plus élevée est considérée comme dominante.
    - **Émotion dominante par mode (25fps)** : Le mode des émotions est calculé sur les 25 images pour chaque seconde. L'émotion qui apparaît le plus souvent comme dominante est affichée.

    #### Visualisations :

    - **Line chart de l'émotion dominante (détection automatique)** : Ce graphique montre l'émotion dominante à chaque seconde basée sur la valeur la plus élevée parmi les 25 frames.
    - **Streamgraph des émotions par seconde (détection automatique)** : Ce graphique montre la répartition des scores des différentes émotions à chaque seconde, basé sur les valeurs les plus élevées parmi les 25 frames.

    Ces visualisations permettent de comprendre les variations des émotions au cours du temps en fonction de l'analyse des 25 images extraites par seconde.
""")
