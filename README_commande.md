
# Creat venv
python -m venv sign2speak_venv
or
py -m venv sign2speak_venv

# Activate  venv
.\sign2speak_venv\Scripts\Activate

# Installer les dépendances
python -m pip install -r requirements.txt
# deactivate
deactivate

# freeze dependance 
python -m pip freeze > requirements.txt 

# For lunch the Projet
py .\keypoint_extractor.py

# For install the dataset 
 python dataset_pipeline.py --download --data-dir ./data
# For explore dataset and creat train/test/val split 
python dataset_pipeline.py --explore --data-dir ./data

# For lunch with 5 exemple 
python dataset_pipeline.py --extract --data-dir ./data --output ./keypoints --max-signs 5 --max-videos-per-sign 5
# For lunch keypoints with 50 exemples and all videos for eatch exemples 
python dataset_pipeline.py --extract --data-dir ./data --output ./keypoints_v2 --max-signs 50 --max-videos-per-sign 999
# For lunch the train data with LSTM
python train_classifier.py --data ./keypoints --output ./model

# for lunch live model
python live_test.py --model ./model



python live_winner.py --model ./model_winner
----
# For data already ready
pip install kaggle
kaggle competitions download -c asl-signs






# for lunch live_winner best model 

python live_winner.py --model ./model_winner



# pour installer le modele 
https://ai.google.dev/edge/litert-lm/android?hl=fr

Commande pour installer le modèle :


# Crée le dossier si nécessaire (l'app doit être installée d'abord)
adb shell mkdir -p /sdcard/Android/data/com.sign2speak.app/files/

# Push le modèle (2.58 GB — prend ~2-5 min selon le câble)
adb push gemma-4-E2B-it.litertlm /sdcard/Android/data/com.sign2speak.app/files/
Sans modèle, l'app fonctionne normalement en fallback (glosses → texte direct). Avec le modèle présent, Gemma 4 E2B chargera en arrière-plan au démarrage et sera prêt avant le premier signe.
adb shell ls -lh /sdcard/Android/data/com.sign2speak.app/files/