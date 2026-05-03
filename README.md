

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

# For lunch the Projet
py .\keypoint_extractor.py

# For install the dataset 
 python dataset_pipeline.py --download --data-dir ./data
# For explore dataset and creat train/test/val split 
python dataset_pipeline.py --explore --data-dir ./data

# For lunch with 5 exemple 
python dataset_pipeline.py --extract --data-dir ./data --output ./keypoints --max-signs 5 --max-videos-per-sign 5