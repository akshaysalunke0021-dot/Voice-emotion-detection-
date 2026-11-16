# Voice Emotion Detection Website

## Run on your computer
1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Open the local address shown in the terminal (usually http://localhost:8501).

The first run downloads the pretrained model `superb/wav2vec2-base-superb-er`; an internet connection is required and the initial load may take a few minutes.

## Put it online
Upload `app.py` and `requirements.txt` to a GitHub repository, then deploy that repository using Streamlit Community Cloud and choose `app.py` as the entry point.

## Limitations
The pretrained model predicts four broad vocal-expression classes: happy, sad, angry and neutral. It does not determine a person's actual internal feelings. Performance can vary by language, accent, speaker and recording conditions. Use only consented recordings and never use predictions for high-stakes decisions.
