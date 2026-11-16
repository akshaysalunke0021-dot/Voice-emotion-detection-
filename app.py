import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import pipeline

st.set_page_config(page_title="VoiceSense | Emotion Detection", page_icon="🎙️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f6f7fb;color:#172033}
.block-container{max-width:1120px;padding-top:2rem;padding-bottom:3rem}
.hero{background:linear-gradient(125deg,#171b36,#34316f 62%,#1b7d7a);border-radius:24px;padding:34px 38px;color:white;margin-bottom:24px;box-shadow:0 16px 40px #1f234b22}
.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:700;color:#b9c6ff}
.hero h1{font-family:Manrope,sans-serif;font-size:clamp(30px,4vw,46px);line-height:1.08;margin:12px 0;color:white}
.hero p{color:#e2e7ff;max-width:700px;font-size:16px;margin-bottom:0}
.pill{display:inline-block;background:#ffffff1f;border:1px solid #ffffff33;padding:7px 12px;border-radius:30px;font-size:12px;margin-top:18px}
.panel{background:white;border:1px solid #e7e9f1;border-radius:18px;padding:22px;box-shadow:0 6px 20px #14193c0a}
.smallcap{color:#6b7280;text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:700}
.stButton>button{background:#48418e;color:white;border:0;border-radius:12px;padding:.65rem 1.1rem;font-weight:700}
.stButton>button:hover{background:#39336f;color:white;border:0}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the AI model… the first load may take a few minutes.")
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)

def prepare_audio(file):
    file.seek(0)
    audio, sr = sf.read(file)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = audio[:12 * sr]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    if peak > 0:
        audio = audio / peak
    return audio, sr

def label_name(label):
    return {"neu":"Neutral","hap":"Happy","ang":"Angry","sad":"Sad",
            "neutral":"Neutral","happy":"Happy","angry":"Angry"}.get(label.lower(), label.title())

def emoji(name):
    return {"Happy":"😊","Sad":"😢","Angry":"😠","Neutral":"😐"}.get(name,"🎧")

st.markdown("""
<div class="hero">
 <div class="eyebrow">AI • Speech Processing • Machine Learning</div>
 <h1>Voice Emotion<br>Detection System</h1>
 <p>Record or upload a short voice sample. The AI model analyses vocal patterns and estimates the emotion expressed in the speech.</p>
 <div class="pill">🎙️ Live recording &nbsp; • &nbsp; 📁 Audio upload &nbsp; • &nbsp; 🧠 AI prediction</div>
</div>
""", unsafe_allow_html=True)

left,right=st.columns([1.1,.9],gap="large")
with left:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("### 1. Add a voice sample")
    st.write("Record a short sentence or upload an audio file. Speak clearly in a quiet place.")
    recording=st.audio_input("Record your voice")
    uploaded=st.file_uploader("Or upload audio",type=["wav","flac"])
    st.caption("WAV and FLAC uploads supported. Recordings are not saved by this app.")
    st.markdown('</div>',unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("### How it works")
    for n,title,desc in [("01","Record","Capture a short speech sample."),
                         ("02","Process","Convert audio to a consistent format."),
                         ("03","Analyse","AI analyses vocal patterns."),
                         ("04","Predict","View the likely vocal-emotion class.")]:
        st.markdown(f"""<div style="display:flex;gap:12px;margin:13px 0;align-items:flex-start">
        <div style="background:#eceaf8;color:#48418e;border-radius:10px;padding:8px 10px;font-weight:800">{n}</div>
        <div><b>{title}</b><div style="color:#687086;font-size:13px">{desc}</div></div></div>""",unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

audio_file=recording if recording is not None else uploaded
if audio_file is not None:
    st.audio(audio_file)
    if st.button("Analyse voice emotion",type="primary",use_container_width=True):
        try:
            audio,sr=prepare_audio(audio_file)
            if len(audio)<int(.5*sr):
                st.error("Please provide at least half a second of speech.")
            elif float(np.max(np.abs(audio)))<1e-4:
                st.error("The recording appears silent. Please try again.")
            else:
                with st.spinner("Analysing vocal patterns…"):
                    result=load_model()({"raw":audio,"sampling_rate":sr},top_k=4)
                top=result[0]
                name=label_name(top["label"])
                score=float(top["score"])
                st.markdown("## 2. Analysis result")
                a,b=st.columns(2)
                with a:
                    st.markdown('<div class="panel">',unsafe_allow_html=True)
                    st.markdown(f"<div class='smallcap'>Most likely vocal expression</div><div style='font-size:52px;margin:8px 0'>{emoji(name)}</div><h2 style='color:#48418e;margin:0'>{name}</h2><p style='color:#6b7280'>Model score: <b>{score*100:.1f}%</b></p>",unsafe_allow_html=True)
                    st.markdown('</div>',unsafe_allow_html=True)
                with b:
                    st.markdown('<div class="panel">',unsafe_allow_html=True)
                    st.markdown("#### Model scores")
                    for item in result:
                        nm=label_name(item["label"]); val=float(item["score"])
                        st.write(f"**{nm}** — {val*100:.1f}%")
                        st.progress(min(max(val,0),1))
                    st.markdown('</div>',unsafe_allow_html=True)
                st.info("This estimates vocal expression; it does not reveal someone's true feelings. Accent, language, background noise, speaker differences and training data can affect results.")
        except Exception as exc:
            st.error("Audio analysis failed. Check your internet connection (the model downloads on first use) and try a clear WAV or FLAC recording.")
            with st.expander("Technical details"):
                st.code(str(exc))

st.divider()
c1,c2=st.columns(2)
with c1:
    st.markdown("### About this project")
    st.write("An educational speech emotion recognition demonstration using audio preprocessing and a pretrained Wav2Vec2-based classifier.")
with c2:
    st.markdown("### Responsible use")
    st.write("Use recordings only with the speaker's permission. Do not use predictions for medical, employment, disciplinary or other high-stakes decisions.")
st.caption("Model: superb/wav2vec2-base-superb-er • Output classes: happy, sad, angry and neutral • Educational demonstration")
