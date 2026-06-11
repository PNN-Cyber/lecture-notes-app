# 🎓 Lecture Notes Converter

> Upload any lecture recording → get structured study notes, an executive summary, and a typeset PDF — all powered by local Whisper transcription and Google Gemini 2.5 Flash.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Local Transcription** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper `base` model) — runs entirely on your machine, no audio data leaves your device |
| **AI Study Notes** | Gemini 2.5 Flash generates fully structured, Markdown-formatted notes with headings, bullets, bold key terms, and a Key Takeaways section |
| **Executive Summary** | Concise ≤250-word prose summary capturing the lecture's thesis and core conclusions |
| **Beautiful PDF Export** | ReportLab Platypus typesetter outputs a high-contrast academic PDF with proper Markdown rendering |
| **Wide-Layout Dashboard** | Polished Streamlit UI with a three-tab workspace (Notes · Summary · Transcript), live metrics, and a one-click PDF download |
| **Smart Caching** | Whisper model and GenAI client are cached in `st.session_state` — re-running the UI never re-triggers expensive inference |
| **GPU Auto-Detection** | Automatically uses CUDA float16 when a GPU is present; falls back to CPU int8 gracefully |

---

## 🗂️ Project Structure

```
lecture-notes-app/
│
├── requirements.txt      # All Python dependencies
├── .gitignore            # Ignores venv, .env, uploads/, generated_pdfs/
├── .env.example          # Environment variable template
│
├── transcriber.py        # AudioTranscriber — faster-whisper wrapper
├── ai_generator.py       # Gemini 2.5 Flash — notes & summary generation
├── pdf_export.py         # ReportLab PDF typesetting engine
├── app.py                # Streamlit front-end (main entry point)
│
└── README.md             # This file
```

---

## 🛠️ Local Installation

### Prerequisites

- Python **3.10 or higher**
- `ffmpeg` installed on your system (required by faster-whisper for audio decoding)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/lecture-notes-app.git
cd lecture-notes-app
```

### Step 2 — Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate (macOS / Linux)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** if you have an NVIDIA GPU with CUDA, install the matching `torch` build first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

### Step 4 — Configure your API key

```bash
cp .env.example .env
```

Open `.env` in any editor and paste your **Google Gemini API key**:

```env
GOOGLE_API_KEY=AIzaSy...your_key_here
```

Obtain a free API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🔑 Environment Configuration

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Google AI Studio API key for Gemini 2.5 Flash |

The app resolves the key using this priority order:

1. **`st.secrets["GOOGLE_API_KEY"]`** — used on Streamlit Community Cloud
2. **`os.getenv("GOOGLE_API_KEY")`** — used locally via `.env` / shell export

---

## ☁️ Deploying to Streamlit Community Cloud

### Step 1 — Push to GitHub

Ensure your repository is public (or the account has private repo access) and that `.env` is listed in `.gitignore` (it is by default).

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2 — Create a new app on Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set the **Main file path** to `app.py`.

### Step 3 — Add your API key via Advanced Settings → Secrets

1. In the deployment dialog click **"Advanced settings"**.
2. Under the **Secrets** section, paste the following in TOML format:

```toml
GOOGLE_API_KEY = "AIzaSy...your_key_here"
```

3. Click **Save** and then **Deploy**.

> **Important:** Never commit `secrets.toml` or your `.env` file.  
> The `.gitignore` in this project already excludes both.

### Step 4 — Wait for the build

Streamlit Cloud will install all packages from `requirements.txt` and launch the app.  The first boot downloads the Whisper `base` model (~150 MB), so it may take 2–3 minutes on a cold start.

---

## 🧩 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                      app.py                          │
│  (Streamlit UI · session_state caching · cleanup)    │
└──────────┬──────────────────────────────┬────────────┘
           │                              │
   ┌───────▼────────┐          ┌──────────▼──────────┐
   │ transcriber.py │          │   ai_generator.py   │
   │ AudioTranscriber│         │ generate_notes()    │
   │ faster-whisper │          │ generate_summary()  │
   │ CPU/GPU detect │          │ Gemini 2.5 Flash    │
   └───────┬────────┘          └──────────┬──────────┘
           │    transcript (str)           │  notes + summary (str)
           └─────────────┬────────────────┘
                         │
                ┌────────▼────────┐
                │  pdf_export.py  │
                │ export_to_pdf() │
                │ ReportLab       │
                │ Markdown→PDF    │
                └────────┬────────┘
                         │  /generated_pdfs/lecture_*.pdf
                         ▼
                  st.download_button
```

---

## 🐛 Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: ffmpeg` | Install ffmpeg (see Prerequisites above) |
| `EnvironmentError: GOOGLE_API_KEY not found` | Check `.env` exists and contains the key; or add it to Streamlit secrets |
| Very slow transcription | Running on CPU; add a CUDA-capable GPU or use a smaller model size in `transcriber.py` |
| PDF download is empty | Check `generated_pdfs/` directory permissions; ensure ReportLab installed correctly |
| `RuntimeError: Whisper transcription failed` | Ensure the audio file is not corrupted; try re-exporting to WAV |

---

## 📄 License

MIT — see `LICENSE` for details.
