# PharmaSignal: Drug-Quality Recall Intelligence

MSc capstone prototype that turns openFDA drug-recall records into evidence for pharmaceutical procurement teams. Everything is in `Drug_Recall_Intelligence_Final.ipynb`.

## Files

| File | What it is |
|---|---|
| `Drug_Recall_Intelligence_Final.ipynb` | The full analysis notebook |
| `drug_recalls.csv` | openFDA drug enforcement snapshot |
| `labelled_recall_reasons.csv` | 1,000 labelled recall reasons for the classifier |
| `blind_check_150.csv` | Blind human-check sheet (optional) |

Keep all files in the same folder as the notebook. Results are written to `outputs/`.

## Setup

Install Python 3.10 or newer, then from the project folder:

```bash
pip install -r requirements.txt
```

## Connecting Claude (optional)

Without an API key the notebook still runs fully; the assistant section answers in **evidence-only mode**. To get AI-written answers, give the notebook an Anthropic API key (create one at https://console.anthropic.com → API Keys).

**Easiest (Mac and Windows):** create a file called `.env` next to the notebook containing one line:

```
ANTHROPIC_API_KEY=sk-ant-...
```

You can copy `.env.example` to `.env` and edit it. The notebook loads it automatically, and `.gitignore` keeps it out of GitHub. **Never commit or share your key.**

> **Windows / Notepad tip:** in *Save As*, set *Save as type* to **All Files (\*.\*)** and type the name as `".env"` (with the quotes). Otherwise Notepad saves it as `.env.txt` and it will not be found.

**Alternative: environment variable** (restart Jupyter afterwards):

- Mac/Linux: `export ANTHROPIC_API_KEY="sk-ant-..."` then start `jupyter notebook` from the same terminal
- Windows: `setx ANTHROPIC_API_KEY "sk-ant-..."` then close and reopen all terminals and Jupyter

## Running

Open the notebook and choose **Kernel → Restart & Run All** (about 2 minutes; a little longer with Claude connected). Section 8 should print `LLM: Claude (claude-sonnet-5-5) connected` when the key is found.

To check whether the key is visible to the notebook, run in a cell:

```python
import os; print(bool(os.environ.get("ANTHROPIC_API_KEY")))
```

## Streamlit app

An interactive web app built on the notebook's results, with four pages:

| Page | What it shows |
|---|---|
| **Overview** | Key numbers, recalls per year, failure types, dosage forms, top firms, record search |
| **Anomaly alerts** | Surge alerts by failure type, dosage form and firm, with a trend chart and the back-test |
| **Tender screening** | Enter bidders and a product to get downloadable evidence briefs, plus a watch list of firms |
| **Ask the assistant** | Plain-English questions answered by Claude using only the data, with sources and the fact-checker |

1. **Run the notebook once** (Restart & Run All). It creates the `outputs/` folder the app reads; the app does not retrain anything.
2. From the project folder, start the app:

   ```bash
   streamlit run app.py
   ```

   On Windows, if `streamlit` is not recognised, use `python -m streamlit run app.py`.
3. Your browser opens at http://localhost:8501. Stop the app with **Ctrl+C** in the terminal.

The assistant uses the same `.env` API key as the notebook. Without a key the app still works, and the assistant shows the evidence only. Re-run the notebook whenever the data changes, then restart the app.

The app code is in `app.py` (pages) and `pharmasignal_core.py` (data loading, tender screening and assistant logic, taken from the notebook).

## Hosting the app online (Streamlit Community Cloud, free)

The files in `outputs/` that the app reads are committed to the repo, so the hosted app works without running the notebook.

1. Go to https://share.streamlit.io and sign in with the GitHub account that owns the repo.
2. Click **Create app → Deploy a public app from GitHub** and fill in:
   - **Repository:** this repo · **Branch:** `master` · **Main file path:** `app.py`
   - **App URL:** choose a name, e.g. `pharmasignal` → `https://pharmasignal.streamlit.app`
3. Open **Advanced settings**:
   - **Python version:** 3.12
   - **Secrets:** paste the following (TOML format, values in quotes):

     ```toml
     ANTHROPIC_API_KEY = "sk-ant-..."
     APP_PASSWORD = "choose-a-password"
     ```

4. Click **Deploy**. The first build takes a few minutes.

**About the secrets:**
- `ANTHROPIC_API_KEY` turns on Claude answers. Leave it out and the app still works in evidence-only mode.
- `APP_PASSWORD` is optional but recommended. The app is public, so anyone with the link could use Claude on your API credit. With a password, visitors see all dashboards and evidence, but Claude answers need the password. Also set a monthly spend limit in the Anthropic Console.
- Secrets are stored by Streamlit, not in GitHub. Change them later under the app's **⋮ → Settings → Secrets**.

**Updating the hosted app:** push to `master`; the app redeploys automatically. If you re-run the notebook (for example with new data), commit the updated `outputs/` files too.

**Search quality:** the hosted app uses the TF-IDF + SVD search fallback unless you uncomment `sentence-transformers` in `requirements.txt`. That installs PyTorch, which makes the build slow and may exceed the free tier's memory.
