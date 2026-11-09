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
pip install numpy pandas matplotlib scikit-learn joblib requests anthropic jupyter
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
