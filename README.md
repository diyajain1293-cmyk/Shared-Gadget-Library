# BorrowBox Shared Gadget Library – Streamlit Dashboard

This app analyses the **Shared Gadget Library / BorrowBox** survey data and builds
classification models to predict who is willing to use the service.

## Files

- `app.py` – main Streamlit app
- `requirements.txt` – Python dependencies (no version pins)
- `Shared_Gadget_Library_Survey_Synthetic_Data.csv` – survey data (add your file here)

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Streamlit Cloud, add these files to a GitHub repo and point the app to `app.py`.
