# AI Health Advisor

A Streamlit web app that estimates general health risk and mental health risk from user input, and helps find nearby hospitals using the Google Maps API.

> **Disclaimer:** This project is for educational purposes only. It is not a medical device and does not give medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

## Features

- **Health risk prediction** using a TensorFlow model trained on lifestyle data
- **Mental health risk prediction** using a separate TensorFlow model
- **Hospital finder** that locates nearby hospitals with the Google Maps API

## Project Structure

```
Health_Advisor/
├── Code/          # App.py (Streamlit app) and model training notebooks
├── Datasets/      # Datasets used for training
├── Documentation/ # Project documentation
└── src/           # Model code, hospital finder, and trained models
    └── models/
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Health-Advisor.git
cd AI-Health-Advisor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Google Maps API key

Create a file at `.streamlit/secrets.toml`:

```toml
GOOGLE_MAPS_API_KEY = "your-api-key-here"
```

This file is listed in `.gitignore` and must never be committed. You can get a key from the [Google Cloud Console](https://console.cloud.google.com/). Restrict it to only the APIs you use.

### 4. Run the app

```bash
streamlit run Code/App.py
```

## Datasets

- `synthetic_health_lifestyle_dataset.csv`: synthetic data generated for training
- `mental_health_dataset.csv`: [add the source and license here]

## Tech Stack

Python, Streamlit, TensorFlow, scikit-learn, Pandas, Google Maps API

## Notes

The `.pkl` files in `src/models` are Python pickles. Only load pickle files from sources you trust.
