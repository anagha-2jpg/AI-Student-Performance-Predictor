# 🧠 Neural AI Student Performance Predictor v3.0

![Python](https://img.shields.io/badge/Python-3.8+-00D4FF?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-00D4FF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-EC4899?style=for-the-badge)

An AI-powered student grade prediction dashboard built with Streamlit, scikit-learn, and XGBoost — styled with a cyberpunk glassmorphism UI.

---

## Features

- **4 ML Models** — Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Real-time grade prediction** — A / B / C / F with per-model confidence scores
- **Deep Analysis page** — derived performance index, risk score, feature importance
- **Model Insights page** — accuracy/precision/recall/F1 comparison + confusion matrix
- **Save & export** — save student records and download them as JSON
- **Load records** — upload a previously exported JSON to restore student data

---

## Installation

```bash
git clone https://github.com/anagha-2jpg/AI-Student-Performance-Predictor.git
cd AI-Student-Performance-Predictor
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure

```
AI-Student-Performance-Predictor/
├── app.py               # Main application
├── requirements.txt     # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## How to Use

1. Open the app and click **👤 Student Info** in the sidebar.
2. Fill in the student's details (name, scores, attendance, etc.).
3. Click **🔮 Get AI Prediction** to see the grade prediction.
4. Visit **📊 Deep Analysis** for a breakdown of key factors.
5. Visit **⚡ Model Insights** to compare all four models.
6. Use **💾 Save Student** and **📥 Download Records** to export data.

---

## ML Details

| Model               | Notes                                      |
|---------------------|--------------------------------------------|
| Logistic Regression | Fast baseline, linear decision boundary    |
| Decision Tree       | Interpretable splits on feature thresholds |
| Random Forest       | 200 trees, best feature importance support |
| XGBoost             | Gradient-boosted trees, strong accuracy    |

The models are trained on 2 000 synthetically generated student records covering attendance, assignment/midterm/final scores, study hours, participation, discipline, engagement, and late submission count.

---

## Technologies

- **Frontend** — Streamlit, Plotly, CSS (glassmorphism / cyberpunk theme)
- **ML** — scikit-learn, XGBoost
- **Data** — pandas, NumPy

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

**Anagha Garode**  
GitHub: [@anagha-2jpg](https://github.com/anagha-2jpg)
