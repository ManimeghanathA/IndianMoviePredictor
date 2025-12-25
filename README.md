# 🎬 Indian Movie Success Predictor (XGBoost)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](YOUR_STREAMLIT_LINK_HERE)

## 📌 Project Overview
A machine learning-driven application designed to predict the success of Indian movies ('Hit', 'Average', or 'Flop'). By leveraging a custom-curated dataset of 10,000+ films and advanced feature engineering like **Target Encoding** and **Weighted Crew Indices**, this project provides data-backed insights for filmmakers and distributors.

## 🚀 Key Features
- **Dynamic Success Score:** A stylized 1-10 success scale indicating the predicted commercial viability.
- **Model Explainability:** Real-time analysis of which parameters (Crew, Scale, Genre) are driving the prediction.
- **Advanced Feature Engineering:** Uses smoothed Target Encoding for high-cardinality data (Directors/Cast).
- **Promotion Scale Integration:** A user-friendly 1-10 slider that maps exponentially to market reach (IMDb Votes proxy).

## 🛠️ Technical Stack
- **Languages:** Python
- **ML Model:** XGBoost Classifier
- **Deployment:** Streamlit Community Cloud
- **Libraries:** Pandas, NumPy, Scikit-Learn, Pickle

## 🔬 Data & Methodology
The model was trained on a proprietary dataset (1920–2024) with a focus on:
1. **Weighted IMDb Score:** Defining success by balancing Rating and Popularity.
2. **Crew Popularity Index:** A weighted metric calculating the historical impact of directors (60%) and cast (40%).
3. **Handling Class Imbalance:** Utilized Inverse Sample Weighting to ensure accurate 'Hit' and 'Flop' detection despite data skew.

## 📈 Model Performance
- **Overall Accuracy:** 95.5%
- **Hit Recall:** 85% (The model correctly identifies 85% of successful films).
- **Macro F1-Score:** 0.78 (Demonstrating robust performance across all success categories).

## 💻 Installation & Usage
1. Clone the repo: `git clone https://github.com/ManimeghanathA/IndianMoviePredictor.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
