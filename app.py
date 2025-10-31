import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import datetime
import re
from collections import defaultdict
import os

# --- 1. CONFIGURATION AND ASSET LOADING ---

MODEL_PATH = 'movie_success_model_xgb.pkl'
ASSETS_PATH = 'model_assets_final.json'

# --- CRITICAL FIX: STATIC COLUMN ORDER ---
STATIC_EXPECTED_COLS = [
    'Year', 'Title_Length', 'Title_Word_Count', 'Has_Number', 
    'Genre_Action', 'Genre_Comedy', 'Genre_Drama', 'Genre_Thriller', 'Genre_Horror', 
    'Genre_Romance', 'Genre_Crime', 'Genre_Mystery', 'Genre_Adventure', 'Genre_Fantasy', 
    'Genre_Sci-fi', 'Genre_Historical', 'Genre_Biographical', 'Genre_War', 'Genre_Family', 
    'Genre_Musical', 'Genre_Sport', 'Genre_Animation', 'Genre_Western', 'Genre_Documentary', 
    'Genre_Political', 
    'Director_Quality_Score', 'runtime', 'log_imdb_votes', 'Crew_Popularity_Index', 
    'Language_Target_Score', 'Country_Target_Score', 'Festival_Release_Flag', 
    'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 
    'month_9', 'month_10', 'month_11', 'month_12'
]

# --- FEATURE NAME MAPPING DICTIONARY (NEW) ---
FEATURE_DISPLAY_NAMES = {
    'log_imdb_votes': 'Promotional Budget Scale',
    'Crew_Popularity_Index': 'Crew Quality Index (Weighted)',
    'Director_Quality_Score': 'Director Historical Score',
    'runtime': 'Movie Runtime (Minutes)',
    'Language_Target_Score': 'Language Success Rate',
    'Country_Target_Score': 'Country Success Rate',
    'Festival_Release_Flag': 'Festival Release Timing',
    # Map OHE features for clarity
    **{f'month_{m}': f'Release Month {m}' for m in range(2, 13)}
}


@st.cache_resource
def load_assets():
    """Loads the model, assets, and look-up lists."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ASSETS_PATH, 'r') as f:
            assets = json.load(f)
            
        all_directors = sorted([k for k in assets['historical_scores']['director'].keys() if k not in ['unknown', 'unknown director']])
        all_cast = sorted([k for k in assets['historical_scores']['cast'].keys() if k not in ['unknown cast', 'unknown']])
        all_languages = sorted([k for k in assets['historical_scores']['language'].keys() if k not in ['unknown language', 'unknown']])
        all_countries = sorted([k for k in assets['historical_scores']['country'].keys() if k not in ['unknown', 'unknown country']])
        all_genres = [col.split('_')[1].lower() for col in STATIC_EXPECTED_COLS if col.startswith('Genre_')]
        
        feature_names = model.get_booster().feature_names

        return model, assets, {
            'directors': all_directors, 'cast': all_cast, 'languages': all_languages,
            'countries': all_countries, 'genres': all_genres, 'feature_names': feature_names
        }
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Please ensure {MODEL_PATH} and {ASSETS_PATH} exist.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

model, assets, lookup_lists = load_assets()

# --- 2. CORE FEATURE ENGINEERING AND SCORING FUNCTIONS ---
# (Helper functions remain the same)
def safe_list_parse(text, delimiter='|'):
    if isinstance(text, list): return [str(t).lower().strip() for t in text]
    if pd.isna(text) or str(text).strip() == '': return []
    return [item.lower().strip() for item in str(text).split(delimiter) if item.strip()]

def get_single_input_score(entities, score_map, global_mean, top_n=None):
    if not entities: return global_mean
    if top_n is not None: entities = entities[:top_n]
    
    scores = []
    for entity in entities:
        score = score_map.get(entity, score_map.get('unknown', global_mean))
        scores.append(score)
        
    return np.mean(scores) if scores else global_mean


def engineer_features(user_data):
    """Transforms raw user input into the final numerical feature vector (X_predict)."""
    
    hs = assets['historical_scores']
    globals_ = assets['globals']
    GM_RATING = globals_['global_mean_rating']
    GM_SCORE = globals_['global_mean_weighted_score']
    
    # --- Calculation ---
    director_score_base = get_single_input_score(user_data['directors'], hs['director'], GM_RATING, top_n=None)
    cast_score_base = get_single_input_score(user_data['cast'], hs['cast'], GM_RATING, top_n=5)
    
    crew_index = (director_score_base * 0.6) + (cast_score_base * 0.4)
    
    lang_score = get_single_input_score(user_data['languages'], hs['language'], GM_SCORE)
    country_score = get_single_input_score(user_data['countries'], hs['country'], GM_SCORE)

    runtime_capped = min(user_data['runtime'], 300)
    
    # *** PROMOTIONAL SCALE MAPPING ***
    # This calculation MUST match the formula used in the analyze_feature_importance function 
    # to maintain consistency for the explanation.
    scale_value = user_data['promotion_scale']
    
    synthetic_votes = 100 * (scale_value ** 3) 
    synthetic_votes = min(synthetic_votes, 500000)
    
    log_imdb_votes_value = np.log1p(synthetic_votes)
    # *********************************
    
    release_month = user_data['release_date'].month
    festival_flag = 1 if release_month in [1, 3, 6, 7, 8, 10, 11, 12] else 0 

    # --- Build the Raw Feature Row (Dictionary) ---
    data_row = {
        'Year': user_data['release_date'].year,
        'Title_Length': len(user_data['title']),
        'Title_Word_Count': len(user_data['title'].split()),
        'Has_Number': 1 if re.search(r'\d', user_data['title']) else 0,
        'Director_Quality_Score': director_score_base, 
        'runtime': runtime_capped, 
        'log_imdb_votes': log_imdb_votes_value, # The mapped feature
        'Crew_Popularity_Index': crew_index,
        'Language_Target_Score': lang_score,
        'Country_Target_Score': country_score,
        'Festival_Release_Flag': festival_flag,
    }
    
    # Add Genre Binary Flags
    for genre in lookup_lists['genres']:
        genre_key = f'Genre_{genre.capitalize()}'
        data_row[genre_key] = 1 if genre in user_data['genres'] else 0
        
    # Add OHE for Release Month
    for m in range(2, 13): 
        data_row[f'month_{m}'] = 1 if release_month == m else 0

    # --- Order Enforcement (CRITICAL STEP) ---
    X_predict = pd.DataFrame([data_row])
    
    for col in STATIC_EXPECTED_COLS:
        if col not in X_predict.columns:
            X_predict[col] = 0
    
    return X_predict[STATIC_EXPECTED_COLS]


# --- 3. EXPLAINABILITY AND PRESENTATION LOGIC ---

def get_success_scale(proba_hit, proba_flop):
    """Calculates a success score out of 10 based on probabilities."""
    score = 5 + (proba_hit - proba_flop) * 5.5 
    return max(1, min(10, round(score)))

def analyze_feature_importance(model, X_predict, predicted_class, promotion_scale_input):
    """
    Analyzes model feature importance (Gain) to provide explanations and recommendations,
    using the actual user input (promotion_scale_input) for the log_imdb_votes feature explanation.
    """
    
    importance = model.feature_importances_
    features = lookup_lists['feature_names']
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    top_5_features_tech = importance_df.head(5)['Feature'].tolist()
    top_5_features_display = [FEATURE_DISPLAY_NAMES.get(f, f) for f in top_5_features_tech]
    
    explanation = []
    
    # --- Analysis Logic ---
    crew_value = X_predict['Crew_Popularity_Index'].iloc[0]
    dir_score = X_predict['Director_Quality_Score'].iloc[0]
    
    # Driver 1: Crew Quality
    if 'Crew_Popularity_Index' in top_5_features_tech or 'Director_Quality_Score' in top_5_features_tech:
        if predicted_class == 'Hit':
             explanation.append(f"⭐ **Crew Index:** The high combined historical success of your creative team ({crew_value:.3f}) is a major driving factor, providing strong confidence in film quality.")
        elif predicted_class == 'Flop' and crew_value < 6.0:
             explanation.append(f"⚠️ **Crew Quality Gap:** The overall team quality score ({crew_value:.3f}) is below average. This indicates a high-risk project that relies heavily on unexpected quality to succeed.")
    
    # Driver 2: Promotional Scale (Uses the original UI input value for readability)
    if 'log_imdb_votes' in top_5_features_tech:
        scale_value_readable = promotion_scale_input # Use the actual input value
        
        if predicted_class == 'Hit' and scale_value_readable >= 7:
             explanation.append(f"🔥 **Promotional Budget Scale (Level {scale_value_readable}/10):** The high investment signal strongly reinforces the Hit prediction, indicating high market reach.")
        elif predicted_class == 'Flop' and scale_value_readable <= 4:
             explanation.append(f"🧊 **Market Reach:** The low promotional scale (Level {scale_value_readable}/10) is a critical weak point, limiting potential box office appeal.")

    # Driver 3: Release Timing and Genre
    if 'Festival_Release_Flag' in top_5_features_tech and X_predict['Festival_Release_Flag'].iloc[0] == 1:
        explanation.append("🗓️ **Release Timing Advantage:** Releasing during a major holiday/festival window is providing a positive lift to the predicted outcome.")
    
    input_genres = [g for g in lookup_lists['genres'] if X_predict[f'Genre_{g.capitalize()}'].iloc[0] == 1]
    if len(input_genres) == 1 and 'drama' in input_genres and predicted_class == 'Average':
        explanation.append("🎭 **Genre Mix:** The model notes that pure Drama films often have a high floor but struggle to cross into the 'Hit' category without significant star power.")

    # Recommendation for Improvement (If Flop/Average)
    if predicted_class in ['Flop', 'Average']:
        lowest_score_features = ['Director_Quality_Score', 'Language_Target_Score', 'Country_Target_Score']
        lowest_performing_feature = X_predict[lowest_score_features].idxmin(axis=1).iloc[0]
        
        if lowest_performing_feature == 'Director_Quality_Score' and dir_score < 6.2:
            explanation.append(f"\n**💡 Recommendation:** The Director's historical score is low ({dir_score:.3f}). Consider securing a director with a stronger historical track record.")
    
    return explanation, top_5_features_display


# --- 4. STREAMLIT UI DESIGN ---

st.set_page_config(
    page_title="Indian Movie Success Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Indian Movie Success Predictor (XGBoost)")
st.markdown("Predicts success based on Crew Quality, Promotional Scale, and Genre using advanced Target Encoding.")
st.markdown("---")

# --- Form for user input ---
with st.form(key='prediction_form'):
    
    # --- Row 1: Title and Release ---
    col1, col2, col3 = st.columns([3, 1.5, 1.5])
    with col1:
        title_input = st.text_input("Movie Title", placeholder="e.g., Jawan, RRR", help="Used to generate Title Length/Word Count features.")
    with col2:
        release_date = st.date_input("Planned Release Date", value=datetime.date.today(), min_value=datetime.date(2023, 1, 1))
    with col3:
        runtime_input = st.number_input("Runtime (Minutes)", min_value=30, max_value=300, value=135, step=5)
    
    st.markdown("---")
    
    # --- Row 2: Crew and Promotional Scale (The key features) ---
    col4, col5, col6 = st.columns(3)
    with col4:
        director_options = lookup_lists['directors']
        director_input = st.multiselect("Director(s) (For Quality Score)", options=director_options, default=None, help="The FIRST director selected determines the Director Quality Score.")
    with col5:
        cast_options = lookup_lists['cast']
        cast_input = st.multiselect("Main Cast (For Crew Index)", options=cast_options, default=None, help="Scores of the top 5 actors will be averaged for the Crew Index.")
    with col6:
        # --- CRITICAL CHANGE: SLIDER FOR PROMOTIONAL SCALE ---
        promotion_scale_input = st.slider(
            "Promotional Budget Scale (1-10)", 
            min_value=1, max_value=10, value=5, 
            help="Estimate of marketing/release effort (1=Small, 10=Mega-Blockbuster).",
            key='promo_scale_slider' # Added a key for stability
        )

    st.markdown("---")
    
    # --- Row 3: Categorical Features ---
    col7, col8, col9 = st.columns(3)
    with col7:
        genre_input = st.multiselect("Select Genres", options=lookup_lists['genres'], default=['action', 'drama'], help="All selected genres create binary features.")
    with col8:
        language_input = st.multiselect("Primary Language(s)", options=lookup_lists['languages'], default=['hindi'], help="Used for Language Target Score.")
    with col9:
        country_input = st.multiselect("Production Country(s)", options=lookup_lists['countries'], default=['india'], help="Used for Country Target Score.")


    # --- Submit Button ---
    st.markdown("---")
    submitted = st.form_submit_button("Predict Movie Success", type="primary")

# --- 5. PREDICTION AND EXPLANATION LOGIC ---

if submitted:
    if not title_input or not director_input or not cast_input:
        st.error("⚠️ Please fill in the Movie Title, Director, and at least one Main Cast member.")
    else:
        # Prepare raw user data dictionary
        raw_user_data = {
            'title': title_input.lower(),
            'release_date': release_date,
            'runtime': runtime_input,
            'promotion_scale': promotion_scale_input, # Passed directly to the analysis function
            'directors': safe_list_parse(director_input),
            'cast': safe_list_parse(cast_input),
            'genres': safe_list_parse(genre_input),
            'languages': safe_list_parse(language_input),
            'countries': safe_list_parse(country_input),
        }
        
        # 1. Engineer the features
        with st.spinner("🚀 Engineering features and predicting..."):
            X_predict = engineer_features(raw_user_data)
        
        # 2. Make the prediction
        try:
            prediction_encoded = model.predict(X_predict)[0]
            prediction_proba = model.predict_proba(X_predict)[0]
        except Exception as e:
            st.error("Prediction Error: The structure of the input data did not match the model's training structure.")
            st.warning(f"Technical Detail: {e}")
            st.stop()
        
        # 3. Decode results
        predicted_class_name = assets['target_encoder'][str(int(prediction_encoded))]
        
        # Get individual probabilities 
        proba_map = {assets['target_encoder'][str(i)]: proba_value for i, proba_value in enumerate(prediction_proba)}
        
        # Calculate the success scale (1-10)
        success_scale = get_success_scale(proba_map.get('Hit', 0), proba_map.get('Flop', 0))

        # 4. Display Results
        
        # Determine color for the styled progress bar
        if predicted_class_name == 'Hit': 
            color = '#4CAF50' # Green
            st.balloons()
        elif predicted_class_name == 'Flop': 
            color = '#F44336' # Red
        else: 
            color = '#FFC107' # Yellow/Orange

        # Styled Progress Bar (Custom HTML for color)
        st.subheader(f"🎉 **{predicted_class_name.upper()}** Forecast")
        st.markdown(f"""
        <style>
        .stProgress > div > div > div > div {{
            background-color: {color};
        }}
        </style>
        <h3 style='text-align: center; color: {color}; margin-bottom: 0px;'>Success Score: {success_scale}/10</h3>
        """, unsafe_allow_html=True)
        st.progress(success_scale * 0.10)
        
        st.markdown(f"**Predicted Outcome:** The model forecasts this project's most likely outcome is **{predicted_class_name}**.")
        
        # Display detailed probabilities and Explanation side-by-side
        col_proba, col_gap, col_exp = st.columns([1.5, 0.1, 2.5])
        
        with col_proba:
            st.markdown("##### Probability Breakdown")
            proba_df = pd.DataFrame({
                'Category': list(proba_map.keys()),
                'Probability': [f"{v*100:.1f}%" for v in list(proba_map.values())]
            }).sort_values(by='Probability', ascending=False).reset_index(drop=True)
            st.dataframe(proba_df, use_container_width=True, hide_index=True)

        with col_exp:
            st.markdown("##### Model Explanation")
            # PASS THE ORIGINAL SCALE INPUT HERE
            explanations, top_features = analyze_feature_importance(model, X_predict, predicted_class_name, promotion_scale_input)
            
            if explanations:
                for exp in explanations:
                    st.markdown(f"{exp}")
            else:
                st.info("The model relies on a balanced mix of features for this prediction. No single factor dominates.")

            st.markdown(f"###### *Top 5 Influencers: {', '.join(top_features)}*")
