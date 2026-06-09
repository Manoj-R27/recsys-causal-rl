import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RecoSys", layout="wide")

@st.cache_data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                          names=['user_id','movie_id','rating','timestamp'])
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                         usecols=[0,1], names=['movie_id','title'])
    return ratings, movies

@st.cache_resource
def train_models(ratings):
    # Base matrix
    matrix = ratings.pivot_table(
        index='user_id', columns='movie_id', values='rating').fillna(0)
    
    # Stage 1 - SVD
    svd = TruncatedSVD(n_components=200, random_state=42)
    svd_matrix = svd.fit_transform(matrix)
    sim_df = pd.DataFrame(cosine_similarity(svd_matrix),
                          index=matrix.index, columns=matrix.index)
    
    # Stage 2 - IPS
    total = len(ratings)
    propensity = ratings.groupby('movie_id')['rating'].count() / total
    propensity = propensity.reset_index()
    propensity.columns = ['movie_id', 'propensity']
    propensity['ips_weight'] = 1 / propensity['propensity']
    propensity['ips_weight'] = propensity['ips_weight'] / propensity['ips_weight'].max()
    ratings_ips = ratings.merge(propensity[['movie_id','ips_weight']], on='movie_id')
    ratings_ips['weighted_rating'] = ratings_ips['rating'] * ratings_ips['ips_weight']
    weighted_matrix = ratings_ips.pivot_table(
        index='user_id', columns='movie_id',
        values='weighted_rating').fillna(0)
    svd_ips = TruncatedSVD(n_components=200, random_state=42)
    svd_ips_matrix = svd_ips.fit_transform(weighted_matrix)
    sim_ips_df = pd.DataFrame(cosine_similarity(svd_ips_matrix),
                              index=weighted_matrix.index,
                              columns=weighted_matrix.index)
    
    return matrix, sim_df, weighted_matrix, sim_ips_df

ratings, movies = load_data()
matrix, sim_df, weighted_matrix, sim_ips_df = train_models(ratings)

def get_recs(user_id, sim, mat, n=10):
    similar = sim[user_id].sort_values(ascending=False)[1:6].index
    watched = mat.loc[user_id][mat.loc[user_id] > 0].index
    scores = mat.loc[similar].mean()
    recs = scores.drop(watched).sort_values(ascending=False).head(n)
    return movies[movies['movie_id'].isin(recs.index)]['title'].values

st.title("🎬 Causal-Aware Recommendation Engine")
st.markdown("**SVD + Causal IPS Debiasing + RL Bandit** | by Manoj-R27")

tab1, tab2, tab3 = st.tabs(["Stage 1 — Recommendations", 
                              "Stage 2 — Bias Comparison", 
                              "Stage 3 — RL Bandit"])

with tab1:
    st.header("Personalised Movie Recommendations")
    user_id = st.slider("Select User ID", 1, 943, 1)
    if st.button("Get Recommendations"):
        recs = get_recs(user_id, sim_df, matrix)
        st.success(f"Top 10 movies for User {user_id}:")
        for i, movie in enumerate(recs, 1):
            st.write(f"{i}. {movie}")

with tab2:
    st.header("Popularity Bias — Before vs After IPS")
    user_id2 = st.slider("Select User ID", 1, 943, 50)
    if st.button("Compare"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before (Biased)")
            for i, m in enumerate(get_recs(user_id2, sim_df, matrix), 1):
                st.write(f"{i}. {m}")
        with col2:
            st.subheader("After (Debiased)")
            for i, m in enumerate(get_recs(user_id2, sim_ips_df, weighted_matrix), 1):
                st.write(f"{i}. {m}")

with tab3:
    st.header("RL Epsilon-Greedy Bandit Learning")
    steps = st.slider("Training steps", 1000, 10000, 5000, step=1000)
    if st.button("Train Bandit"):
        movie_ids = ratings['movie_id'].unique()
        true_rewards = {}
        for mid in movie_ids:
            mr = ratings[ratings['movie_id']==mid]['rating']
            true_rewards[mid] = (mr >= 4).mean()
        
        q_values = {m: 0.0 for m in movie_ids}
        counts = {m: 0 for m in movie_ids}
        history = []
        total = 0
        
        progress = st.progress(0)
        for i in range(steps):
            if np.random.random() < 0.1:
                mid = np.random.choice(movie_ids)
            else:
                mid = max(q_values, key=q_values.get)
            reward = np.random.binomial(1, true_rewards[mid])
            counts[mid] += 1
            q_values[mid] += (reward - q_values[mid]) / counts[mid]
            total += reward
            history.append(total/(i+1))
            if i % 500 == 0:
                progress.progress(i/steps)
        
        progress.progress(1.0)
        st.line_chart(history)
        st.success(f"Final reward: {round(history[-1], 3)} | Improvement: {round((history[-1]-history[10])/history[10]*100,1)}%")
        
        top_movies = sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:10]
        top_df = pd.DataFrame(top_movies, columns=['movie_id','reward'])
        top_df = top_df.merge(movies, on='movie_id')
        st.subheader("Top 10 movies bandit learned:")
        st.dataframe(top_df[['title','reward']].reset_index(drop=True))