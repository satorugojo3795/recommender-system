# from flask import Flask, request, render_template
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)

# def load_db():
#     df1 = pd.read_csv('tmdb_5000_credits.csv')
#     df2 = pd.read_csv('tmdb_5000_movies.csv')
#     df1.columns = ['id', 'tittle', 'cast', 'crew']  # Ensure correct column naming
#     df2 = df2.merge(df1, on='id')
#     return df2

# def get_tfidf_matrix(df):
#     tfidf = TfidfVectorizer(stop_words='english')
#     df['overview'] = df['overview'].fillna('')
#     tfidf_matrix = tfidf.fit_transform(df['overview'])
#     return tfidf_matrix

# def calculate_cosine_similarity(df):
#     tfidf_matrix = get_tfidf_matrix(df)
#     cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     indices = pd.Series(df.index, index=df['title']).drop_duplicates()
#     return cosine_similarities, indices

# def get_recommendations(df, title):
#     cosine_similarities, indices = calculate_cosine_similarity(df)
    
#     if title not in indices:
#         return []  # Return empty if title not found
    
#     idx = indices[title]
#     sim_scores = list(enumerate(cosine_similarities[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     top_movies = sim_scores[1:11]  # Get top 10 recommendations
#     movie_indices = [i[0] for i in top_movies]
#     return df['title'].iloc[movie_indices].tolist()

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     recommendations = []
#     title = ''  # Default empty title
#     if request.method == 'POST':
#         title = request.form['title'].strip()
#         df = load_db()
#         recommendations = get_recommendations(df, title)
#     return render_template('index.html', recommendations=recommendations, title=title)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template
import pandas as pd
import http.client
import json
from urllib.parse import quote  # Import for URL encoding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def load_db():
    df1 = pd.read_csv('tmdb_5000_credits.csv')
    df2 = pd.read_csv('tmdb_5000_movies.csv')
    df1.columns = ['id', 'tittle', 'cast', 'crew']
    df2 = df2.merge(df1, on='id')
    return df2

def get_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    df['overview'] = df['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    return tfidf_matrix

def calculate_cosine_similarity(df):
    tfidf_matrix = get_tfidf_matrix(df)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_similarities, indices

def get_movie_poster(movie_title):
    conn = http.client.HTTPSConnection("api.collectapi.com")
    headers = {
        'content-type': "application/json",
        'authorization': "apikey 3MHXBGTE7PbaQqWSgF1E77:6jBh5vVNNzsH9Ftb01dDKZ"
    }
    
    # URL-encode the movie title
    encoded_title = quote(movie_title)
    conn.request("GET", f"/imdb/imdbSearchByName?query={encoded_title}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    data = json.loads(data.decode("utf-8"))
    
    if data['success'] and data['result']:
        return data['result'][0]['Poster']
    return None

def get_recommendations(df, title):
    cosine_similarities, indices = calculate_cosine_similarity(df)
    
    if title not in indices:
        return []  # Return empty if title not found
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = sim_scores[1:11]  # Get top 10 recommendations
    movie_indices = [i[0] for i in top_movies]
    
    recommendations = []
    for index in movie_indices:
        movie_title = df['title'].iloc[index]
        poster_url = get_movie_poster(movie_title)
        recommendations.append({'title': movie_title, 'poster': poster_url})
    
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    title = ''  # Default empty title
    if request.method == 'POST':
        title = request.form['title'].strip()
        df = load_db()
        recommendations = get_recommendations(df, title)
    return render_template('index.html', recommendations=recommendations, title=title)

if __name__ == '__main__':
    app.run(debug=True)
