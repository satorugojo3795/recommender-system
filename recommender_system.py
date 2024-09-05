import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# function to load the database 
def load_db():
    # df = pd.read_pickle('dataframe.pkl')
    # df = df[:4001]
    # return df
    df1=pd.read_csv('tmdb_5000_credits.csv')
    df2=pd.read_csv('tmdb_5000_movies.csv')
    df1.columns = ['id','tittle','cast','crew']
    df2= df2.merge(df1,on='id')
    return df2

def get_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Replace empty strings with NaN
    df.loc[:, 'overview'] = df['overview'].fillna('')
    
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    
    return tfidf_matrix

def calculate_cosine_similarity(df):
    
    # Calculate cosine similarity
    tfidf_matrix = get_tfidf_matrix(df)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    #Construct a reverse map of indices and movie titles
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return cosine_similarities, indices

def get_recommendations(df, title):
    
    cosine_similarities,indices = calculate_cosine_similarity(df)
    #get the index corresponding to the titke
    idx = indices[title]

    #get the list of cosine score corresponding to that index
    similarity_score = list(enumerate(cosine_similarities[idx]))
    # similarity_score[:10]
    
    similarity_scores_descending = sorted(similarity_score, key=lambda x: x[1],reverse=True)
    # similarity_scores_descending[:11]
    
    movie_index_all = similarity_scores_descending[1:11]
    movie_index = [i[0] for i in movie_index_all]
    # movie_index
    
    return df['title'].iloc[movie_index]

# df = load_db()
# df = df[:4001]
# df = df[:10001]

if __name__ == '__main__':
    df = load_db()
    title = input("Enter the movie title: ").strip()
    print(get_recommendations(df, title))
    # title = sys.argv[1].strip()

    # print(get_recommendations(title,df))



