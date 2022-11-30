# Virtual Env Guide
import pandas as pd
import numpy as np



# usecols allows us to select our choice of features
movies_df=pd.read_csv('MovieData/movie.csv', usecols=['movieId','title'], dtype={'movieId':'int32','title':'str'})
#print(movies_df.head())

ratings_df=pd.read_csv('MovieData/rating.csv',
    usecols=['userId', 'movieId', 'rating','timestamp'],dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
print(ratings_df.head())