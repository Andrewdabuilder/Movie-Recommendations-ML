# Lesson from here - https://blog.jovian.ai/creating-a-movie-recommendation-system-using-python-5ba88a7eb6df
# Data - https://www.kaggle.com/grouplens/movielens-20m-dataset

# Virtual Env Guide 
import pandas as pd
import numpy as np

#importing visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt


# usecols allows us to select our choice of features
movies_df=pd.read_csv('MovieData/movie.csv', usecols=['movieId','title'], dtype={'movieId':'int32','title':'str'})
#print(movies_df.head())

ratings_df=pd.read_csv('MovieData/rating.csv',
    usecols=['userId', 'movieId', 'rating','timestamp'],dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
#print(ratings_df.head())

#check for any null values present andnumber of entries in both data sets
#movies_df.isnull().sum()
#ratings_df.isnull().sum()
#print("Movies:", movies_df.shape)

#Merge the dataframes on the column 'movieId'
movies_merged_df=movies_df.merge(ratings_df, on='movieId')
#print(movies_merged_df.head())

#We have now merged the imported datasets sucessfully

#Adding features to analyse data
#Adding 'Average Rating' & 'Rating Count' columns 

movies_average_rating = movies_merged_df.groupby(
    'title')['rating'].mean().sort_values(ascending=False).reset_index(
    ).rename(columns={'rating':'Average Rating'})
#print(movies_average_rating.head())

#Adding Rating Count to assess numbers of ratings

movies_rating_count = movies_merged_df.groupby(
    'title')['rating'].count().sort_values(ascending=False).reset_index(
    ).rename(columns={'rating':'Rating Count'})
#print(movies_rating_count.head())

#Combing ratings of average with counts
movies_rating_count_average = movies_average_rating.merge(movies_rating_count, on='title')
#print(movies_rating_count_average.head())

#We now have a new list with Title Average and Rating Count
#We will need to rule out movies with too low Rating Counts
# We will visualise the data using seaborn and matplotlib

# Let us plot histograms for ournewly creates features to findout where we will cut the amount 
# of reviews that is a minimum threshold
sns.set(font_scale = 1)
plt.rcParams["axes.grid"] = False
plt.style.use('dark_background')
#matplotlib inline
plt.figure(figsize=(12,4))
plt.hist(movies_rating_count_average['Rating Count'],bins=80,color='tab:purple')
plt.ylabel('Ratings Count(Scaled)', fontsize=16)
#plt.savefig('ratingcounthist.jpg')

plt.figure(figsize=(12,4))
plt.hist(movies_rating_count_average['Average Rating'],bins=80,color='tab:purple')
plt.ylabel('Average Rating',fontsize=16)
#plt.savefig('avgratinghist.jpg')

#We now join the 2 plots together

plot=sns.jointplot(x='Average Rating',y='Rating Count',data=movies_rating_count_average,kind="reg")
#plot.savefig('joinplot.jpg')

#Analysis of plots 
#Plot rating count hist shows that there is a lot of movies with not a lot of ratings
# Plot 2 shows that there is more movies between 2- 4 with more reviews
# Join plot shows there is only a subset of values with a higher ratingand considerable amount of ratings

#Eliminating outliers
rating_with_RatingCount = movies_merged_df.merge(movies_rating_count, left_on='title', right_on='title', how = 'left')
#print(rating_with_RatingCount.head())
#Eliminating data by stting a threshold
#We can get the quantiles and standard dev by using describe


#pd.set_option('display.float_format', lambda x: '%.3f' % x)
#print(rating_with_RatingCount['Rating Count'].describe())

#From the above we find that 50% of movies only have 18 reviews, where the max have 67K reviews
#75% of movies have more than 200 reviews

#Now we will set a threshold value andcreate a dataframe of entries above the threshold
# We are going to include the top 25% of movies
popularity_threshold = 1967
popular_movies = rating_with_RatingCount[rating_with_RatingCount['Rating Count']>=popularity_threshold]
#print(popular_movies.shape)

# This has cleaned our data apparently

#Now we create a Pivot table with users as indices and movies as columns 
import os
movie_features_df = popular_movies.pivot_table(index = 'title', columns='userId',values='rating').fillna(0)
#print(movie_features_df.head())
# Movie feature df is a pivot table that we can pass to our model 

