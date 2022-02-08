"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

import requests
from PIL import Image

#Visualizations
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
import seaborn as sns
from wordcloud import WordCloud
_lock = RendererAgg.lock

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model


# Data Loading
title_list = load_movie_titles('./resources/data/movies.csv')

# Creating dataframes
movies = pd.read_csv('./resources/data/movies.csv')
imdb_data = pd.read_csv('./resources/data/imdb_data.csv')
tags = pd.read_csv('./resources/data/tags.csv')


# App declaration
def main():

    # Loading Company logo
    st.sidebar.image('resources/imgs/flck.png', width=300)
    st.sidebar.title('Flick Insights')

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Insights", "Company Information", "Product Information"]


    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------------


    if page_selection == 'Product Information':
        st.write('# Flick Insights')
        st.image('resources/imgs/dots.png', width=50)
        st.write('### Product Description')
        st.info('Flick Insights is a software as a service(saas) platform used by Cinemas and movie streaming platforms as a recommendation engine, to recommend product to their customers.')
        st.info('Just like Google has build the worlds best search engine, at Flick Insights we are bulding the cutting edge recommendation engine tailored at providing users with the best recommended content. The same way companies like Stripe, PayPal and Square are facilitating payments we are faciitating and influencing the way people interacts with products.')

        st.write('### Benefits of using Flick Insights')
        st.image('resources/imgs/recommendation-system.png',
                use_column_width=True)

        """* During movie night-out/at the cinema –A group of friends or family can enter a list of their most liked movies into the Self-Service portal at the cinema"""
        """* The platform will then use Flick Insights API call recommend movies that the user/users will probably like based on the metrics"""
        """ * Increased Sales and convertion due to increased customer retention"""
        """ * Increased customer satisfaction and new customer attraction due to peer to peer marketing"""
        """ * Reduced churn, there will be a decrease in number of customers who stop using the product"""
        """ * Increase customer traffic and loyalty"""


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    if page_selection == "Insights":

        st.title("Movies Insights")
        st.image('resources/imgs/dots.png', width=50)
        st.info("Flick Insights allow users to visualize customer related analystics, as to understand thier online behaviour")

        plotviz = ["Please select", "-- Popular Movie Genres",
                "-- Top 20 Movies of All Times",
                "-- Top 20 Actors in most Movies",
                "-- Top 20 Directors With Most Movies", 
                "-- Top 20 Popular Play Plots", "-- Popular Movie Tags"]


        plot_selection = st.sidebar.selectbox("Select visualisation", plotviz)
        
        if plot_selection == "-- Popular Movie Genres":

            ################# Plot 1 ############
            # Create dataframe containing only the movieId and genres

            movies_genres = pd.DataFrame(movies[['movieId', 'genres']],
                    columns=['movieId', 'genres'])
            # Split genres seperated by "|" and create a list containing the genres allocated to each movie
            movies_genres.genres = movies_genres.genres.apply(lambda x: x.split('|'))
            # Create expanded dataframe where each movie-genre combination is in a seperate row
            movies_genres = pd.DataFrame([(tup.movieId, d) for tup in movies_genres.itertuples() for d in tup.genres], columns=['movieId', 'genres'])

            fig1 = Figure()
            ax = fig1.subplots()
            sns.countplot(y="genres", data=movies_genres, order=movies_genres['genres'].value_counts(ascending=False).index, color='pink', alpha=0.9, ax=ax)
            ax.set_ylabel('Genre')
            ax.set_title('Popular Movie Genres')
            st.pyplot(fig1)
            st.write("")
            st.info('Drama and Comedy are the most popular Genres among viewers')
                    ################# Plot 2 ############

        #if plot_selection == "":

            #Merge movies and train tables
           # data = pd.merge(train , movies, on = 'movieId')

            # Convert timestamp to year column representing the year the rating was made on merged dataframe
            #data['rating_year'] = data['timestamp'].apply(lambda timestamp: datetime.fromtimestamp(timestamp).year)
            #data.drop('timestamp', axis=1, inplace=True)

            # Calculating avarage rating and storing the results as a DataFrame
           # rating = pd.DataFrame(data.groupby('movieId')['rating'].mean())

            # Calculating total ratings count and storing the results as a DataFrame
            #rating['ratings_count'] = pd.DataFrame(data.groupby('movieId')['rating'].count())
            #rating=rating.sort_values(by=['ratings_count','ratings_count'],ascending=False).reset_index()

            # Joining both DataFrames
            #inner_join = pd.merge(rating,movies,on ='movieId',how ='inner')
            #popular_movies = inner_join[['title','ratings_count']]


            # Plot the figure.
            #fig2 = Figure(figsize=(17, 12), dpi=85)
            #ax = fig2.subplots()
            #ax = popular_movies.plot(kind='barh', color='pink', 
             #       fontsize=17, xlim=[45, 84], width=.75, alpha=0.8, ax=ax)

            #ax.set_ylabel('title', fontsize=30)
            #ax.set_xlabel('count', fontsize=30)
            #ax.set_title('Top 20 Movies of All Times', fontsize=30)
            #ax.set_yticklabels(y_labels)
            #st.pyplot(fig2)
            #st.write("")
           # st.info('ShawShank and Forest Gump have high ratings count')


                    ################# Plot 3 ############
        if plot_selection == "-- Top 20 Actors in most Movies":
            movies_actor = pd.DataFrame(imdb_data[['movieId', 'title_cast']],
                    columns=['movieId', 'title_cast'])

            # Split title_cast seperated by "|" and create a list containing the title_cast allocated to each movie
            movies_actor = movies_actor[movies_actor['title_cast'].notnull()]
            movies_actor.title_cast = movies_actor.title_cast.apply(lambda x: x.split('|'))

            # Create expanded dataframe where each movie-tite_cast combination is in a seperate row
            movies_actor = pd.DataFrame([(tup.movieId, d) for tup in movies_actor.itertuples() for d in tup.title_cast],
                    columns=['movieId', 'title_cast'])
            movies_actor = movies_actor.groupby(
                ['title_cast'])['movieId'].count().reset_index(name='Number of Movies')
            movies_actor = movies_actor.sort_values(by='Number of Movies', ascending=False)

            # Sececting the Top 20 actors in movies
            movies_actor = movies_actor .head(20)
            movies_actor = movies_actor.sort_values(by='Number of Movies', ascending=True)
            y_labels = movies_actor['title_cast']

            # Plot the figure.
            y_labels = movies_actor['title_cast']
            fig2 = Figure(figsize=(17, 12), dpi=85)
            ax = fig2.subplots()
            ax = movies_actor['Number of Movies'].plot(
                kind='barh', color='pink', fontsize=17, xlim=[45, 84], width=.75, alpha=0.8, ax=ax)
            #sns.countplot(y='title_cast', data=movies_actor,order=movies_actor['title_cast'].value_counts(ascending=False).index,palette='deep',ax=ax)
            ax.set_ylabel('Name of Actor', fontsize=30)
            ax.set_xlabel('Number of movies featuring the actor', fontsize=30)
            ax.set_title('Top 20 Actors in IMDB Dataset ', fontsize=30)
            ax.set_yticklabels(y_labels)
            st.pyplot(fig2)
            st.write("")
            st.info('Popular Actors tend to be featured in most movies')


         ################## Plot 4 ################################
        if plot_selection == "-- Top 20 Directors With Most Movies":

            # grouping the movies by the director and counting the total number of movies per director
            movies_director = pd.DataFrame(
                imdb_data[['movieId', 'director']], columns=['movieId', 'director'])
            movies_director = movies_director.groupby(
                ['director'])['movieId'].count().reset_index(name="count")
            movies_director = movies_director.sort_values(
                by='count', ascending=False)
            movies_director = movies_director .head(20)
            movies_director = movies_director.sort_values(
                by='count', ascending=True)

            y_labels = movies_director['director']
            # Plot the figure.
            fig3 = Figure(figsize=(18, 12), dpi=85)
            ax = fig3.subplots()
            ax = movies_director['count'].plot(
                kind='barh', color='blue',  width=.7, fontsize=16, xlim=[8, 30], alpha=0.9, ax=ax)
            ax.set_title(
                'Top 20 directors with the  most Movies from imdb database', fontsize=30)
            ax.set_xlabel('Number of Movies Directed', fontsize=30)
            ax.set_ylabel('Name of director', fontsize=30)
            ax.set_yticklabels(y_labels)
            st.pyplot(fig3)
            st.write("")
            st.info('Directors that are mostly liked by people tend to release the most number movies')
            st.write("")
            st.info('Actors and directors are featured in most movies are likely to receive more resources for them to improve on  their craft, and the more people will like their movies')

        ################## Plot 5 ################################
        if plot_selection == "-- Top 20 Popular Play Plots":

            movies_plot = pd.DataFrame(imdb_data[['movieId', 'plot_keywords']], columns=[
                                       'movieId', 'plot_keywords'])
            # Split play plot seperated by "|" and create a list containing the play plot allocated to each movie
            movies_plot = movies_plot[movies_plot['plot_keywords'].notnull()]
            movies_plot.plot_keywords = movies_plot.plot_keywords.apply(
                lambda x: x.split('|'))
            # Create expanded dataframe where each movie-play_plot combination is in a seperate row
            movies_plot = pd.DataFrame([(tup.movieId, d) for tup in movies_plot.itertuples(
            ) for d in tup.plot_keywords], columns=['movieId', 'plot_keywords'])
            movies_plot = movies_plot.groupby(['plot_keywords'])[
                'movieId'].count().reset_index(name="count")
            movies_plot = movies_plot.sort_values(by='count', ascending=False)
            movies_plot = movies_plot.head(20)
            movies_plot = movies_plot.sort_values(by='count', ascending=True)

            y_labels = movies_plot['plot_keywords']
            # Plot the figure.
            fig4 = Figure(figsize=(18, 12), dpi=85)
            ax = fig4.subplots()
            ax = movies_plot['count'].plot(kind='barh', color='lightblue', fontsize=17, width=.7, alpha=0.7, ax=ax)
            ax.set_title('Top 20 Popular Play Plots ', fontsize=30)
            ax.set_xlabel('Total Number of Play Plots', fontsize=30)
            ax.set_ylabel('Movie plot', fontsize=30)
            ax.set_yticklabels(y_labels)
            st.pyplot(fig4)
            st.write("")
            st.info('The  graph show\'s that people tend to enjoy a movie with certain movie plots, such as love and nudity.')


         ###################### Plot 6 ##############################
        if plot_selection == "-- Popular Movie Tags":

            tags_2 = str(list(tags['tag']))

            wc = WordCloud(background_color="white", max_words=100,
                           width=1600, height=800, collocations=False).generate(tags_2)
            plt.imshow(wc)
            plt.axis("off")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.rcParams["axes.grid"] = False
            st.pyplot()

    if page_selection == "Company Information":
        st.title("Company Information")
        st.header('Flick Insights')

        """ * Flick Insights is a software as a service(saas) platform used by Cinemas and movie streaming platform as a recommendation engine, to recommend movies to their customers"""
    
        """* We charge Cinemas 7.5% per customer based on the price. Movie Streaming platforms are charged R2.50 pay per click model, Flick Insights also provides customer insights through understanding user behaviorial patterns and provides consultancy advice to their clientele"""
         
        link='Cinema Prices [link](https://businesstech.co.za/news/media/292592/movie-ticket-prices-in-south-africa-ster-kinekor-vs-nu-metro/")'
        st.markdown(link,unsafe_allow_html=True)
        st.write("")
        st.write('### Problem Statement')

        """* The digital age has brought streaming technology to the shores,with unlimited supply of movies and rising demand for new and exciting content,it can be overwhelming for a user to choose a movie from an unlimited list."""
        """* This can lead to users randomly selecting movies that they might not like or end up not watching it. which reduces customer satisfaction and increases customer churn"""
        """* This can be costly and time consuming for customers to make a choice."""

if __name__ == '__main__':
    main()
