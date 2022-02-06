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

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Changing the background
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
# Applying the function
set_png_as_page_bg('resources/imgs/pic.jpg')


# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Exploratory Data Analysis","Solution Outline","Us"]

    st.sidebar.image("resources/imgs/netflixai.png", use_column_width=True)
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

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Outline":
        st.write('# Solution Outline')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        #st.title("Solution Overview")
        
        st.write("In today’s technology driven world, recommender systems are socially and economically critical to ensure that individuals can make optimised choices surrounding the content they engage with on a daily basis. One application where this is especially true is movie recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed, based on their historical preferences.Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being personalised recommendations - generating platform affinity for the streaming services which best facilitates their audience's viewing. We followed a data science workflow as shown below.")
        st.image('resources/imgs/Machine_learning.png',use_column_width=True)
        st.write("Source - The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB")
        st.write("We used the MovieLens dataset. Below is a brief description of the dataset")
        st.write("### Data given : ")
        st.write("###### train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.")
        st.write("###### imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.")
        st.write("###### links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.")
        st.write("###### genome_scores.csv - a score mapping the strength between movies and tag-related properties.")
        st.write("###### sample_submission.csv - Sample of the submission format.")
        st.write("###### tags.csv - User assigned for the movies within the dataset.")
        st.write("###### test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.")
        
        st.write("We used the machine learning process which is data acquisition , data transforming, geeting insights from the data, and  Model building then we deployed the model through streamlit app and the data we obtained it from Kaggle as given by Explore Data Science Academy. ")
        st.write("After data preprocessing we started building our models. We built different collaborative base models one of them was Singular Value Decomposition (SVD).")

        imdb = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h3 style="color:black;text-align:left;">Cleaning the imdb_data data</h3>
        """
        st.markdown(imdb, unsafe_allow_html=True)
        st.write('The runtime was imputed with the average runtime.\n\nCreated a list plot keywords for each movie.\n\nCreated a list of title casts for each movie. \n\n We dropped all the title_cast rows which were missing. \n\n We dropped all the plot_keywords rows which were missing.".\n\n We dropped all the director rows which were missing".')

        st.write('### Cleaning the movies dataset')
        st.write('Created a list of genres in every movie in the movies column\n\nThe release_year column was added which was extracted from the title of the movies.')

        st.write('### Cleaning the train dataset')
        st.write('The timestamp column was dropped.')




        st.write('After cleaning the data, we then merged them ')
        st.write('We proceeded to the third step, the Exploratory Data Analysis. We plotted many visualisations using our data and these are on our Exploratory Data Analysis page.')

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    
        # information about the app
    if page_selection == "Us":
        title_about = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;">  Background </h1>
        <h3 style="color:black;text-align:center;">This is a group of students from Explore Data Science academy we are tasked to develop an algorithm that will recommend movies that someone might enjoy watching and we are students from different background.</h3>
        """
        mission = """
	    <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;">  Aim  </h1>
        <h3 style="color:black;text-align:center;">Is to keep users of our app happy by recommending a movie they are most likely to love after watching it and saving them from the headache of having to go through our large database system filled with our movies.
        </h3>
        """

        contributors = """
        <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;">  The Team </h1>
        <h3 style="color:black;text-align:center;">Jamie Snyders (Supervisor)</h5>
        <h3 style="color:black;text-align:center;">Sello Mafikeng (R & D Coordinator)</h5>
        <h3 style="color:black;text-align:center;">Vuyisile Danisile (Developer analyst)</h5>
        <h3 style="color:black;text-align:center;">Peter Mulaudzi (Research analyst)</h5>
        <h3 style="color:black;text-align:center;">Malibongwe Shange (Developer analyst)</h5>
        <h3 style="color:black;text-align:center;">Reloef Khoza (R & D Coordinator) </h5>
        
        """

        contacts = """
        <div style="background-color:#464e5f00;padding:10px;border-radius:10px;margin:10px;">
	    <h1 style="color:black;text-align:center;">  Contact Us </h1>
        <h3 style="color:black;text-align:center;">Jamie Snyders (jamie@explore-datascience.net)</h5>
        <h3 style="color:black;text-align:center;">Sello Mafikeng (tebogomafikeng@gmail.com)</h5>
        <h3 style="color:black;text-align:center;">Vuyisile Danisile (vuyiedannie@gmail.com)</h5>
        <h3 style="color:black;text-align:center;">Peter Mulaudzi (ndivhupmulaudzi@gmail.com)</h5>
        <h3 style="color:black;text-align:center;">Malibongwe Shange (shangemalibongwe@gmail.com)</h5>
        <h3 style="color:black;text-align:center;">Reloef Khoza (reloefmika@gmail.com) </h5>
        
        """
        
        st.markdown(title_about, unsafe_allow_html=True)
        st.markdown(mission, unsafe_allow_html=True)
        st.markdown(contributors, unsafe_allow_html=True)
        st.markdown(contacts, unsafe_allow_html=True)
    # EDA
    if page_selection == "Exploratory Data Analysis":

        st.write('# Exploratory Data Analysis')
        st.image('resources/imgs/netflix2.png',use_column_width=True)
        st.image('resources/imgs/netflix2.jpeg',use_column_width=True)
        
    

        # available options = ["Ratings", "Genre", "Director", "Movies", "Actors"]

        sys_eda = st.radio("Choose an EDA option",
        ('Actors','Directors','Genres','Movies','Ratings'))
        # Ratings option
        if sys_eda == "Ratings":



            
    
            st.image("resources/imgs/the_average_rating_per_genre_using_a_box-plot.png")
               

        # Genres option
        if sys_eda == "Genres":
        
            op_genre = st.radio("Choose an option under Genres",("Distribution of movie genres","Distribution of Runtime"))
            

            if op_genre == "Distribution of movie genres":
            
                st.image('resources/imgs/movie_genres.png',use_column_width=True)
               


            if op_genre == "Distribution of Runtime":
                st.image("resources/imgs/Average_runtime.png")
                


        # Directors option
        if sys_eda == "Directors":

            

            st.image("resources/imgs/Top_10_Most_Popular_Movie_Directors.png")
            

        # Movies option
        if sys_eda == "Movies":
            op_movie = st.radio("Choose an option under movies",("Total movies produced per year","Wordcloud of the titles of the movies"))


            if op_movie == "Total movies produced per year":
                st.image("resources/imgs/Total_movies_released_per_year.png")
            
                

            if op_movie == "Wordcloud of the titles of the movies":
                st.image("resources/imgs/title_wordcloud.png")
               

        if sys_eda == "Actors":

            op_actors = st.radio("Choose an option under actors",("Names of Popular actors in the movies","Top 20 popular actors"))

            if op_actors == "Names of Popular actors in the movies":
                st.image("resources/imgs/titlecast_wordcloud.png")
                

            if op_actors == "Top 20 popular actors":
                st.image("resources/imgs/Top_20_Popular_Actors.png")
               




if __name__ == '__main__':
    main()
