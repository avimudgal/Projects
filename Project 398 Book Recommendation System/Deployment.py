import pickle
import pandas as pd
import streamlit as st

# Load your content based model (adjust the path to your actual model file)
with open('content_recom_model.pkl', 'rb') as file:
    df, cosine_sim, tfidf_matrix = pickle.load(file)

# Load collaborative based model
with open('collab_recommendation_model.pkl', 'rb') as file:
    model_knn, combined_sparse_matrix, book_user_matrix, book_user_sparse_matrix, df = pickle.load(file)

# Load popularity based model
with open('popularity_model.pkl', 'rb') as file:
    display_df, display_df1 = pickle.load(file)


# Define the content recommendation function
def get_recoms_content_based(title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all books
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return df[['title', 'author', 'link']].iloc[book_indices]


# Define the collaborative based recommendation
def get_recoms_collab_based(isbn, model_knn=model_knn, data=combined_sparse_matrix, n_recommendations=10):
    # Get the index of the book that matches the ISBN

    book_idx = book_user_matrix.index.get_loc(isbn)

    distances, indices = model_knn.kneighbors(data[book_idx], n_neighbors=n_recommendations + 1)
    book_indices = indices.squeeze().tolist()
    book_indices = book_indices[1:]
    recommended_isbns = book_user_matrix.iloc[book_indices].index.tolist()

    # Get the details of the recommended books
    recommended_books = df[df['ISBN'].isin(recommended_isbns)][['title', 'link']]

    return recommended_books


# Streamlit app
st.title("Book Recommendation System")

# Select recommendation method
method = st.radio("Select Recommendation Method:", ("Content-Based", "Collaborative-Based"))

if method == "Content-Based":
    # Dropdown for book titles
    book_title = st.selectbox("Select a book title", df['title'])
    if st.button("Get Recommendations"):
        recommendations = get_recoms_content_based(book_title)
        st.write("Recommendations:")
        for idx, row in recommendations.iterrows():
            st.write(f"{row['title']} by {row['author']}: [Link]({row['link']})")

elif method == "Collaborative-Based":
    # Dropdown the ISBN
    selected_isbn = st.selectbox("Select the ISBN", df['ISBN'])
    if st.button("Get Recommendations"):
        recommendations = get_recoms_collab_based(selected_isbn)
        st.write("Recommendations:")
        for idx, row in recommendations.iterrows():
            st.write(f"{row['title']}:[Link]({row['link']})")

# Sidebar with top 50 books tables
col1, col2 = st.columns(2)
with col1:
    st.header("Modern Era Top 50 Books (years 2000 and above)")
    st.table(display_df.style.set_properties(**{'max-height': '50px', 'overflow-y': 'auto'}))

with col2:
    st.header("Pre-Modern Era Top 50 Books (years below 2000)")
    st.table(display_df1.style.set_properties(**{'max-height': '50px', 'overflow-y': 'auto'}))