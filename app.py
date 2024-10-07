import gradio as gr
from transformers import pipeline
import matplotlib.pyplot as plt
import requests
from huggingface_hub import InferenceClient
import os

# Define models for local and remote inference
local_model = "distilbert-base-uncased-finetuned-sst-2-english"
remote_model = "siebert/sentiment-roberta-large-english"

# Load the local sentiment analysis pipeline with the specified model
local_pipeline = pipeline("sentiment-analysis", model=local_model)

# Initialize the inference client
remote_inference_client = InferenceClient(remote_model)

# OMDb API key (replace with your own API key)
OMDB_API_URL = 'http://www.omdbapi.com/'
OMDB_API_KEY = os.getenv("OMDB")  # Fetching API key from environment variables

# Function to fetch movie information from OMDb API
def fetch_movie_info(movie_name):
    try:
        response = requests.get(OMDB_API_URL, params={'t': movie_name, 'apikey': OMDB_API_KEY})
        data = response.json()
        if data['Response'] == 'True':
            return {
                'Title': data.get('Title', 'N/A'),
                'Description': data.get('Plot', 'N/A'),
                'Year': data.get('Year', 'N/A'),
                'Director': data.get('Director', 'N/A'),
                'Genre': data.get('Genre', 'N/A'),
                'Actors': data.get('Actors', 'N/A'),
                'Rating': data.get('imdbRating', 'N/A'),
            }
        else:
            return {'Error': data.get('Error', 'Movie not found')}
    except Exception as e:
        return {'Error': str(e)}

# Function to perform sentiment analysis using the local pipeline
def local_sentiment_analysis(review):
    try:
        result = local_pipeline(review)
        sentiment = result[0]['label']
        score = result[0]['score']
        return sentiment, score
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Function to perform sentiment analysis using the remote pipeline
def remote_sentiment_analysis(review):
    try:
        # Make a request to the Hugging Face Inference API for text classification
        response = remote_inference_client.text_classification(review)
        sentiment = response[0]['label']
        score = response[0]['score']
        return sentiment, score
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Function to analyze sentiment and fetch movie details
def analyze_sentiment(movie_name, review, mode):
    if not review.strip():
        return "Error: Review text cannot be empty.", None, None, None

    if mode == "Local Pipeline":
        sentiment, score = local_sentiment_analysis(review)
        model_info = f"Using local model: {local_model}"
    elif mode == "Inference API":
        sentiment, score = remote_sentiment_analysis(review)
        model_info = f"Using remote model: {remote_model}"
    else:
        return "Invalid mode selected.", None, None, None

    # Fetch movie information
    movie_info = fetch_movie_info(movie_name)

    # Format the sentiment result
    result_text = f"Sentiment: {sentiment}, Confidence: {score:.2f}\n{model_info}"

    # Extract movie description
    movie_description = format_movie_description(movie_info)
    
    # Enhanced plot
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['POSITIVE', 'NEGATIVE']
    sentiment_scores = [score if sentiment == 'POSITIVE' else (1 - score), score if sentiment == 'NEGATIVE' else (1 - score)]
    colors = ['#4CAF50', '#F44336']

    bars = ax.bar(categories, sentiment_scores, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Confidence Score')
    ax.set_title('Sentiment Analysis Result')

    # Add text labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Return the Matplotlib figure directly to plot_output
    return result_text, movie_description, fig

# Function to format movie information into a plain text format
def format_movie_description(movie_info):
    if 'Error' in movie_info:
        return f"Error: {movie_info['Error']}"
    
    return (
        f"Title: {movie_info['Title']}\n"
        f"Year: {movie_info['Year']}\n"
        f"Actors: {movie_info['Actors']}\n"
        f"Director: {movie_info['Director']}\n"
        f"Rating: {movie_info['Rating']}\n"
        f"Genre: {movie_info['Genre']}\n"
        f"Description: {movie_info['Description']}"
    )

# Enhanced CSS for a modern, clean look
custom_css = """
body {
    background-color: #1e1e2f;
    color: #ffffff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

.gradio-container {
    border-radius: 10px;
    background-color: #2c2f48;
    padding: 20px;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
}

.gr-textbox, .gr-radio {
    margin-bottom: 20px;
    padding: 12px;
    border-radius: 8px;
    border: none;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    background-color: #3b3e56;
    color: #ffffff;
}

.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 12px 24px;
    font-size: 16px;
    cursor: pointer;
    transition: 0.3s;
    border-radius: 8px;
    margin-top: 10px;
    box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}

.gr-button:hover {
    background-color: #388e3c;
}

h1 {
    text-align: center;
    font-size: 34px;
    margin-bottom: 20px;
    color: #00bcd4;
}
"""

# Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>🎬 Movie Review Sentiment Analysis</h1>")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            movie_input = gr.Textbox(
                label="🎥 Movie Name", placeholder="Enter the movie name...", lines=1
            )
            review_input = gr.Textbox(
                label="📝 Movie Review", placeholder="Enter your movie review...", lines=4
            )
            mode_input = gr.Radio(
                ["Local Pipeline", "Inference API"], label="🔍 Processing Mode", value="Inference API"
            )
            analyze_button = gr.Button("🔍 Analyze Sentiment")

        with gr.Column(scale=2):
            sentiment_output = gr.Textbox(label="🗨️ Sentiment Analysis Result", interactive=False)
            movie_description_output = gr.Textbox(label="📃 Movie Description", interactive=False, lines=10)
            plot_output = gr.Plot(label="📊 Sentiment Score Graph")

    analyze_button.click(
        analyze_sentiment, 
        [movie_input, review_input, mode_input], 
        [sentiment_output, movie_description_output, plot_output]
    )

# Run the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
