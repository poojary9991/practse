import gradio as gr
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import InferenceClient

# Define models for local and remote inference
local_model = "distilbert-base-uncased-finetuned-sst-2-english"
remote_model = "distilbert-base-uncased-finetuned-sst-2-english"  # You can use the same model for both for now

# Load the local sentiment analysis pipeline with the specified model
local_pipeline = pipeline("sentiment-analysis", model=local_model)

# Initialize the inference pipeline for remote model (same model for simplicity here)
remote_pipeline = InferenceClient(model=remote_model)

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
        response = inference_client.text_classification(inputs=review)
        sentiment = response[0]['label']
        score = response[0]['score']
        return f"Sentiment: {sentiment}, Confidence: {score:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to analyze sentiment and return the result as a string and plot
def analyze_sentiment(review, mode):
    if not review.strip():
        return "Error: Review text cannot be empty.", None

    if mode == "Local Pipeline":
        sentiment, score = local_sentiment_analysis(review)
        model_info = f"Using local model: {local_model}"
    elif mode == "Inference API":
        sentiment, score = remote_sentiment_analysis(review)
        model_info = f"Using remote model: {remote_model}"
    else:
        return "Invalid mode selected.", None

    # Format the sentiment result
    result_text = f"Sentiment: {sentiment}, Confidence: {score:.2f}\n{model_info}"

    # Enhanced plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['POSITIVE', 'NEGATIVE']
    sentiment_scores = [score if sentiment == 'POSITIVE' else (1 - score), score if sentiment == 'NEGATIVE' else (1 - score)]
    colors = ['#4CAF50' if sentiment == 'POSITIVE' else '#F44336', '#F44336']

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

    return result_text, fig  # Return the Matplotlib figure directly

# Custom CSS for styling
custom_css = """
body {
    background-color: #2c2f33;
    color: #f0f0f0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gr-textbox, .gr-radio {
    margin-bottom: 20px;
    border: 1px solid #444;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    background-color: #3a3d41;
}

.gr-button {
    background-color: #7289da;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: 0.3s;
    border-radius: 8px;
    margin-top: 10px;
}

.gr-button:hover {
    background-color: #5b6eae;
}

#component-2 {
    font-size: 18px;
    margin-bottom: 20px;
}

#component-3 {
    font-size: 18px;
    margin-bottom: 20px;
}

#component-4 {
    font-size: 16px;
    padding: 15px;
    background-color: #3a3d41;
    border: 1px solid #444;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    font-size: 32px;
    margin-bottom: 40px;
    color: #7289da;
}
"""

# Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>Movie Review Sentiment Analysis</h1>")

    with gr.Column():
        with gr.Row():
            review_input = gr.Textbox(
                label="Enter Movie Review", placeholder="Type your movie review here...", lines=4
            )

        with gr.Row():
            mode_input = gr.Radio(
                ["Local Pipeline", "Inference API"], label="Select Processing Mode", value="Inference API"
            )

        with gr.Row():
            analyze_button = gr.Button("Analyze Sentiment")

        # Output box to display the sentiment analysis result
        sentiment_output = gr.Textbox(label="Sentiment Analysis Result", interactive=False)
        # Plot output to display the sentiment score graph
        plot_output = gr.Plot(label="Sentiment Score Graph")

    analyze_button.click(analyze_sentiment, [review_input, mode_input], [sentiment_output, plot_output])

# Run the Gradio app
if __name__ == "__main__":
    demo.launch('share=True')
