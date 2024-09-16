---
title: CaseStudy1
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

# CS553 Case Study 1 - Group 11

A pipeline has been added to the [CS553CaseStudy1 GitHub repository](https://github.com/jvroo/CS553CaseStudy1.git) to ensure that [Hugging Face Space](https://huggingface.co/spaces/ML-OPS-Grp11/CaseStudy1) is automatically updated.

## Group 11
**Authors**: Keon Roohani, Shipra Poojary, Jagruti Chitte

## Assignment Overview

**Note**: The work done in this assignment used (Paffenroth, 2024) and (Wu, 2024) as a starting point. Additionally, ChatGPT was used on occasion for formatting of text. For instance this README was formatted as such. 

### Part 1.a: Inference Client Model
The use of a Client Inference model from Hugging Face was done using `"siebert/sentiment-roberta-large-english"`. This is a text sentiment analysis LLM for the English language (Hartmann et al., 2023).

### Part 2.a: Local Model
The use of a local model was done using `"distilbert-base-uncased-finetuned-sst-2-english"` (Sanh et al., 2020). This is a lightweight version of BERT that can be used to classify text sentiment.

### CI/CD Implementation
The CI/CD of this project is implemented using `main.yml`. This file ensures that pytest is run for the local model before any pushes are made to Hugging Face. After a successful test, the main commit is pushed to Hugging Face automatically. This covers sections 1.b and 2.b of the assignment.

### Testing
Pytest was used to test the local model and ensure correct functionality before deployment to the Huggingface environment.
This covers part 2.b of the assignment. The tests can be found in the tests folder. 

### Part 3: Report Submission
The report for part 3 of the Case Study is submitted on Canvas.

### Part 4: Video Submission
The video for part 4 of the Case Study is submitted on Canvas.

### Part 5: Discord Notifications
An attempt was made to connect to Discord notifications. This was done by creating a Discord server and a channel within it. A webhook was created and linked from the channel to the GitHub repository via the secrets in settings. A GitHub Action uses this secret to push a notification every time a commit to the main branch is made. This GitHub action can be found in `main.yml` for your reference. This covers section 5 of the assignment.

**Note**: The GitHub action workflow was used from [nogibjj's GitHub repository](https://github.com/nogibjj/hugging-face) as a reference. His corresponding YouTube video was particularly helpful, which can be found [here](https://www.youtube.com/watch?v=VYSGjUa5sc4&feature=youtu.be).

## References
- Hartmann, J., Heitmann, M., Siebert, C., & Schamp, C. (2023). More than a Feeling: Accuracy and Application of Sentiment Analysis. *International Journal of Research in Marketing, 40*(1), 75-87.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2020). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
- Paffenroth (2024). Prof Paffenroth's Chatbot. [Hugging Face Space](https://huggingface.co/spaces/rcpaffenroth/chatbot).
- Wu, Y. (2024). Yang's Chatbot. [Hugging Face Space](https://huggingface.co/spaces/YangWu001/CS553_Example).
