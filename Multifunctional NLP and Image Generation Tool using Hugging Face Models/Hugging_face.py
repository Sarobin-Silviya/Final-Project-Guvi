# Import necessary packages
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from diffusers import StableDiffusionPipeline
import streamlit as st
import re
import matplotlib.pyplot as plt
from PIL import Image
import os

# Checking for CUDA availability and suggesting accelerate library
device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    st.warning("CUDA not available, running on CPU. Consider installing `accelerate` for optimized performance on CPU:\n`pip install accelerate`")

# Streamlit App Title and Description
st.title('🌟 Multifunctional NLP and Image Generation Tool using Hugging Face Models')
st.write('''
    This app provides various AI-powered functionalities, including text summarization, 
    next-word prediction, story generation, chatbot interaction, sentiment analysis, 
    question answering, and image generation.
''')

# Sidebar for task selection
task = st.sidebar.selectbox('Choose a task', [
    'Text Summarization', 'Next Word Prediction', 'Story Prediction', 
    'Chatbot', 'Sentiment Analysis', 'Question Answering', 'Image Generation'
])

# Function for text summarization
if task == 'Text Summarization':
    st.subheader('📝 Text Summarization')
    user_input = st.text_area('Enter text to summarize:')
    if st.button('Summarize'):
        with st.spinner("Generating summary..."):
            model_name = "facebook/bart-large-cnn"
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
            
            def summarize(text, model, tokenizer):
                inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
                summary_ids = model.generate(inputs, max_length=200, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
                return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            summary = summarize(user_input, model, tokenizer)
            st.write("### Summary")
            st.write(summary)

# Function for next word prediction
elif task == 'Next Word Prediction':
    st.subheader('🔮 Next Word Prediction')
    user_input = st.text_area('Enter text for prediction:')
    if st.button('Predict'):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        
        def predict_next_word(prompt, model, tokenizer, top_k=5):
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            top_k_tokens = torch.topk(next_token_logits, top_k).indices[0].tolist()
            return [tokenizer.decode([token]) for token in top_k_tokens]
        
        predicted_words = predict_next_word(user_input, model, tokenizer)
        st.write("### Next Word Predictions")
        st.write(predicted_words)

# Function for story prediction
elif task == 'Story Prediction':
    st.subheader('📖 Story Prediction')
    user_input = st.text_area('Enter text to continue the story:')
    if st.button('Generate'):
        with st.spinner("Generating story..."):
            story_predictor = pipeline('text-generation', model='gpt2')
            story = story_predictor(user_input, max_length=200, clean_up_tokenization_spaces=True)[0]['generated_text']
            st.write("### Generated Story")
            st.write(story)

# Function for chatbot interaction
elif task == 'Chatbot':
    st.subheader('🤖 Chatbot')
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    chat_history_ids = None
    user_input = st.text_input("You:")
    if st.button('Chat'):
        if user_input.lower() != 'quit':
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            st.write("Bot:", response)

# Function for sentiment analysis
elif task == 'Sentiment Analysis':
    st.subheader('😊 Sentiment Analysis')
    user_input = st.text_area('Enter text for sentiment analysis:')
    if st.button('Analyze'):
        sentiment_analysis = pipeline("sentiment-analysis")
        results = sentiment_analysis(re.split(r'([.!?])', user_input))
        for result in results:
            st.write(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")

# Function for question answering
elif task == 'Question Answering':
    st.subheader('❓ Question Answering')
    context = st.text_area('Enter context text:')
    question = st.text_input('Enter question:')
    if st.button('Answer'):
        with st.spinner("Finding answer..."):
            model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
            
            inputs = tokenizer.encode_plus(question, context, return_tensors="pt").to(device)
            answer_start_scores, answer_end_scores = model(**inputs).start_logits, model(**inputs).end_logits
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0][torch.argmax(answer_start_scores):torch.argmax(answer_end_scores) + 1])
            )
            st.write("Answer:", answer)

# Function for image generation
elif task == 'Image Generation':
    st.subheader('🖼️ Image Generation')
    user_input = st.text_area('Enter prompt for image generation:')
    if st.button('Generate Image'):
        with st.spinner("Generating image..."):
            pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
            image = pipe(user_input).images[0]
            st.image(image)
