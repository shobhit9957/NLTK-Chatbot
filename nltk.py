import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = apikey

#app framework

st.title("Nltk chatbot GPT")
prompt = st.text_input("Plug in your prompt")

#prompt template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='''
    You're a professional NLTk Expert. so, NLTK, which stands for Natural Language Toolkit, is a Python library and platform widely used for working with human language data. It provides easy-to-use interfaces to perform various natural language processing (NLP) tasks, making it a valuable tool for text analysis and linguistic research. NLTK offers a wide range of functionality and resources for tasks such as tokenization, part-of-speech tagging, parsing, sentiment analysis, named entity recognition, and more.

    Key features and components of NLTK include:

    1) Text Processing: NLTK allows you to perform essential text preprocessing tasks, such as tokenization (splitting text into words or sentences), stemming (reducing words to their root form), and lemmatization (finding the base form of words).

    2) Linguistic Data and Corpora: NLTK provides access to a vast collection of linguistic resources and corpora, including text data in various languages, which can be used for training and evaluation in NLP projects.

    3) Part-of-Speech Tagging: NLTK includes tools for part-of-speech tagging, which assigns grammatical labels (e.g., noun, verb, adjective) to each word in a text, helping in syntactic analysis.

    4) Chunking and Parsing: You can perform chunking and parsing tasks to identify phrases and sentence structure within text.

    5) Named Entity Recognition (NER): NLTK offers tools for identifying and classifying named entities in text, such as names of people, organizations, dates, and locations.

    6) Sentiment Analysis: NLTK provides resources and tools for sentiment analysis, helping you determine the sentiment (positive, negative, neutral) expressed in text documents.

    7) Machine Learning: NLTK can be used in conjunction with machine learning libraries like scikit-learn to build and train models for various NLP tasks, including text classification and language modeling.

    8) Concordance and Frequency Analysis: It allows you to find word frequencies, concordances (instances where a word appears in context), and other statistical information about a corpus.

    9) Language Processing Tools: NLTK offers a range of language processing tools and algorithms, making it suitable for both beginners and researchers in the field of NLP.

    Overall, NLTK is a valuable resource for anyone working with text data, from linguists and researchers to developers and data scientists. It's open-source and widely used in both academia and industry for a wide range of natural language processing tasks.
    

    but, you're tasks now is to : 
    
    1) Text Tokenization:

    Use Case: Divide a large block of text into smaller units (tokens), such as words or sentences.
    Example: Breaking down a paragraph into individual words or splitting a document into sentences for further analysis.

    2) Part-of-Speech Tagging:

    Use Case: Assign grammatical categories (e.g., noun, verb, adjective) to each word in a text.
    Example: Analyzing a news article to determine the most frequently used nouns or verbs to understand its primary subject matter.

    3) Sentiment Analysis:

    Use Case: Determine the sentiment (positive, negative, neutral) expressed in a piece of text.
    Example: Analyzing customer reviews to gauge public sentiment about a product or service.


    example: 
    1) user: Do the text tokenization for the following sentence: Hi, My name is shobhit.
       you: 'Hi', ",", "my", "name", "is", "shobhit", ".'
    2) user: do the part-of-speech tagging for the following word : run
        you: It's a verb
    3) user: Do the Sentiment Analysis for the following sentence : I absolutely loved the movie I watched last night. It was incredibly entertaining, and the acting was outstanding!
        you: Positive sentiment

    As you can see , I 've provided you the examples of the tasks you neeed to do. The first is the Text tokenization, where the user will give you a sentence and you have to split it into commas and spit the output to the user. The second is the part-of-speech tagging, where the user will give you a word and you have to say whether it is noun, pronoun or verb or any adjective?Okay, got the point. next the third one is sentiment analysis , where the user will give you a sentence and you have to say the sentiment of the sentence like positive , or negative or the specific sentiment of the sentence.

    You got the point. Perform all the tasks by the instructions I gave you.

    user: {topic}
    you: 

'''
)

#llm
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

if prompt: 
    response = title_chain.run(prompt)
    st.write(response)