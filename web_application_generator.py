import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import slam_analysis
import sys

@st.cache
def load_data():
    transcription, confidence, analysis = slam_analysis.main(sys.argv[1], sys.argv[2])
    return transcription, confidence, analysis


transcription, confidence, analysis = load_data()
st.title(f"Transcription: {transcription}, \n Confidence: {confidence}")

options = [a["word"] for a in analysis]
word_number = st.slider("Choose word to see its markup", min_value=1, max_value=len(analysis))
st.write(f"Word number {word_number} is selected")
word_number -= 1

fig = plt.figure()
ticks_num = analysis[word_number]["smoothed_semitones"].size
xticks = np.arange(ticks_num) / ticks_num
xticks = [f"{x:.2f}" for x in xticks]
plt.plot(xticks, analysis[word_number]["semitones"], label="Semitones obtained from f0")
plt.plot(xticks, analysis[word_number]["smoothed_semitones"], label="Smoothed semitones")
xticks_labels = [f'{xtick}' if ind % 5 == 0 else '' for ind, xtick in enumerate(xticks)]
plt.xticks(xticks, xticks_labels, rotation='vertical')

plt.legend()
current_markup = analysis[word_number]["markup"]

markup_string = current_markup["init_markup"] + current_markup["final_markup"]
if None not in current_markup["salient_peek_info"]:
    markup_string += current_markup["salient_peek_info"][0] + f'{current_markup["salient_peek_info"][1]:.2f}'
plt.title(f"Word: {analysis[word_number]['word']} \nmarkup: {markup_string}")
st.pyplot(fig)
