import numpy as np
import os
from os import path as op
import sys
import matplotlib.pyplot as plt
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from google.cloud import speech
import io
import wave
import json


SCRIPT_DIR = op.dirname(__file__)
WORKSPACE_DIR = op.join(SCRIPT_DIR, "..")
credentail_path = op.join(WORKSPACE_DIR, "courseworkdevelopment-2789e2c1ca50.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentail_path

SALIENCY_DIFF_CONSTRAINT = 2
FUNDAMENTAL_FREQUENCY_FRAME_LENGTH_MS = 35
FRAME_SPACE_MS = 10


def extract_fundamental_frequency(path):
    signal = basic.SignalObj(path)
    pitchY = pYAAPT.yaapt(signal, frame_length=FUNDAMENTAL_FREQUENCY_FRAME_LENGTH_MS, frame_space=FRAME_SPACE_MS)
    sample_values = pitchY.samp_values
    return sample_values


def transcribe_file_with_word_time_offsets(path, language):
    """Transcribe the given audio file asynchronously and output the word time
    offsets."""
    client = speech.SpeechClient()
    with io.open(path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    # extract info about wav file
    wave_file = wave.open(path)
    wave_file_info = wave_file.getparams()
    wave_file_info = wave_file_info._asdict()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=wave_file_info["framerate"],
        language_code=language,
        enable_word_time_offsets=True,
    )
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for operation to complete...")
    result = operation.result(timeout=90)
    Words_info = []
    for result in result.results:
        alternative = result.alternatives[0]
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            Words_info.append({"word":  word,
                               "start": start_time.total_seconds(),
                               "finish": end_time.total_seconds()})
    return Words_info, result.alternatives[0].transcript, result.alternatives[0].confidence


def get_semitone_markup(semiton_value):
    # from relative semitones to register
    if semiton_value > 6:
        markup = 'H'
    elif semiton_value > 2:
        markup = 'h'
    elif semiton_value > -2:
        markup = 'm'
    elif semiton_value > -6:
        markup = 'l'
    elif semiton_value < -6:
        markup = 'L'
    return markup


def lowess_smoothing(x, y, f=1. / 5., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations."""
    x_length = x.shape[0]
    used_range = int(np.ceil(f*x_length))

    # establish weighting
    # distance from xi to the Nth neighbour
    distances = [np.sort(np.abs(x-x[i]))[used_range] for i in range(x_length)]
    distances = np.array(distances)
    # use numpy broadcasting rules to simplify computations
    x1 = np.expand_dims(x, 0)
    x2 = np.expand_dims(x, - 1)
    neighbours_differences = x1 - x2
    weights = np.clip(np.abs(neighbours_differences / distances), 0.0, 1.0)
    weights = np.power(1 - np.power(weights, 3), 3)

    estimated_y = np.zeros(x_length)
    delta = np.ones(x_length)
    for iteration in range(iter):
        for i in range(x_length):
            current_weights = delta * weights[:, i]
            b = np.array([sum(current_weights*y), sum(current_weights*y*x)])
            A = np.array([[sum(current_weights), sum(current_weights*x)], [sum(current_weights*x), sum(current_weights*x*x)]])
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
            estimated_y[i] = beta[0] + beta[1]*x[i]
        residuals = y - estimated_y
        s = np.median(abs(residuals))
        if s:
            delta = np.clip(residuals/(6*s), -1, 1)
            delta = 1 - delta * delta
            delta = delta * delta
        else:
            delta = 0.0
    return estimated_y


def get_semitones_from_f0(f0):
    mean_f0 = np.mean(f0)
    f0_above_zero = np.maximum(f0, np.array([1E-5]))
    semitones = 12 * np.log2(f0_above_zero / mean_f0)
    return semitones


def stylize_semitones_sequence(semitones):
    # first smooth the sequnece of semitones
    semitones_amount = semitones.shape[0]
    # get normalized time
    time = np.arange(semitones_amount)
    time = time / semitones_amount
    smoothed_semitones = lowess_smoothing(time, semitones)
    initial_value_markup = get_semitone_markup(smoothed_semitones[0])
    final_value_markup = get_semitone_markup(smoothed_semitones[-1])

    # define presence of saliency
    saliency_markup = None
    saliency_time_pos = None
    # main saliency should differ from both inital and final values by more than 2 st
    # or else it is considered insignificant and does not take part in markup
    init_diffs = np.abs(smoothed_semitones - smoothed_semitones[0])
    init_diff_condition = init_diffs > SALIENCY_DIFF_CONSTRAINT
    final_diffs = np.abs(smoothed_semitones - smoothed_semitones[-1])
    final_diff_condition = final_diffs > SALIENCY_DIFF_CONSTRAINT
    saliency_candidates_ixs = np.nonzero(np.logical_and(init_diff_condition, final_diff_condition))[0]
    if saliency_candidates_ixs.size > 0:
        # if there are points that satisfy both conditions choose the one that differs most
        cand_diffs = init_diffs[saliency_candidates_ixs] + final_diffs[saliency_candidates_ixs]
        candidate_index = saliency_candidates_ixs[np.argmax(cand_diffs)]
        saliency_markup = get_semitone_markup(smoothed_semitones[candidate_index])
        saliency_time_pos = time[candidate_index]

    stylization = {"init_markup": initial_value_markup,
                   "final_markup": final_value_markup,
                   "salient_peek_info": [saliency_markup, saliency_time_pos]}
    return stylization, smoothed_semitones


def main(sample_path, language, save_path=None):
    words_info, transcript, confidence = transcribe_file_with_word_time_offsets(sample_path, language)
    f0 = extract_fundamental_frequency(sample_path)
    semitones = get_semitones_from_f0(f0)
    visualizer_info = []
    if save_path is not None:
        markup_to_save = {"Transcription": transcript, "Words_info": []}
    for word_info in words_info:
        # choose corresponding semitones sequence and get its markup
        word = word_info["word"]
        start_time_ms = word_info["start"] * 1000
        finish_time_ms = word_info["finish"] * 1000
        start_index = int(start_time_ms / FRAME_SPACE_MS)
        finish_index = int(min(finish_time_ms / FRAME_SPACE_MS, f0.shape[0]))
        word_f0 = f0[start_index: finish_index]
        word_semitones = semitones[start_index: finish_index]
        markup, smoothed_semitones = stylize_semitones_sequence(word_semitones)
        visualizer_info.append({
            "word": word,
            "markup": markup,
            "semitones": word_semitones,
            "f0": word_f0,
            "smoothed_semitones": smoothed_semitones,
            "time": finish_time_ms - start_time_ms
        })

        if save_path is not None:
            markup_string = markup["init_markup"] + markup["final_markup"]
            if None not in markup["salient_peek_info"]:
                markup_string += markup["salient_peek_info"][0] + f'{markup["salient_peek_info"][1]:.2f}'
            markup_to_save["Words_info"].append({
                "word": word,
                "markup": markup_string,
                "start_time": start_time_ms,
                "finish_time": finish_time_ms
            })

    if save_path is not None:
        with open(save_path, 'w') as markup_file:
            json.dump(markup_to_save, markup_file, indent=4)
    return transcript, confidence, visualizer_info


if __name__ == "__main__":
    # parse command line arguments to extract audiofile path, language and save_path
    audio_path = sys.argv[1]
    language = sys.argv[2]
    save_path = sys.argv[3]
    main(audio_path, language, save_path)