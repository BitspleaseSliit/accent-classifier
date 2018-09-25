import os
#import glob
import numpy as np
import subprocess
from pydub import AudioSegment
from scipy.io import wavfile
from PIL import Image
from matplotlib import pyplot as plt
AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe ="C:\\ffmpeg\\bin\\ffprobe.exe"
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('tf')


classifier_path = './models/model.h5'
classifier = load_model(classifier_path)
model = InceptionV3(weights='imagenet', include_top=False )


def videoAudioToWav(video):
    command = "ffmpeg -i "+video+" -ab 160k -ac 1 -ar 44100 -vn audio.wav"
    subprocess.call(command, shell=True)
    
    
def sliceAudio(audio):
    t1 = 5000  #Works in milliseconds
    t2 = 25000
    newAudio = AudioSegment.from_wav(audio)
    newAudio = newAudio[t1:t2]
    newAudio.export('slice.wav', format="wav")
    
    
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    #mono = data[:,0]
    return rate, data


def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    print(type(data), len(data))
    nfft = 256  # Length of the windowing segments
    fs = 256  # Sampling frequency
    pxx, freqs, bins, im = plt.specgram(data, nfft, fs)
    print("pxx : ", len(pxx))
    print("freqs : ", len(freqs))
    print("bins : ", len(bins))
    # plt.axis('on')
    # plt.show()
    plt.axis('off')
    print(wav_file.split('.wav')[0])
    plt.savefig(wav_file.split('.wav')[0] + '.jpg',
                dpi=100,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)


def predict(file):
    global model
    x = image.load_img(file, target_size=(150,150))
    x = image.img_to_array(x)
    x = x/255
    x = np.expand_dims(x, axis=0) 
    features = model.predict(x)
    result = classifier.predict_classes(features)
    if result[0] == 0:
        prediction ='native'
    elif result[0] == 1:
        prediction = 'non-native'
    return prediction

def audio_classification(video):
    videoAudioToWav(video)
    sliceAudio('audio.wav')
    graph_spectrogram('slice.wav')
    #predict('slice.jpg')
    
audio_classification('test.mp4')
predict('slice.jpg')
    
    
    


    


