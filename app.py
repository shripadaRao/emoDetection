from cProfile import label
import os
from flask import Flask, render_template
from flask import request

from transformers import pipeline
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

#emotion_labels = emotion("I'm sorry!!")

#print(emotion_labels[0]['score'])

def stringToList(string):
    listRes = list(string.split(". "))
    return listRes


def returnEmotion(texts):
    trainedEmotionList = []
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    textsList = stringToList(texts)
    #print(textsList)
    for text_l in textsList:
        #print('text_l' ,text_l)
        emotion_labels = emotion(text_l)
        #print("emotion_label: ", emotion_labels)
        trainedEmotionList.append(emotion_labels)
    
    #print(trainedEmotionList)
    return trainedEmotionList


#returnEmotion("I'm so sorry. I'm so Happy!")

def get_emotion_label(text):
  return(emotion(text)[0]['label'])


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
  
    if request.method == 'POST':
      form = request.form
      result = []
      text_input = form['paragraph']
      
      result.append(returnEmotion(text_input))
      
      result.append(form['paragraph'])

      return render_template("index.html",result = result)

    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

