import gradio as gr
import numpy as np
from transformers import pipeline

title = "Masked Language Model"
description = """
Write a sentence and put 'miss1'(without quotes) at the place where you want to predict the most suitable word.

For example, Mark is the cofounder of miss1

As it is fine tuned on IMDB dataset, therefore it's prediction will be somewhat related to movies.
<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""

article = "Check out [my github repository](https://github.com/Neural-Net-Rahul/P3-Fine-tuning-a-masked-language-model) and my [fine tuned model](https://huggingface.co/neural-net-rahul/distilbert-finetuned-imdb)"

textbox = gr.Textbox(label="Type your sentence here :", placeholder="My name is Bill Gates.", lines=3)

model = pipeline('fill-mask',model='neural-net-rahul/distilbert-finetuned-imdb')

def predict(text):
  list1 = text.split()
  found = False
  index = -24;
  for i in range(0,len(list1)):
    if ("miss1" in list1[i] and len(list1[i])==6):
      index = i;
      list1[i] = list1[i][5:];
      found = True
      break
    elif list1[i]=='miss1':
      list1[i] = "[MASK]"
      found = True
      break
  if found == False:
    return text
  if index != -24:
    list1.insert(index,"[MASK]")
  text = " ".join(list1)
  return model(text)[0]['sequence']


gr.Interface(
    fn=predict,
    inputs=textbox,
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=[["Mark founded miss1, shaping global social media connectivity."], ["Delhi is the most miss1 state after Kerala"]],
).launch(share=True)