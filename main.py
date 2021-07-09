import json
import os

import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
from torchvision.io import read_image
from transformers import BertTokenizer,BertModel

import matplotlib.pyplot as pp
import matplotlib.image as img



# image - question - answer 형태로 torch.data.dataset 구성
# question.json에서 question ID와 annotation.json에서
# question을 matching하여 질문 한 쌍 구성 후, 해당하는 image와 함께 return
# !!!!!! dataloader로 batch processing 구성



class myDataset(torch.utils.data.Dataset):

    # data의 link 저장
    def __init__(self, annotationData, MquestionData, OquestionData, imageData):
        super(myDataset,self).__init__()
        self.annotationData = annotationData
        self.MquestionData = MquestionData
        self.OquestionData = OquestionData
        self.imageData = imageData

        self.annotationDict = dict()
        self.questionDict = dict()
        self.imagePathList = list()

        # annotationDict, questionDict, imagePathList 만들기
        for i in range(len(self.annotationData['annotations'])):

            question_id = self.annotationData['annotations'][i]['question_id']
            image_id = self.annotationData['annotations'][i]['image_id']

            image_id = os.path.join("train2014", "train2014", "COCO_train2014_000000" + str(image_id) + ".jpg")

            if self.MquestionData['questions'][i]['question_id'] == question_id:
                question = self.MquestionData['questions'][i]['question']
                self.questionDict[question_id] = question # questionDict

            self.imagePathList.append(image_id) # imagePathList


            answerList = []
            for j in range(len(self.annotationData['annotations'][i]['answers'])):
                answer_id = self.annotationData['annotations'][i]['answers'][j]['answer']
                answerList.append(answer_id) # answerList를 만들고 annotationDict에 넣기
                self.annotationDict[question_id] = answerList # annotationDict

            if i == 10:
                break
        print()
        print("annotationDict:")
        print(self.annotationDict)
        print("questionDict:")
        print(self.questionDict)
        print("imagePathList:")
        print(self.imagePathList)

        quit()



        self.transform = transforms.Compose([
            transforms.Resize([64, 64]) # 일단 (64, 64)로 Resize
        ])

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # tokenizer question/answer에 사용 ??????

    def __len__(self):
        return

    # 각 data를 한 개씩 return
    # 이미지를 동일한 size로 transform 필요
    # image - question - answer를 한 개씩 return
    def __getitem__(self, idx):
        if self.annotationData['annotations'][idx]['question_id'] == self.MquestionData['questions'][idx]['question_id'] or \
                self.annotationData['annotations'][idx]['question_id'] == self.OquestionData['questions'][idx]['question_id']:

            question = self.MquestionData['questions'][idx]['question'] # MquestionData == OquestionData
            answer = self.annotationData['annotations'][idx]['answers'] # 일단 list 자체 뽑기 (multiple choice answer 없다고 생각)

            imagePath = self.MquestionData['questions'][idx]['image_id']
            imagePath = os.path.join("train2014", "train2014", "COCO_train2014_000000" + str(imagePath) + ".jpg")

            img = read_image(imagePath) # imagePath의 image를 불러오고
            img = self.transform(img) # transform을 통해 Resize
            img = img.float() # read_image는 byte로 수를 저장하기 때문에 float로 변환

            tok = self.tokenizer(self.MquestionData['questions'][idx]['question'])

            print(tok)

            return question, answer, img # question & answer는 list 형태, img는 float 형태




# question이 들어오면 tokenization한 뒤에, BERT  -> output에서 CLS token의 result만 사용 (1-dim vector)
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.conv1 = nn.Conv2d()



# # test annotation
# with open('mscoco_val2014_annotations.json', 'r') as f_question:
#     json_MquestionData = json.load(f_question)
# # print(json.dumps(json_MquestionData, indent="\t")) #json formatting
#
# # 121511
# for i in range(len(json_MquestionData['annotations'])):
#     if json_MquestionData['annotations'][i]['question_id'] == 4870250:
#         print("['annotations']", [i],['question_id'], ": ", json_MquestionData['annotations'][i]['question_id'])
#         print("['annotations']", [i], ": ", json_MquestionData['annotations'][i])



# image
with open('mscoco_train2014_annotations.json', 'r') as f_annotation:
    json_annotationData = json.load(f_annotation)
# print(json.dumps(json_annotationData, indent="\t")) #json formatting


# multiple choice question (객관식)
with open('MultipleChoice_mscoco_train2014_questions.json', 'r') as f_question:
    json_MquestionData = json.load(f_question)
# print(json.dumps(json_MquestionData, indent="\t")) #json formatting


# open ended question (주관식)
with open('OpenEnded_mscoco_train2014_questions.json', 'r') as f_question:
    json_OquestionData = json.load(f_question)
# print(json.dumps(json_OquestionData, indent="\t")) #json formatting


# for i in range(len(json_OquestionData['questions'])):
#     # print("['annotations']", [0], ": ", json_OquestionData['questions'][0]['question'])
#
#     # "man doing" string이 question의 어디에 있는지 check
#     if "man doing" in json_OquestionData['questions'][i]['question']:
#         print("['annotations']", [i], ": ", json_OquestionData['questions'][i])
#
# quit()

# image file directory
path = os.path.join("train2014", "train2014")
imageData = [os.path.join(path, i) for i in os.listdir(path)]

# main
dataset = myDataset(json_annotationData, json_MquestionData, json_OquestionData, imageData)

batch = 8
myDataLoader = DataLoader(myDataset, batch_size=1)

# print(myDataLoader)

for i in myDataLoader:
    print("a")



