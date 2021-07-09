import json


# torch.data.dataset  ! Custom Dataset
# 데이터셋 다루는 도구로 torch.utils.data.Dataset, torch.utils.data.DataLoader 제공
# --> mini batch 학습, 데이터 shuffle, parallel 처리
# Dataset 정의 -> DataLoader에 전달







######################################### mscoco_train2014_annotations.json #########################################

# image
with open('mscoco_train2014_annotations.json', 'r') as f_annotation:
    json_annotationData = json.load(f_annotation)
# print(json.dumps(json_annotationData, indent="\t")) #json formatting

print(json_annotationData.keys())


# print("['annotations']['question_type']", json_annotationData['annotations'][0]['question_type'])
# print("['annotations']['question_type']['answers'] :", json_annotationData['annotations'][0]['answers'])
# print("['annotations']['question_type']['answers'][0] :", json_annotationData['annotations'][0]['answers'][0])


# for i in range(248349):
#     print("['annotations']", [i], ": ", json_annotationData['annotations'][i])



# json_annotationData['annotations']의 길이: 248,349
# 데이터 구조:
# ['annotations'][i]: {question_type: what, multiple_choice_answer: curved, answers:[{answer: oval, answer_confidence: yes, answer_id: 1}],
#                                                                             {answer: semi circle, answer_confidence: yes, answer_id: 2},
#                                                                             {answer: curved, answer_confidence: yes, answer_id: 3},
#                                                                             {answer: curved, answer_confidence: yes, answer_id: 4},
#                                                                             {answer: double curve, answer_confidence: yes, answer_id" 5},
#                                                                             {answer: banana, answer_confidence: maybe, answer_id: 6},
#                                                                             ... {answer: curved, answer_confidence: maybe, answer_id: 10}],
# image_id: 487025, answer_type: other, question_id: 4870250} --> 같은 image_id가 3개



############################################################################################################################################################






######################################### MultipleChoice_mscoco_train2014_questions.json #########################################

# multiple choice question (객관식)
with open('MultipleChoice_mscoco_train2014_questions.json', 'r') as f_question:
    json_questionData = json.load(f_question)
# print(json.dumps(json_questionData, indent="\t")) #json formatting



# for i in range(248349):
#     print("['questions']",[i],": ", json_questionData['questions'][i])

# print("['questions']: ", json_questionData['questions'])


# json_questionData['questions']의 길이: 248,349   multiple_choices는 항상 len() = 18
# 데이터 구조:
# ['questions'][i]: {image_id: 219502, question: Is he a prince?,
# multiple_choices: ['on wrapper', '2', 'red', '3', 'green', 'southern', 'northern', 'white', '4', 'yes', 'cars coming',
#                       'blue', '1', 'no', 'irish lion', 'monitor', 'not', 'carry stuff'],
# question_id: 2195021} --> 같은 image_id가 3개, question


############################################################################################################################################################




######################################### OpenEnded_mscoco_train2014_questions.json #########################################


# open ended question (주관식)
with open('OpenEnded_mscoco_train2014_questions.json', 'r') as f_question:
    json_questionData = json.load(f_question)
# print(json.dumps(json_questionData, indent="\t")) #json formatting



# for i in range(248349):
#     print("['questions']",[i],": ", json_questionData['questions'][i])

# json_questionData['questions']의 길이: 248,349
# 데이터 구조:
# {question: What kind of shirt is this?, image_id: 343608, question_id: 3436081} --> 같은 image_id 당 question 3개


############################################################################################################################################################



# torch.data.dataset  ! Custom Dataset
# 데이터셋 다루는 도구로 torch.utils.data.Dataset, torch.utils.data.DataLoader 제공
# --> mini batch 학습, 데이터 shuffle, parallel 처리
# Dataset 정의 -> DataLoader에 전달

import torch
from torch.utils.data import Dataset, DataLoader

# image - question - answer 형태로 torch.data.dataset 구성
# question.json에서 question ID와 annotation.json에서
# question을 matching하여 질문 한 쌍 구성 후, 해당하는 image와 함께 return
# !!!!!! dataloader로 batch processing 구성


# 각 data를 한 개씩 return
    # 이미지를 동일한 size로 transform 필요
    # image - question - answer를 한 개씩 return



class myDataset(torch.utils.data.Dataset):
    def __init__(self): # data의 link 저장
        super(myDataset,self).__init__()

        return

    def __len__(self):
        return


    def __getitem__(self, idx):
        return




#
# import torch
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from transformers import BertTokenizer, BertModel
#
# class MyDataset(Dataset):
#     def __init__(self, data_path): # 데이터 불러오기
#         super(MyDataset, self).__init__()
#         self.data = pd.read_csv(data_path, sep="\t")[:100]
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
#     def __len__(self): #
#         return len(self.data)





        # for i in range(248349):
        #     # annotationData & MquestionData
        #     if self.annotationData['annotations'][i]['question_id'] == self.MquestionData['questions'][0]['question_id']:
        #         question = self.MquestionData['questions'][i]['question']
        #         answer = self.MquestionData['questions'][i]['multiple_choices']
        #
        #         imageNumber = self.MquestionData['questions'][i]['image_id']
        #         imageNumber = "train2014/train2014\\" + "COCO_train2014_000000" + str(imageNumber) + ".jpg"
        #
        #         for i in range(len(imageData)):
        #             if imageData[i] == imageNumber:
        #                 imagePath = imageData[i]
        #         return question, answer, imagePath
        #     # elif  self.annotationData['annotations'][i]['question_id'] == self.OquestionData['questions'][0]['question_id']:





# print("AnnotationData question_id: ", self.annotationData['annotations'][0]['question_id'])
#         print("AnnotationData question_type: ",self.annotationData['annotations'][0]['question_type'])
#         print("AnnotationData image_id: ", self.annotationData['annotations'][0]['image_id'])
#         print("AnnotationData answers: ", self.annotationData['annotations'][0]['answers'])
#         print()
#
#         imageNumber = self.annotationData['annotations'][0]['image_id']
#         imageNumber = "train2014/train2014\\" + "COCO_train2014_000000" + str(imageNumber) + ".jpg"
#
#
#         for i in range(len(imageData)):
#             if imageData[i] == imageNumber:
#                 imagePath = imageData[i]
#                 print(imagePath)
#                 print()
#
#         print("MquestionData question_id: ", self.MquestionData['questions'][0]['question_id'])
#         print("MquestionData question: ", self.MquestionData['questions'][0]['question'])
#         print("MquestionData multiple_choices: ", self.MquestionData['questions'][0]['multiple_choices'])
#         print()
#
#         print("OquestionData question_id: ",self.OquestionData['questions'][0]['question_id'])
#         print("OquestionData question: ", self.OquestionData['questions'][0]['question'])