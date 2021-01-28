import nltk
from nltk.util import Index
nltk.download('punkt')
import pandas as pd
import os
from tqdm import tqdm
os.environ["MODEL_DIR"] = './content/'
model_dir='./content'
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

#split the paragraph in to sentences.
def split_para(text):
  from nltk import tokenize
  out=tokenize.sent_tokenize(text)
  return out

def keyboard_aug(text):
  aug = nac.KeyboardAug()
  augmented_text = aug.augment(text)
  return augmented_text

def spelling_aug(text):
  aug=naw.SpellingAug()
  augmented_text= aug.augment(text, n=1)
  return augmented_text

def word2vec_aug(text):
  aug = naw.WordEmbsAug(
    model_type='word2vec', model_path='GoogleNews-vectors-negative300.bin',
    action="substitute")
  augmented_text = aug.augment(text)
  return augmented_text

def bert_sub_aug(text):
    aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute")
    augmented_text = aug.augment(text)
    return augmented_text

def bert_ins_aug(text):
    aug = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased', action="insert")
    augmented_text = aug.augment(text)
    return augmented_text

def xlnet_sub_aug(text):
    aug=naw.ContextualWordEmbsAug(
    model_path='xlnet-base-cased',action="substitute")
    augmented_text=aug.augment(text)
    return augmented_text


def run_augmentation(func,newaugs,df):
  for idx,row in tqdm(df.iterrows(),total=df.shape[0]):
        #text=row['Creation Content_last']
        text=str(row['text'])
        #print(type(text))
        #flag=row['Left_out_flag']
        flag=row['label']
        #sents=split_para(text)
        sents=text.split('@@|')
        s=len(sents)
        #print('num of sent {}'.format(s))
        cnt=0
        #change one sentence each time.
        for i in range(s):
            augmented_text=globals()[a](sents[i])
            new_text=''
            for k in range(s):
                if k==i:
                    new_text+=augmented_text
                else:
                    new_text+=sents[k]
            newaugs.append([new_text,flag])
            #print('save augmentation for para changed sent {}'.format(idx))
  return newaugs

def save_em(name,newaugs,outdf1):
    newaugdf=pd.DataFrame(newaugs,columns=['text', 'label'])
    outdf1=outdf1.append(newaugdf,ignore_index=True)
    outdf1.to_csv(name,index=False,header=False)
    print('saved to {}'.format(name))
    return outdf1

import argparse
from time import time
start_time=time()
parser=argparse.ArgumentParser(description="augmentations to run on google drive csv format columns=text,label")
parser.add_argument("-gdrive", help="gdrive url (must be shared to all with link)")
parser.add_argument("-output",help="output file stem (ex. saveit.csv)")
parser.add_argument("-augs",help="use a comma separate list of augmentations keyboard_aug, spelling_aug,word2vec_aug,bert_sub_aug,bert_ins_aug,xlnet_sub_aug")

args=parser.parse_args()

url=args.gdrive
import pandas as pd

path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path,names=['text','label'])
newaugs=[]
output=args.output
augs=[x.strip() for x in args.augs.split(",")]
print(augs)
for a in augs:
  newaugs=run_augmentation(a,newaugs,df)
  name=a+"_"+output
  df=save_em(name,newaugs,df)
  print("{} has {} records".format(name,df.shape[0]))
print(f"total time {time()-start_time:0.4f}")