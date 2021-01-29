import nltk
from nltk.util import Index
nltk.download('punkt')
import pandas as pd
import os
from tqdm import tqdm
os.environ["MODEL_DIR"] = '/spell/content/'
model_dir='/spell/content'
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action, action

#split the paragraph in to sentences.
def split_para(text):
  from nltk import tokenize
  out=tokenize.sent_tokenize(text)
  return out

def keyboard_aug(row):
  aug=nac.KeyboardAug()
  new_row=row.to_dict()
  new_row['text'] = aug.augment(row['text'])
  aug_out.append(new_row)

def spelling_aug(row):
  aug=naw.SpellingAug()
  new_row=row.to_dict()
  new_row['text']=aug.augment(row['text'],n=1)
  aug_out.append(new_row)

def word2vec_aug(row):
  aug=naw.WordEmbsAug(
    model_type='word2vec', model_path='/spell/leftout/GoogleNews-vectors-negative300.bin',
    action="substitute")
  new_row=row.to_dict()
  new_row['text']=aug.augment(row['text'])
  
def bert_sub_aug(row):
  aug=naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute")
  new_row=row.to_dict()
  new_row['text']=aug.augment(row['text'])

def nbert_ins_aug(row):
  aug=naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased', action="insert")
  new_row=row.to_dict()
  new_row['text']=aug.augment(row['text'])

def xlnet_sub_aug(row):
    aug=naw.ContextualWordEmbsAug(
    model_path='xlnet-base-cased',action="substitute")
    new_row=row.to_dict()
    new_row['text']=aug.augment(text)


def save_em(output,a,outdf1):
    name=a+"_"+output
    outdf1.to_csv(name,index=False,header=False)
    print('saved to {}'.format(name))
    return outdf1

def run_augmentation(func,newaugs,df,nodisp):
  from time import time
  #split sentences, build new df
  output=[]
  def new_split_para(row):
    from nltk import tokenize
    out=tokenize.sent_tokenize(row['text'])
    for o in out:
      new_row=row.to_dict()
      new_row['text']=o
      output.append(new_row)

  
  df.progress_apply(lambda row: new_split_para(row),axis=1)
  new_df=pd.DataFrame(output)
  #run the augmentation on new dataframe
  global aug_out
  aug_out=[]
  new_df_list=new_df['text'].tolist()
  new_df_label=new_df['label'].tolist()
  for a in func:
    print('starting {} augmentation'.format(a))
    start_time=time()
    new_method=func_dict[a]['method']
    meth_args=func_dict[a]['args']
    m_pieces=new_method.split(".")
    m=globals()[m_pieces[0]]
    func=getattr(m,m_pieces[1])
    aug=nafc.Sometimes(func(**meth_args))
    aug_out=aug.augment(new_df_list)

    tmp_df=pd.DataFrame(list(zip(aug_out,new_df_label)),columns=["text","label"])
    new_df=new_df.append(tmp_df,ignore_index=True)
    aug_out=[]
    print('{} augmentation complete, time elapsed {}'.format(a,(datetime.timedelta(seconds=time()-start_time))))
    save_em(args.output,a,new_df)
  return new_df

from time import time
import datetime
start_time=time()
import argparse
parser=argparse.ArgumentParser(description="augmentations to run on google drive csv format columns=text,label")
parser.add_argument("-gdrive", help="gdrive url (must be shared to all with link)")
parser.add_argument("-output",help="output file stem (ex. saveit.csv)")
parser.add_argument("-augs",help="use a comma separate list of augmentations keyboard_aug, spelling_aug,word2vec_aug,bert_sub_aug,bert_ins_aug,xlnet_sub_aug")
parser.add_argument("-nodisp",help="supress display",action="store_true")
args=parser.parse_args()

func_dict={"keyboard_aug":{"method":"nac.KeyboardAug","args":{}},
  "spelling_aug":{"method":"naw.SpellingAug","args":{}},
  "word2vec_aug":{"method":"naw.WordEmbsAug","args":{"model_type":"word2vec","model_path":"GoogleNews-vectors-negative300.bin","action":"substitute"}},
  "bert_sub_aug":{"method":"naw.ContextualWordEmbsAug","args":{"model_path":"bert-base-uncased","action":"substitute"}},
  "bert_ins_aug":{"method":"naw.ContextualWordEmbsAug","args":{"model_path":"distilbert-base-uncased","action":"insert"}},
  "xlnet_sub_aug":{"method":"naw.ContextualWordEmbsAug","args":{"model_path":"distilbert-base-uncased","action":"insert"}}

}
url=args.gdrive
import pandas as pd
if args.nodisp==False:
    tqdm.pandas()
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path,names=['text','label'])
print('downloaded file {} records to start'.format(df.shape[0]))
print('label values {}'.format(df['label'].value_counts()))
newaugs=[]
output=args.output
nodisp=args.nodisp
augs=[x.strip() for x in args.augs.split(",")]
print(augs)
newaugs_df=run_augmentation(augs,newaugs,df,nodisp)
print('{} total records'.format(newaugs_df.shape[0]))
