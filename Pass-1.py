import nltk
import numpy as np
import numpy
import pandas as pd 
import re
import math
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import emoji
import math
from collections import Counter

WORD = re.compile(r"\w+")
stop_words = set(stopwords.words('english')) 
my_punctuation = '!"$%&\'()*+,.:;<=>?[\\]^_`{|}~•@#'   # left with / and -   my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

def give_emoji_free_text(text):
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

def murge_quantity(text):
    flist = ['hours','hour','km/h','km/hr','kph','km','kmph','days','km/day','hrs','am','pm','mm/hr','mm',
         'miles','mph','hpa','kilometers']
    slist=text.split()
    for i in range(len(slist)):
        if slist[i].isdigit() and i<len(slist)-1 and slist[i+1] in flist:
            ind = text.find(slist[i])
            ind+=len(slist[i])
            text=text[:ind]+text[ind+1:]
    return text
    
def murge_date(text):                 #4 dec 2014 to 4dec2014
    flist = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','december']
    slist=text.split()
    for i in range(len(slist)):
        if slist[i].isdigit() and i<len(slist)-1 and slist[i+1] in flist and i<len(slist)-2 and slist[i+2].isdigit():
            si = text.find(slist[i+1])
            ei = si + len(slist[i+1])
            text=text[:si-1]+text[si:ei]+text[ei+1:]
    return text 

def murge_day(text):                 #4 dec to 4dec OR dec 4 to dec4
    flist = ['dec']
    slist=text.split()
    for i in range(len(slist)):
        if slist[i].isdigit() and i<len(slist)-1 and slist[i+1] in flist:
            ind = text.find(slist[i])
            ind+=len(slist[i])
            text=text[:ind]+text[ind+1:]
            
        if slist[i] in flist and i<len(slist)-1 and slist[i+1].isdigit():
            ind = text.find(slist[i])
            ind+=len(slist[i])
            text=text[:ind]+text[ind+1:]
    return text

def remove_less_then_3(text):
    new_text=""
    for word in text:
        if len(word)>2:
            new_text += word +' ' 
    return new_text

Noise=['utc','rt','mt','wp','wt','ts','cma','pst','jtwc','rubyph','chuuk','jma','22w','/','w/','f',
    'd','-','--','l','w-n-w','e','ir','m','x','t','b','m','r','1/','h','wx','cc','fr','w','http/','s'
    ,'http//','htt','"','-','ct','g','n','...','ndrrmc', 'hurricane', 'shooting']

def Preprocessing(text):
    text= re.sub('http://\S+|https://\S+', '', text)            # remove url
    text= " ".join(filter(lambda x:x[0]!='@', text.split()))    # remove @twitter
    text= give_emoji_free_text(text)                            # remove emotions
    text= re.sub("[\(\[].*?[\)\]]", "", text)                   # remove (..) [..]
    text= re.sub('['+my_punctuation + ']+', '', text)           # remove punctuation
    text= text.lower()                                          # lower case
    text= re.sub('([0-9]+)([a-z]+)\s'," ", text)                # remove string contain digit+char 
    text= re.sub('([a-z]+)([0-9]+)'," ", text)                  # remove string contain char+digit   
    word_tokens= word_tokenize(text)                        
    text= str(' '.join([w for w in word_tokens if not w in Noise]))  # remove special words 
    text=murge_quantity(text)                                   # murge speed 
    text=murge_day(text)                                        # combine day
    text=murge_day(text)                                        # combine date
    text= text.split()
    text= remove_less_then_3(text)                              # remove words with length less than 3
    word_tokens = word_tokenize(text) 
    filtered_text = str(' '.join([w for w in word_tokens if not w in stop_words]))
    return filtered_text

##### Input tweets
data = pd.read_csv('LA_airport_shootings_input_data.csv')
data1= pd.read_csv('LA_airport_shootings_input_data.csv')
data2= pd.read_csv('LA_airport_shootings_input_data.csv')

data['text'] = data['text'].apply(Preprocessing)
data=data.drop_duplicates()
data.reset_index(drop=True,inplace=True)
data2['text'] = data2['text'].apply(Preprocessing)

original_tweet=[]
for i in range(len(data)):
    for j in range(len(data2)):
        if data['text'][i]==data2['text'][j]:
            original_tweet.append(data1['text'][j])
            break

data= pd.DataFrame({'text': original_tweet})
data1= pd.DataFrame({'text': original_tweet})
data['text'] = data['text'].apply(Preprocessing)

######## Extra work on preprocessing  LAShoot
for i in range(len(data)):
    if 'sendingprayers' in data['text'][i]:
        data['text'][i] = data['text'][i].replace('sendingprayers', 'sending prayers')
    if 'prayforla' in data['text'][i]:
        data['text'][i] = data['text'][i].replace('prayforla', 'pray for la')
    if 'prayforlax' in data['text'][i]:
        data['text'][i] = data['text'][i].replace('prayforlax', 'pray for lax')
    if 'prayforallvictims' in data['text'][i]:
        data['text'][i] = data['text'][i].replace('prayforallvictims', 'pray forallvictims')
    if 'prayingNoonegothurt' in data['text'][i]:
        data['text'][i] = data['text'][i].replace('prayingNoonegothurt', 'praying Noonegothurt')
		
res=[]
for l in data['text']:
    t = nltk.word_tokenize(l)
    pos_tagged = nltk.pos_tag(t)
    test=[]
    for i in range(len(pos_tagged)):
        if pos_tagged[i][1]=='NN' or pos_tagged[i][1]=='NNS' or pos_tagged[i][1]=='NNP'or pos_tagged[i][1]=='NNPS' or pos_tagged[i][1]=='VB'or pos_tagged[i][1]=='VBN' or pos_tagged[i][1]=='VBZ' or pos_tagged[i][1]=='VBD' or pos_tagged[i][1]=='VBP' or pos_tagged[i][1]=='VBG' or pos_tagged[i][1]=='JJ' or pos_tagged[i][1]=='JJR' or pos_tagged[i][1]=='JJS':
            test.append(pos_tagged[i][0])
    res.append(test)   

list_new=[]
for i in range(len(res)):
    l=[]
    for j in range(len(res[i])):
        p=res[i][j].split('-')
        if len(p)==2:
            l.append(p[0])
            l.append(p[1])
        else:
            l.append(res[i][j])
    ll=[]
    for j in range(len(l)):
        p=l[j].split('/')
        if len(p)==2:
            ll.append(p[0])
            ll.append(p[1])
        else:
            ll.append(l[j])    
    list_new.append(ll)
    
combined_list_new=[]
for i in range(len(list_new)):
    new_text=""
    for word in list_new[i]:
        if len(word)>3:
            new_text += word +' ' 
    combined_list_new.append(list(set(new_text.split())))
     
##### Step-4 from ontology

my_punctuation = '<>'   # only < >

def Preprocess(text):
    text = re.sub('['+my_punctuation + ']+', '', text)           # remove punctuation
    text = re.sub(r'\s+', ' ', text)                             # remove extra spaces between words
    text = text.strip()                                         # remove gap
    return text

document=[]    
f = open("empathi.txt", "r", encoding="utf8")
for x in f:
    document.append(x)
    
data= pd.DataFrame({'text': document})
data_original= pd.DataFrame({'text': document})
data['text'] = data['text'].apply(Preprocess)
    
class_list=[]
for i in range(len(data)):
    if data['text'][i]=='Declaration':
        j=i+1
        while data['text'][j] !='/Declaration':
            d = data['text'][j].split()
            if 'Class' in d:
                p = re.findall(r'"([^"]*)"', data['text'][j])
                q = p[0].split('/')
                if len(q)>1:
                    length=len(q)
                    qq=q[length-1].split('#')
                    if len(qq)>1:
                        class_list.append(qq[1])
                    else:
                        class_list.append(qq[0])
                else:
                    qq=q[0].split('#')
                    if len(qq)>1:
                        class_list.append(qq[1])
                    else:
                        class_list.append(p[0])
            j=j+1

label_list1=[]   
for i in range(len(data)):
    data_original['text'][i]=  data_original['text'][i].strip()
    if data_original['text'][i]=='<AnnotationAssertion>':
        j=i+2
        while data_original['text'][j].strip() !='</AnnotationAssertion>':
            d= re.sub("[\<\[].*?[\>\]]", "", data_original['text'][j])
            q = d.split('/')            
            if len(q)>1:
                length=len(q)
                qq=q[length-1].split('#')
                if len(qq)>1:
                    label_list1.append(qq[1].strip())
                else:
                    label_list1.append(qq[0].strip())
            else:
                qq=d.split('#')
                if len(qq)>1:
                    label_list1.append(qq[1].strip())
                else:
                    label_list1.append(d.strip())
            j=j+1
        label_list1.append(" ")  

for i in range(len(label_list1)):
    d= label_list1[i].split("*")
    if len(d)>1:
        label_list1[i]=d[1]

vocab=[]
for i in range(len(class_list)):
    l=[]
    for j in range(len(label_list1)):
        if class_list[i]==label_list1[j]:
            l.append(label_list1[j+1])
    vocab.append(l)       
        
for i in range(len(vocab)):
    for j in range(len(vocab[i])):
        b= vocab[i][j].split('_')
        if len(b)>1:
            bb= b[0]+' '+b[1]
            vocab[i][j]=bb
        vocab[i][j]=vocab[i][j].lower()
    
    for j in range(len(vocab[i])):
        if len(vocab[i][j])>60:    
            vocab[i][j]=' '

vocab_new=[]
class_list_new=[]
for i in range(len(vocab)):
    if len(vocab[i])>1:
        vocab_new.append(vocab[i])
        class_list_new.append(class_list[i])

for i in range(len(vocab_new)):
    for j in range(len(vocab_new[i])):
        if vocab_new[i][j]=='':
            vocab_new[i][j]=' '
    if ' ' in vocab_new[i]:
        vocab_new[i].remove(' ')
    if ' ' in vocab_new[i]:
        vocab_new[i].remove(' ')
    if ' ' in vocab_new[i]:
        vocab_new[i].remove(' ')
    if ' ' in vocab_new[i]:
        vocab_new[i].remove(' ')
    if ' ' in vocab_new[i]:
        vocab_new[i].remove(' ')

Subclass_list1=[]   
for i in range(len(data)):
    if data['text'][i]=='SubClassOf':
        j=i+1
        while data['text'][j] !='/SubClassOf':
            d = data['text'][j].split()
            if 'Class' in d:
                p = re.findall(r'"([^"]*)"', data['text'][j])
                
                q = p[0].split('/')
                if len(q)>1:
                    length=len(q)
                    qq=q[length-1].split('#')
                    if len(qq)>1:
                        Subclass_list1.append(qq[1])
                    else:
                        Subclass_list1.append(qq[0])
                else:
                    qq=q[0].split('#')
                    if len(qq)>1:
                        Subclass_list1.append(qq[1])
                    else:
                        Subclass_list1.append(p[0])
            j=j+1
        Subclass_list1.append(" ")                  

#### Identify class linkage
l_new=[]
i=0
while i!=(len(Subclass_list1)-3):
    l=[]
    j=i
    l.append(Subclass_list1[j+1])
    l.append(Subclass_list1[j])
    l_new.append(l)
    i=i+3

flag=[]
for i in range(len(l_new)):
    flag.append(0)

l_n=[]
for i in range(len(l_new)):
    l=[]
    if flag[i]!=1:
        for j in l_new[i]:
            ls=[]
            k=i+1
            for k in range(len(l_new)):
                if j in l_new[k] and flag[k]!=1:
                    ls.append(l_new[i])
                    ls.append(l_new[k])
                    flag[k]=1
            l.append(ls)
        l_n.append(l)   
        
lp=[]
for i in range(len(l_n)):
    ll=[]
    for j in range(len(l_n[i])):
        for k in range(len(l_n[i][j])):
            for l in range(len(l_n[i][j][k])):
                ll.append(l_n[i][j][k][l]) 
    lp.append(ll)

for i in range(len(lp)):
    if len(lp)>0:
        lp[i]=list(set(lp[i]))

pi=[]
for i in range(len(lp)):
    if len(lp[i])>0:
        pi.append(list(set(lp[i])))

flag=[]
for i in range(len(pi)):
    flag.append(0)

ll_n=[]
for i in range(len(pi)):
    l=[]
    if flag[i]!=1:
        for j in pi[i]:
            ls=[]
            k=i+1
            for k in range(len(pi)):
                if j in pi[k] and flag[k]!=1:
                    ls.append(pi[i])
                    ls.append(pi[k])
                    flag[k]=1
            l.append(ls)
        ll_n.append(l)   

lpp=[]
for i in range(len(ll_n)):
    ll=[]
    for j in range(len(ll_n[i])):
        for k in range(len(ll_n[i][j])):
            for l in range(len(ll_n[i][j][k])):
                ll.append(ll_n[i][j][k][l]) 
    lpp.append(list(set(ll)))
    
flag=[]
for i in range(len(lpp)):
    flag.append(0)

n_l=[]
for i in range(len(lpp)):
    l=[]
    if flag[i]!=1:
        for j in lpp[i]:
            ls=[]
            k=i+1
            for k in range(len(lpp)):
                if j in lpp[k] and flag[k]!=1:
                    ls.append(lpp[i])
                    ls.append(lpp[k])
                    flag[k]=1
            l.append(ls)
        n_l.append(l)   

lpp_n=[]
for i in range(len(n_l)):
    ll=[]
    for j in range(len(n_l[i])):
        for k in range(len(n_l[i][j])):
            for l in range(len(n_l[i][j][k])):
                ll.append(n_l[i][j][k][l]) 
    lpp_n.append(list(set(ll)))
    
    
theme_list_ontology=['Age_Group','Event','Facility','Hazard_Type','HazardPhase','Impact','Involved',
                     'Modality_of_data','Place','Report','Service','Status']

r=[]
for i in range(len(theme_list_ontology)):
    ra=[]
    for j in range(len(lpp_n)):
        if theme_list_ontology[i] in lpp_n[j]:
            ra.append(j)
    r.append(ra)    
    
subthemelist=[]
for i in range(len(theme_list_ontology)):
    subthemelist.append(lpp_n[r[i][0]])

####################
ontology_theme_new=['Event','Impact','Service']
Subtheme_list=[]
for i in range(len(ontology_theme_new)):
    for j in range(len(theme_list_ontology)):
        if ontology_theme_new[i]==theme_list_ontology[j]:
            Subtheme_list.append(subthemelist[j])

s_list=[]
for i in range(len(Subtheme_list)):
    for j in range(len(Subtheme_list[i])):
        s_list.append(Subtheme_list[i][j])

vocab_=[]
for i in range(len(s_list)):
    l=[]
    for j in range(len(class_list_new)):
        if s_list[i]==class_list_new[j]:
            l=vocab_new[j]
    vocab_.append(l)

vocab_4=[]
class_list_4=[]
for i in range(len(vocab_)):
    if len(vocab_[i])>0:
        vocab_4.append(vocab_[i])
        class_list_4.append(s_list[i])   
 
v=[]                 # like unigram
for i in range(len(vocab_4)):
    l=[]
    for j in range(len(vocab_4[i])):
        piy=vocab_4[i][j].split(' ')
        for k in piy:
            l.append(k)
    p=list(set(l))                  #######chose unique by using set
    v.append(p)
    
vo=[]
for i in range(len(v)):
    l=[]
    for j in range(len(v[i])):
        p=v[i][j].split('.')
        if len(p)==2:
            l.append(p[0])
        else:
            l.append(v[i][j])
    ll=[]
    for j in range(len(l)):
        p=l[j].split(',')
        if len(p)==2:
            ll.append(p[0])
        else:
            ll.append(l[j])    
    vo.append(ll)    
    
vocab_4_new=[]
for i in range(len(vo)):
    new_text=""
    for word in vo[i]:
        if len(word)>3:
            new_text += word +' ' 
    vocab_4_new.append(new_text.split())     

##################################################new_filter unnecery classes

df_s=pd.read_csv('Cat_list_for_removal.csv')
cat_list = df_s['subtheme'].tolist()

c_list=[]
v_list=[]
for i in range(len(class_list_4)):
    if class_list_4[i] not in cat_list:
        c_list.append(class_list_4[i])
        v_list.append(vocab_4_new[i])

class_list_4 = list(c_list)
vocab_4_new  = list(v_list)

from nltk.stem import PorterStemmer
ps = PorterStemmer() 

###############
###########
########
#df_sub=pd.DataFrame({'subtheme':class_list_4, 'Vocab':vocab_4_new})
#df_sub.to_csv('Cat_with_Vocab.csv', index=False)
df_s=pd.read_csv('Cat_with_Vocab_updated_final.csv')

for i in range(len(df_s)):
    for j in range(len(class_list_4)):
        if df_s['subtheme'][i]==class_list_4[j]:
            vocab_4_new[j]=df_s['Vocab'][i].split(', ')
########
###########
###############

stem_vocab=[]
for i in range(len(vocab_4_new)):
    l=[]
    for j in range(len(vocab_4_new[i])):
        l.append(ps.stem(vocab_4_new[i][j]))
    stem_vocab.append(l)
    
words = ['people', 'city', 'close', 'line','last', 'number', 'about',
         'pressure', 'disturbance', 'please','change','office', 
         'torrential', 'disturbance', 'examine', 'parts', 'poor', 'sick', 'depression', 
         'operation', 'army', 'caught', 'check', 'give', 'thousand', 'get',
         'serious','total', 'flood', 'floods']

for i in range(len(words)):
    words[i]= ps.stem(words[i])   
    
for i in range(len(stem_vocab)):
    l=[]
    for j in range(len(stem_vocab[i])):
        if stem_vocab[i][j] in words:
            continue
        else:
            l.append(stem_vocab[i][j])    
    stem_vocab[i]=list(set(l))    

############# #######
   
stem_tweet=[]
for i in range(len(combined_list_new)):
    l=[]
    for j in range(len(combined_list_new[i])):
        l.append(ps.stem(combined_list_new[i][j]))
    stem_tweet.append(list(set(l)))


def overlap_count(list1, list2):
    count=0
    for i in list1:
        if i in list2:
            count=count+1
    return count

def overlap_list(list1, list2):
    l=[]
    for i in list1:
        if i in list2:
            l.append(i)
    return l

#### break into unique unigrams 
###  Then calculate score(number of overlap words) and themes 
score1=[]                 # class index
score1_value=[]           # Number of overlping words
Score1_vocab=[]           # Overlping words list
theme_list1=[]            # assigned themes or classes
for i in range(len(stem_tweet)):
    l=[]
    d=[]
    v=[]
    t=[]
    for j in range(len(stem_vocab)):
        if len(set(stem_tweet[i]) & set(stem_vocab[j]))>0:
            l.append(j)
            d.append(overlap_count(stem_tweet[i], stem_vocab[j]))
            v.append(overlap_list(stem_tweet[i], stem_vocab[j]))
            t.append(class_list_4[j])
    score1.append(l)
    score1_value.append(d)
    Score1_vocab.append(v)
    theme_list1.append(t)


n_vocab=[]
summ=0
for i in range(len(stem_vocab)):
    n_vocab.append(len(stem_vocab[i]))
    summ = summ + len(stem_vocab[i])

norm_vocab=[]
l=0
for i in range(len(n_vocab)):
    l=(n_vocab[i]/summ)
    norm_vocab.append(l)

### Frist score method
### W*Q
## W = score (similarity score between tweet and concept)
## Q = concept weight (Number of words of a concept)
tweet_label1=[]              # label of a tweet based on WQ score 
Semet_score1=[]             # WQ score list 
score_list=[]
for i in range(len(score1_value)):
    l=[]
    t=[]
    maxs=0
    ind=0
    for j in range(len(score1_value[i])):
        W = score1_value[i][j]
        WQ=W
        l.append(WQ)
        if maxs<=WQ:
            maxs=WQ
            ind=j
    Semet_score1.append(l)
    if len(score1_value[i])>0:
        tweet_label1.append(theme_list1[i][ind])
        score_list.append(maxs)
    else:
        tweet_label1.append(t)
        score_list.append(maxs)
        
###########
########### uppper code same ######################################

# arrange in decending order or score
for i in range(len(Semet_score1)):
    dframe=pd.DataFrame({'text':theme_list1[i], 'score':Semet_score1[i], 'overlap_word':Score1_vocab[i]})
    dframe=dframe.sort_values('score', ascending=False)
    dframe.reset_index(drop=True,inplace=True)
    theme_list1[i]=list(dframe['text'])
    Semet_score1[i]=list(dframe['score'])
    Score1_vocab[i]=list(dframe['overlap_word'])
    

dataset=[]
for i in range(len(data1)):
    p=data1['text'][i].split('\n')
    dataset.append(p[0])

for i in range(len(tweet_label1)):
    if len(tweet_label1[i])==0:
        tweet_label1[i]='0'
        
tweet_label2=[]
for i in range(len(tweet_label1)):
    tweet_label2.append([tweet_label1[i]])

df1= pd.DataFrame({'tweet':dataset, 'subtheme':tweet_label2, 'subtheme_list':theme_list1, 'score_list':Semet_score1, 'score': score_list ,'overlap_word':Score1_vocab})
df1 = df1.sort_values('subtheme')
df1.reset_index(drop=True,inplace=True)

# ############llower code same

ind=0
unlabaled_tweet=[]
for i in range(len(df1)):
    if df1['subtheme'][i][0]=='0':              
        unlabaled_tweet.append(df1['tweet'][i])
        ind=i

df_unlabelled= pd.DataFrame({'text':unlabaled_tweet})
df_labelled = df1.iloc[ind+1 : len(df1)]
df_labelled.reset_index(drop=True,inplace=True)
Dataframe = df1.iloc[ind+1 : len(df1)]               # just copy for next code
Dataframe.reset_index(drop=True,inplace=True)

   
subtheme_list = pd.read_csv('Category_list_w.r.t._number_updated.csv')
subtheme_list = subtheme_list.sort_values('index')
subtheme_list.reset_index(drop=True,inplace=True)

cat_list = subtheme_list['theme'].tolist()
l=[]
for i in range(len(Dataframe)):
    if Dataframe['subtheme'][i][0] not in cat_list:
        l.append(i)

ll=[]
for i in range(len(Dataframe)):
    ls=[]
    for j in range(len(Dataframe['subtheme_list'][i])):
        for k in range(len(subtheme_list)):
            if Dataframe['subtheme_list'][i][j]==subtheme_list['theme'][k]:
                ls.append(subtheme_list['index'][k])
    ll.append(ls)                
Dataframe['subtheme_num_list'] = ll                

##### adding a column to Dataframe catnum
them_num=[]
for i in range(len(Dataframe)):
    flag=0
    for j in range(len(subtheme_list)):
        if Dataframe['subtheme'][i][0]==subtheme_list['theme'][j]:
            them_num.append(subtheme_list['index'][j])
            flag=1
    if flag==0:
        them_num.append(27)
    
Dataframe['subtheme_num'] = them_num

##### Matching 
data = pd.read_csv('LA_airport_shootings_input_data.csv')
Manual_file = pd.read_csv('4th_annotation_file_LAshoot_updated.csv')

for i in range(len(Manual_file)):
    if Manual_file['Combined'][i]==2:
        Manual_file['Combined'][i]=24
    if Manual_file['Combined'][i]==12:
        Manual_file['Combined'][i]=24
    if Manual_file['Combined'][i]==19:
        Manual_file['Combined'][i]=24
    if Manual_file['Combined'][i]==20:
        Manual_file['Combined'][i]=24
    if Manual_file['Combined'][i]==23:
        Manual_file['Combined'][i]=24
    if Manual_file['Combined'][i]==6:
        Manual_file['Combined'][i]=26

M_annotation=[]
lli=[]
for i in range(len(Dataframe)):
    flag=0
    for j in range(len(Manual_file)):
        if Dataframe['tweet'][i]==data['text'][j]:
            M_annotation.append(Manual_file['Combined'][j])
            flag=1
            break
    if flag==0:
        lli.append(i)
Dataframe['Annotator_cat'] = M_annotation


#### finding percentage  
count=0
index=[]
t=[]
su=[]
sc=[]
snl=[]
App_cat=[]
A_cat=[]
o_w=[]
for i in range(len(Dataframe)):
    if Dataframe['subtheme_num'][i]!=Dataframe['Annotator_cat'][i]:
        count=count+1
        index.append(i)
        t.append(Dataframe['tweet'][i])
        su.append(Dataframe['subtheme_list'][i])
        sc.append(Dataframe['score_list'][i])
        snl.append(Dataframe['subtheme_num_list'][i])
        App_cat.append(Dataframe['subtheme_num'][i])
        A_cat.append(Dataframe['Annotator_cat'][i])
        o_w.append(Dataframe['overlap_word'][i])
        
df_mismatch=pd.DataFrame({'tweet':t, 'subtheme_list':su, 'score_list':sc, 'overlap_word':o_w,'subtheme_num_list':snl, 'subtheme_num':App_cat, 'Annotator_cat':A_cat})
df_mismatch = df_mismatch.sort_values('Annotator_cat')
df_mismatch.reset_index(drop=True,inplace=True)
#df_mismatch.to_csv('Mismatch_tweet_list.csv', index=False)


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print('combined F1 Score: %.4f' % f1_score(them_num, M_annotation, average='weighted'))

pp = pd.read_csv('4th_annotation_file_LAshoot_updated.csv')

count=0
for i in range(len(pp)):
    if pp['Combined'][i]==27:
        count=count+1

mis=[]
lli=[]
for i in range(len(df_unlabelled)):
    flag=0
    for j in range(len(Manual_file)):
        if df_unlabelled['text'][i]==data['text'][j]:
            mis.append(Manual_file['Combined'][j])
            flag=1
    if flag==0:
        lli.append(i)

df_unlabelled['Annotator_cat'] = mis
df_unlabelled = df_unlabelled.sort_values('Annotator_cat')
df_unlabelled.reset_index(drop=True,inplace=True)


