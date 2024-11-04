# modified as reported in paper

import numpy
import numpy as np
import nltk
import pandas as pd 
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import emoji
import math 

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
    ,'http//','htt','"','-','ct','g','n','...','ndrrmc']

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

df1 = pd.read_csv('labeled_tweets_pass-1_PakQuake.csv')
df2 = pd.read_csv('labeled_tweets_pass-2_PakQuake.csv')

tweet = df1['tweet'].tolist()
category = df1['subtheme'].tolist()
a = df2['text'].tolist()
b = df2['subtheme'].tolist()

for i in range(len(a)):
    tweet.append(a[i])
    category.append(b[i])

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('crisisNLP_word_vector.bin', binary=True)
from sklearn.metrics.pairwise import cosine_similarity 

df= pd.DataFrame({'text':tweet, 'subtheme':category})

for i in range(len(df)):
    if df['subtheme'][i]=='Death_and_Toll' or df['subtheme'][i]=='trapped_people' or df['subtheme'][i]=='missing_people' or df['subtheme'][i]=='Suicidal_Throughts':
        df['subtheme'][i]='AffectedPopulation'
    elif df['subtheme'][i]=='Severity':
        df['subtheme'][i]='Impact'
    elif df['subtheme'][i]=='Mental_Stress':
        df['subtheme'][i]='Emotional_Distress'
    elif df['subtheme'][i]=='Food' or df['subtheme'][i]=='Rescue_Operation' or df['subtheme'][i]=='Healthcare_Service' or df['subtheme'][i]=='Supply' or df['subtheme'][i]=='Recovery_Plan' or df['subtheme'][i]=='Helpline' or df['subtheme'][i]=='Service':
        df['subtheme'][i]='Volunteer_Support'
    elif df['subtheme'][i]=='Broken_Bridge' or df['subtheme'][i]=='BlockedRoad':
        df['subtheme'][i]='Infrastructure_Damage'
          
df = df.sort_values('subtheme')
df.reset_index(drop=True,inplace=True)

d1=pd.read_csv('Final_ext_vocab_wiki.csv')
d2=pd.read_csv('Cat_with_Vocab_updated_final.csv')
onto_theme= d1['subtheme'].tolist()

for i in range(len(d1)):
    d1['Vocab'][i]=d1['Vocab'][i].split(', ')
for i in range(len(d2)):
    d2['Vocab'][i]=d2['Vocab'][i].split(', ')

terms=[]
for i in range(len(d1)):
    t=[]
    for j in range(len(d1['Vocab'][i])):
        t.append(d1['Vocab'][i][j])
    for j in range(len(d2['Vocab'][i])):
        t.append(d2['Vocab'][i][j])
    t= list(set(t))
    terms.append(t)

vector_terms=[]
for i in range(len(terms)):
    vt=[]
    for j in range(len(terms[i])):
        sectence_score_2 = numpy.zeros(300, dtype=float)
        xy=terms[i][j].split()
        for value in xy:
            if value in model.vocab:
                sectence_score_2 = sectence_score_2 + model.wv[value]
        vt.append(sectence_score_2)
    vector_terms.append(vt)

def cosine_onto_sim(tweet, terms, th):
    s1 = tweet.split()
    pos_tagged = nltk.pos_tag(s1)
    test=[]
    for i in range(len(pos_tagged)):
        if pos_tagged[i][1]=='NN' or pos_tagged[i][1]=='NNS' or pos_tagged[i][1]=='NNP'or pos_tagged[i][1]=='NNPS' or pos_tagged[i][1]=='VB'or pos_tagged[i][1]=='VBN' or pos_tagged[i][1]=='VBZ' or pos_tagged[i][1]=='VBD' or pos_tagged[i][1]=='VBP' or pos_tagged[i][1]=='VBG' or pos_tagged[i][1]=='JJ' or pos_tagged[i][1]=='JJR' or pos_tagged[i][1]=='JJS':
            test.append(pos_tagged[i][0])  
    indx=0
    for i in range(len(onto_theme)):
        if onto_theme[i]==th:
             indx=i
             break   
    scor=[]
    for i in range(len(test)):
        sor=[]
        for j in range(len(terms[indx])):
            sectence_score_1 = numpy.zeros(300, dtype=float)
            if test[i] in model.vocab:
                    sectence_score_1 = sectence_score_1 + model.wv[test[i]]
            pp=[]
            pp.append(sectence_score_1)
            pps=[]
            pps.append(vector_terms[indx][j])
            simm__=cosine_similarity(pp, pps)
            sor.append(simm__[0][0])
        scor.append(max(sor))
    if len(scor)>0:
        return sum(scor)/len(scor)
    else: 
        return 0

tweet=[]
cat=[]
anno=[]
k=0
num_tweet=[]
while k<len(df):
    a_tweet=[]
    indexs=0
    theme=df['subtheme'][k]
    for i in range(len(df)):
        if df['subtheme'][i]==theme:
            a_tweet.append(df['text'][i])
            indexs=indexs+1
    k=k+indexs
    
    co=0
    if len(a_tweet)>25: 
        d=pd.DataFrame({'text':a_tweet})
        d['text'] = d['text'].apply(Preprocessing)
        
        similarity=[]
        for sent in d['text']:
            similarity.append(cosine_onto_sim(sent, vector_terms, theme))
        # Normizing scores 
        aa= sum(similarity)  
        Tweet_score=[]
        for i in range(len(similarity)):
            Tweet_score.append(round(similarity[i]/aa,5))
        
        dataframe=pd.DataFrame({'text':a_tweet, 'score':Tweet_score})
        dataframe = dataframe.sort_values('score', ascending=False)
        dataframe.reset_index(drop=True,inplace=True)
        
        val=math.ceil(len(a_tweet)*0.25)
        
        if val>25:
            #x=np.percentile(Tweet_score, 75) #np.percentile(dataframe['score'], [25, 50, 75], interpolation='midpoint')
            for i in range(val):
                #if dataframe['score'][i]>val:
                tweet.append(dataframe['text'][i])
                cat.append(theme)
                anno.append('')
                co=co+1
            num_tweet.append(co)
        else:
            for i in range(25):
                tweet.append(dataframe['text'][i])
                cat.append(theme)
                anno.append('')
                co=co+1
            num_tweet.append(co)
    else:
        for i in range(len(a_tweet)):
            tweet.append(a_tweet[i])
            cat.append(theme)
            anno.append('')
            co=co+1
        num_tweet.append(co)
            
                
d=pd.DataFrame({'tweet':tweet, 'category':cat, 'summary label':anno})    
d.to_csv('Pakquake_for_Annotator_modified.csv', index=False)    

    