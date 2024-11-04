import nltk
import pandas as pd 
import re
import math
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import emoji

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

N=['utc','rt','mt','wp','wt','ts','cma','pst','jtwc','rubyph','chuuk','jma','22w','/','w/','f',
    'd','-','--','l','w-n-w','e','ir','m','x','t','b','m','r','1/','h','wx','cc','fr','w','http/','s'
    ,'http//','htt','"','-','ct','g','n','...','ndrrmc']
N1=['please','magnitude','people','lives']

def Preprocessing1(text):
    text= re.sub('http://\S+|https://\S+', '', text)            # remove url
    text= " ".join(filter(lambda x:x[0]!='@', text.split()))    # remove @twitter
    text= give_emoji_free_text(text)                            # remove emotions
    text= re.sub("[\(\[].*?[\)\]]", "", text)                   # remove (..) [..]
    text= re.sub('['+my_punctuation + ']+', '', text)           # remove punctuation
    text= text.lower()                                          # lower case
    text= re.sub('([0-9]+)([a-z]+)\s'," ", text)                # remove string contain digit+char 
    text= re.sub('([a-z]+)([0-9]+)'," ", text)                  # remove string contain char+digit   
    text= ''.join(i for i in text if not i.isdigit()) # remove digits
    text= str(''.join([i if ord(i) < 128 else ' ' for i in text])) # remove ascii code characters
    word_tokens= word_tokenize(text)                        
    text= str(' '.join([w for w in word_tokens if not w in N]))  # remove special words 
    text= str(' '.join([w for w in word_tokens if not w in N1]))  # remove special words 
    text= text.split()
    text= remove_less_then_3(text)                              # remove words with length less than 3
    word_tokens = word_tokenize(text) 
    filtered_text = str(' '.join([w for w in word_tokens if not w in stop_words]))
    return filtered_text

data= pd.read_csv('unlabelled_tweets_for_pass2.csv')
data1= pd.read_csv('unlabelled_tweets_for_pass2.csv')
data['text'] = data['text'].apply(Preprocessing1)

# Extract NVA only 
res=[]
for l in data['text']:
    t = nltk.word_tokenize(l)
    pos_tagged = nltk.pos_tag(t)
    test=[]
    for i in range(len(pos_tagged)):
        if pos_tagged[i][1]=='NN' or pos_tagged[i][1]=='NNS' or pos_tagged[i][1]=='NNP'or pos_tagged[i][1]=='NNPS' or pos_tagged[i][1]=='VB'or pos_tagged[i][1]=='VBN' or pos_tagged[i][1]=='VBZ' or pos_tagged[i][1]=='VBD' or pos_tagged[i][1]=='VBP' or pos_tagged[i][1]=='VBG' or pos_tagged[i][1]=='JJ' or pos_tagged[i][1]=='JJR' or pos_tagged[i][1]=='JJS':
            test.append(pos_tagged[i][0])
    res.append(test) 

# do some extra preprocessing on extracted tweets
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
    lll=[]
    for j in range(len(ll)):
        p=ll[j].split('/')
        if len(p)==2:
            lll.append(p[0])
            lll.append(p[1])
        else:
            lll.append(ll[j])  
    list_new.append(lll)

# Remove words less than 2    
combined_list_new=[]
for i in range(len(list_new)):
    new_text=""
    for word in list_new[i]:
        if len(word)>2:
            new_text += word +' ' 
    combined_list_new.append(list(set(new_text.split())))

# Do stamming
from nltk.stem import PorterStemmer
ps = PorterStemmer() 

stem_tweet=[]
for i in range(len(combined_list_new)):
    l=[]
    for j in range(len(combined_list_new[i])):
        l.append(ps.stem(combined_list_new[i][j]))
    stem_tweet.append(list(set(l)))

######################################################
df_s=pd.read_csv('Final_ext_vocab_wiki.csv')

BOG_theme1=[]
for i in range(len(df_s)):
    df_s['Vocab'][i]=df_s['Vocab'][i].split(', ')
    BOG_theme1.append(df_s['subtheme'][i])

stem_vocab1=[]
for i in range(len(df_s)):
    l=[]
    for j in range(len(df_s['Vocab'][i])):
        l.append(ps.stem(df_s['Vocab'][i][j]))
    stem_vocab1.append(l)

simm_matrix1=[]                 
simm_matrix2=[]                 
for i in range(len(stem_tweet)):
    s=[]
    ss=[]
    simm=0
    for j in range(len(stem_vocab1)):
        if len(set(stem_tweet[i]) & set(stem_vocab1[j]))>0:
            a = len(set(stem_tweet[i]))
            b = len(set(stem_vocab1[j]))
            un= math.sqrt(a*b)
            c = len(set(stem_tweet[i]) & set(stem_vocab1[j]))
            #simm = c/un
            simm = c
            s.append(simm)
            ss.append(list(set(stem_tweet[i]) & set(stem_vocab1[j])))
        else:
            s.append(0)
            ss.append(0)
    simm_matrix1.append(s)
    simm_matrix2.append(ss)
    
tweet_theme1=[]              
simm_score1=[]             
tweet_voacb=[]
for i in range(len(simm_matrix1)):
    word_=[]
    maxs=0
    ind=0
    for j in range(len(simm_matrix1[i])):
        if maxs<simm_matrix1[i][j]:
            maxs=simm_matrix1[i][j]
            word_=simm_matrix2[i][j]
            ind=j
    if maxs!=0:
        simm_score1.append(maxs)
        tweet_theme1.append(BOG_theme1[ind])
        tweet_voacb.append(word_)
    else:
        simm_score1.append(maxs)
        tweet_theme1.append('0')
        tweet_voacb.append(word_)

data1['subtheme']=tweet_theme1
data1['vocab']=tweet_voacb
data1 = data1.sort_values('Annotator_cat')
data1.reset_index(drop=True,inplace=True)



subtheme_list = pd.read_csv('Category_list_w.r.t._number_updated.csv')
subtheme_list = subtheme_list.sort_values('index')
subtheme_list.reset_index(drop=True,inplace=True)

cat_list = subtheme_list['theme'].tolist()
l=[]
for i in range(len(data1)):
    if data1['subtheme'][i] not in cat_list:
        l.append(i)

##### adding a column to Dataframe catnum
them_num=[]
for i in range(len(data1)):
    flag=0
    for j in range(len(subtheme_list)):
        if data1['subtheme'][i]==subtheme_list['theme'][j]:
            them_num.append(subtheme_list['index'][j])
            flag=1
    if flag==0:
        them_num.append(0)
    
data1['subtheme_num'] = them_num
data1 = data1.sort_values('subtheme_num')
data1.reset_index(drop=True,inplace=True)


t=[]
l=[] 
for i in range(len(data1)):
    if data1['subtheme'][i]!='0':
        t.append(data1['text'][i])
        l.append(data1['subtheme'][i])
    
d=pd.DataFrame({'text':t, 'subtheme':l})
d.to_csv('labeled_tweets_pass-2_LAShoot.csv', index=False)
        
tt=[]
ll=[] 
aa=[]
for i in range(len(data1)):
    if data1['subtheme'][i]=='0':
        tt.append(data1['text'][i])
        ll.append(data1['subtheme'][i])
        aa.append(data1['Annotator_cat'][i])
    
dd=pd.DataFrame({'text':tt,'Annotator_cat':aa})
dd = dd.sort_values('Annotator_cat')
dd.reset_index(drop=True,inplace=True)
dd.to_csv('Unlabeled_tweets_after_pass-2_LAShoot.csv', index=False)
    
