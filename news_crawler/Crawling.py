#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 크롤러를 만들기 전 필요한 도구들을 임포트합니다.
import requests
import pandas as pd
from bs4 import BeautifulSoup

# 페이지 수, 카테고리, 날짜를 입력값으로 받습니다.
def make_urllist(page_num, code, date): 
  urllist= []
  for i in range(1, page_num + 1):
    url = 'https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1='+str(code)+'&date='+str(date)+'&page='+str(i)   
    news = requests.get(url)
    

    # BeautifulSoup의 인스턴스 생성합니다. 파서는 html.parser를 사용합니다.
    soup = BeautifulSoup(news.content, 'html.parser')

    # CASE 1
    news_list = soup.select('.newsflash_body .type06_headline li dl')
    
    # CASE 2
    news_list.extend(soup.select('.newsflash_body .type06 li dl'))
        
    # 각 뉴스로부터 a 태그인 <a href ='주소'> 에서 '주소'만을 가져옵니다.
    for line in news_list:
        urllist.append(line.a.get('href'))
  return urllist

url_list = make_urllist(2, 101, 20200506)
print('뉴스 기사의 개수: ',len(url_list))
url_list[:5]


# In[2]:


idx2word = {'101' : '경제', '102' : '사회', '103' : '생활/문화', '105' : 'IT/과학'}


# In[3]:


from newspaper import Article

#- 데이터프레임을 생성하는 함수입니다.
def make_data(urllist, code):
  text_list = []
  for url in urllist:
    article = Article(url, language='ko')
    article.download()
    article.parse()
    text_list.append(article.text)
    #print(text_list)


  #- 데이터프레임의 'news' 키 아래 파싱한 텍스트를 밸류로 붙여줍니다.
  df = pd.DataFrame({'news': text_list})

  #- 데이터프레임의 'code' 키 아래 한글 카테고리명을 붙여줍니다.
  df['code'] = idx2word[str(code)]
  print(str(code))
  return df
 

data = make_data(url_list, 101)
#- 상위 10개만 출력해봅니다.
data[:10]


# In[4]:


code_list = [102, 103, 105]

code_list


# In[5]:


def make_total_data(page_num, code_list, date):
  df = None

  for code in code_list:
    url_list = make_urllist(page_num, code, date)
    df_temp = make_data(url_list, code)
    print(str(code)+'번 코드에 대한 데이터를 만들었습니다.')

    if df is not None:
      df = pd.concat([df, df_temp])
    else:
      df = df_temp

  return df


# In[6]:


df = make_total_data(1, code_list, 20200506)


# In[7]:


print('뉴스 기사의 개수: ',len(df))


# In[8]:


def make_total_data(page_num, code_list, date):
  df = None

  for code in code_list:
    url_list = make_urllist(page_num, code, date)
    df_temp = make_data(url_list, code)
    print(str(code)+'번 코드에 대한 데이터를 만들었습니다.')

    df = pd.concat([df, df_temp])

  return df


# In[9]:


df = make_total_data(1, code_list, 20200506)


# In[ ]:


print('뉴스 기사의 개수: ',len(df))

df.sample(10)


# In[ ]:


# 네이버 크롤링 (오래걸림 주의)

#df = make_total_data(3, code_list, 20200506)


# In[ ]:


#import os

# 데이터프레임 파일을 csv 파일로 저장합니다.
#csv_path = os.getenv("HOME") + "/aiffel/news_crawler/news_data.csv"
#df.to_csv(csv_path, index=False)

#if os.path.exists(csv_path):
#  print('{} File Saved!'.format(csv_path))


# In[10]:


import os

csv_path = os.getenv("HOME") + "/aiffel/news_crawler/news_data.csv"
df1 = pd.read_table(csv_path, sep=',')

csv_path = os.getenv("HOME") + "/aiffel/news_crawler/news_data2.csv"
df2 = pd.read_table(csv_path, sep=',')

df=df1.append(df2)

print('뉴스 기사의 개수: ',len(df))
df.sample(10)


# In[11]:


# 정규 표현식을 이용해서 한글 외의 문자는 전부 제거합니다.
df['news'] = df['news'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
df['news']


# In[12]:


print(df.isnull().sum())


# In[13]:


# 중복 샘플 제거 전
df.drop_duplicates(subset=['news'], inplace=False)

print('뉴스 기사의 개수: ',len(df))


# In[14]:


# 중복 샘플 제거 후
df.drop_duplicates(subset=['news'], inplace=True)

print('뉴스 기사의 개수: ',len(df))


# In[15]:


df['code'].value_counts().plot(kind = 'bar')


# In[16]:


print(df.groupby('code').size().reset_index(name = 'count'))


# In[17]:


from konlpy.tag import Komoran
tokenizer = Komoran()

kor_text = '밤에 귀가하던 여성에게 범죄를 시도한 대 남성이 구속됐다서울 제주경찰서는             상해 혐의로 씨를 구속해 수사하고 있다고 일 밝혔다씨는 지난달 일 피해 여성을             인근 지하철 역에서부터 따라가 폭행을 시도하려다가 도망간 혐의를 받는다피해             여성이 저항하자 놀란 씨는 도망갔으며 신고를 받고 주변을 수색하던 경찰에             체포됐다피해 여성은 이 과정에서 경미한 부상을 입은 것으로 전해졌다'

#- 형태소 분석, 즉 토큰화(tokenization)를 합니다.
print(tokenizer.morphs(kor_text))


# In[18]:


stopwords = ['에','ㄴ','며','다','는','은','을','했','에게','있','이','의','하','한','다','과','때문','할','수','무단','따른','및','금지','전재','경향신문','기자','는데','가','등','들','파이낸셜','저작','등','뉴스']


# In[19]:


# 토큰화 및 토큰화 과정에서 불용어를 제거하는 함수입니다.
def preprocessing(data):
  text_data = []

  for sentence in data:
    temp_data = []
    #- 토큰화
    temp_data = tokenizer.morphs(sentence) 
    #- 불용어 제거
    temp_data = [word for word in temp_data if not word in stopwords] 
    text_data.append(temp_data)

  text_data = list(map(' '.join, text_data))

  return text_data


# In[20]:


text_data = preprocessing(df['news'])
print(text_data[0])


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# In[22]:


#- 훈련 데이터와 테스트 데이터를 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(text_data, df['code'], random_state = 0)


# In[23]:


print('훈련용 뉴스 기사의 개수 :', len(X_train))
print('테스트용 뉴스 기사의 개수 : ', len(X_test))
print('훈련용 레이블의 개수 : ', len(y_train))
print('테스트용 레이블의 개수 : ', len(y_test))


# In[24]:


#- 단어의 수를 카운트하는 사이킷런의 카운트벡터라이저입니다.
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

#- 카운트벡터라이저의 결과로부터 TF-IDF 결과를 얻습니다.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#- 나이브 베이즈 분류기를 수행합니다.
#- X_train은 TF-IDF 벡터, y_train은 레이블입니다.
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[25]:


def tfidf_vectorizer(data):
  data_counts = count_vect.transform(data)
  data_tfidf = tfidf_transformer.transform(data_counts)
  return data_tfidf


# In[ ]:


new_sent = preprocessing(["민주당 일각에서 법사위의 체계·자구 심사 기능을 없애야 한다는                            주장이 나오는 데 대해 “체계·자구 심사가 법안 지연의 수단으로                           쓰이는 것은 바람직하지 않다”면서도 “국회를 통과하는 법안 중 위헌                          법률이 1년에 10건 넘게 나온다. 그런데 체계·자구 심사까지 없애면 매우 위험하다”고 반박했다."])
print(clf.predict(tfidf_vectorizer(new_sent)))


# In[ ]:


new_sent = preprocessing(["인도 로맨틱 코미디 영화 <까립까립 싱글>(2017)을 봤을 때 나는 두 눈을 의심했다.                           저 사람이 남자 주인공이라고? 노안에 가까운 이목구비와 기름때로 뭉친 파마머리와,                           대충 툭툭 던지는 말투 등 전혀 로맨틱하지 않은 외모였다. 반감이 일면서                           ‘난 외모지상주의자가 아니다’라고 자부했던 나에 대해 회의가 들었다.                           티브이를 꺼버릴까? 다른 걸 볼까? 그런데, 이상하다. 왜 이렇게 매력 있지? 개구리와                            같이 툭 불거진 눈망울 안에는 어떤 인도 배우에게서도 느끼지 못한                             부드러움과 선량함, 무엇보다 슬픔이 있었다. 2시간 뒤 영화가 끝나고 나는 완전히 이 배우에게 빠졌다"])
print(clf.predict(tfidf_vectorizer(new_sent)))


# In[ ]:


new_sent = preprocessing(["20분기 연속으로 적자에 시달리는 LG전자가 브랜드 이름부터 성능, 디자인까지 대대적인 변화를                           적용한 LG 벨벳은 등장 전부터 온라인 커뮤니티를 뜨겁게 달궜다. 사용자들은 “디자인이 예쁘다”,                           “슬림하다”는 반응을 보이며 LG 벨벳에 대한 기대감을 드러냈다."])
print(clf.predict(tfidf_vectorizer(new_sent)))


# In[ ]:


y_pred = clf.predict(tfidf_vectorizer(X_test))
print(metrics.classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




