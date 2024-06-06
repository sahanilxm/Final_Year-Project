import pandas as pd
import string
import spacy
import matplotlib.pyplot as plt # type: ignore
from wordcloud import STOPWORDS, WordCloud # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
import seaborn as sns # type: ignore
import joblib

STOPWORDS = STOPWORDS.union({'re' , 's' , 'subject','hpl','hou','enron'})

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv('dataset/spam_ham_dataset.csv')


print('STARTED')
email_subjects=[]
email_text=[]
def split_subject(text):
    global email_subjects
    global email_text
    subject=""
    for ch in text:
        if(ch=='\r'):
            break
        subject+=ch
        
    email_subjects.append(subject)
    email_text.append(text.replace(subject,""))

print(df)

df["text"]=df["text"].str.replace("Subject:","")
df.loc[df["label"]=="ham","label"]=0
df.loc[df["label"]=="spam","label"]=1
df["text"].apply(split_subject)
df["subject"]=email_subjects
df["text"]=email_text

print(df)

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df
df=df.drop(df.columns[0], axis=1)
df=df.drop('label_num', axis=1)
df=swap_columns(df, 'label', 'subject')

print(df);


def clean_text(s): 
    for cs in s:
        if  not cs in string.ascii_letters:
            s = s.replace(cs, ' ')
    return s.rstrip('\r\n')

def remove_little(s): 
    wordsList = s.split()
    k_length=2
    resultList = [element for element in wordsList if len(element) > k_length]
    resultString = ' '.join(resultList)
    return resultString

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

df['text'] = df['text'].apply(lambda x: clean_text(x))
df['text'] = df['text'].apply(lambda x: remove_little(x))
df['text'] = df['text'].apply(lambda x: lemmatize_text(x))

df['subject'] = df['subject'].apply(lambda x: clean_text(x))
df['subject'] = df['subject'].apply(lambda x: remove_little(x))
df['subject'] = df['subject'].apply(lambda x: lemmatize_text(x))


dic_all={}
def count_words(s):
    global dic_all
    wordsList = s.split()
    for w in wordsList:
        if not w in dic_all:
             dic_all[w]=1
        else:
            dic_all[w]+=1

dic_all={}
df['subject'].apply(lambda x: count_words(x))
dic_all=sorted(dic_all.items(), key=lambda x:x[1],reverse=True)
df_new=pd.DataFrame(dic_all)
df_new.head(20)

df['label']=df['label'].astype(str).astype(int)
X_train, X_test , y_train, y_test = train_test_split(df['text'], df['label'] , test_size=0.2)

Vectorizer = CountVectorizer()
count= Vectorizer.fit_transform(X_train.values)

Spam_detection = MultinomialNB()
targets = y_train.values
Spam_detection.fit(count, targets)


MultinomialNB()

y_predict = Spam_detection.predict(Vectorizer.transform(X_test))

print(accuracy_score(y_test, y_predict))


#create a confusion matrix 
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#print the classification report 
print(classification_report(y_test , y_predict))

joblib.dump(Spam_detection, 'models/spam_detection_model.pkl')
joblib.dump(Vectorizer, 'models/count_vectorizer.pkl')

input_email = input("Write the email to check: ")
input_email = lemmatize_text(remove_little(clean_text(input_email)))

prediction = Spam_detection.predict(Vectorizer.transform([input_email]))
if prediction == 1:
    print("Prediction: Spam")
else:
    print("Prediction: Ham")