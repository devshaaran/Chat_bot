from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
import random as rand

path = 'om.tsv'

'''
data = pd.read_table(df, header=['die', 'sym'])
#var = data.drop(data.index[0])
lemmatizer = WordNetLemmatizer()

'''

medic = [{'die': 'good to hear that', 'sym': 'yeah iam fine'},
         {'die': 'Iam bloo', 'sym': 'who are you'},
         {'die': 'eczema', 'sym': 'rashes dryness flakiness bumps fissures peeling redness,itching'},
         {'die': 'ulcer', 'sym': 'belching, heartburn, indigestion, nausea, passing excessive amounts of gas, vomiting,fatigue, feeling full sooner than normal,loss of appetite,abdominal discomfort'},
         {'die': 'anemia', 'sym': 'dizzy, fatigue, light-headedness, malaise, weak,fast heart rate, palpitations,brittle nails, headache, pallor,shortness of breath'},
         {'die': 'glucoma', 'sym': ' blurred vision, distorted vision, vision loss'},
         {'die': 'conjunctivitis', 'sym': 'eyes pain,red eyes, irritation in the eyes, redness of eyelid, dryness, itchiness, puffy eyes, swollen lining of the eye,watery eyescongestion, runny nose,sneezing,sensitivity to light'},
         {'die': 'diarrhea', 'sym': 'Loose, watery stools,Abdominal cramps,Abdominal pain,Fever,Blood in the stool,Bloating,Nausea,Urgent need to have a bowel movement,Urgent need to have to use the toilet, repeated use of the toilet'}]
df = pd.DataFrame(medic)
#y_params = {'Hello':0, 'obama':1 ,'eczema':2,'ulcer':3,'anemia':4,'glucoma':5,'conjunctivitis':6,'diarrhea':7}

#names = {'hello':0, 'yeah iam fine, what about you ?':1 ,'eczema':2,'ulcer':3,'anemia':4,'glucoma':5,'conjunctivitis':6,'diarrhea':7}
acc={}
n=0
k=0
for i in medic:
    for key,value in i.items():
        if k==0:
            acc[value]=n//2
        n +=1
        k +=1
    #n=0
    k=0


y_params = acc
df['label_num'] = df.die.map(y_params)

X = df.sym
y = df.label_num

#print(data.sym.shape)
#print(data.sym.head())f

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

simple_train = ['headache','bleeding','pain','stomach ache','inflammation','bulging']
con=0
poss=[]

'''
felt_person = input('how do u feel? ')
words_file = word_tokenize(felt_person)
main_words = []
proc_word = []
'''

exclaim = ['can I have a little more description? ','anymore? ']


while True:

    if con<10:

        print("BOT: Hi,how are you?")
        felt_person = input("YOU: ")

    else:

        felt_person = input('YOU: ')

    words_file = word_tokenize(felt_person)
    print(words_file)
    main_words = []
    proc_word = []
    print(main_words)
    print(proc_word)

    for i in words_file:

        stop_words = list(stopwords.words('english'))
        stop_words.extend([',','!','?',':'])
        eliminator = ['how','are','you']
        for elim in eliminator:
            stop_words.remove(elim)
        if i in stop_words:
            pass
        elif i in main_words:
            pass
        else:
            main_words.append(i)


    main_words_string = ' '.join(main_words)
    proc_word.append(main_words_string)
    print(main_words_string)
    vect = CountVectorizer()
    vect.fit(X)

    simple_train_dtm = vect.transform(proc_word)
    X_train = vect.transform(X)
    densed_train = simple_train_dtm.toarray()
    print(pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names()))
    nb.fit(X_train, y)
    pred_model = nb.predict(simple_train_dtm)
    print(pred_model)
    con = con + 10
    for key, value in y_params.items():
        if value == pred_model[0]:
            print(key)




