import os
import gensim
import plotly.express as px
from nltk import sent_tokenize, word_tokenize
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

story = []
stop_words = set(stopwords.words('english'))

for filename in os.listdir('data'):
    f = open(os.path.join('data', filename))
    corpus = f.read()
    raw_sent = sent_tokenize(corpus)
    for sent in raw_sent:
        filtered_vocab = [word for word in word_tokenize(sent) if word.lower() not in stop_words]
        filtered_sent = ' '.join(filtered_vocab)
        story.append(simple_preprocess(filtered_sent))  #creates a list of filtered_sentence elements

model = gensim.models.Word2Vec(window=100, min_count=2)      #Vectorization into a 100d space
model.build_vocab(story)    #keeps only unique words

model.train(story, total_examples=model.corpus_count, epochs=model.epochs)      #examples = #sentences  -> Feature Extraction
'''print(model.wv.most_similar('winterfell'))
print(model.wv.doesnt_match(['jon','robb','arya','sansa','bran']))
print(model.wv.similarity('arya','sansa'))'''
print(model.wv.similarity('robb', 'tyrion'))


model.wv.get_normed_vectors()   #creates a 2D matrix of .... x 100
y = model.wv.index_to_key   #returns the words from vector value

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X = pca.fit_transform(model.wv.get_normed_vectors())

fig = px.scatter_3d(X[500:700],x=0,y=1,z=2, color=y[500:700])
fig.show()