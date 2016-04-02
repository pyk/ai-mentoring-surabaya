import csv
import collections
import re
import math
from sklearn import linear_model

# read csv file
# download data di https://data-engineers-id.slack.com/files/bay/F0XCYMAF5/data_3k_comments.csv
csvfile = open('data_3k_comments.csv', 'r')
reader = csv.DictReader(csvfile, delimiter=',')

# stopwords dictionary
sw_dictionary = collections.OrderedDict()
dictionary = collections.OrderedDict()

# create stopwords dictionary
stopwords_file = open('stopwords_id.txt')
for line in stopwords_file:
    # skip comment
    if line[0] is '#':
        continue
    netralized = re.sub('\s+', '', line)
    sw_dictionary[netralized] = len(sw_dictionary) + 1

docs = []
labels = []
# create a dictionary
for row in reader:
    # split by whitespace
    netralized = re.sub('[^a-zA-Z0-9\s]+', ' ', row['message'])
    netralized = netralized.strip()
    netralized = re.sub('\s+', ' ', netralized)
    terms = netralized.split(' ')
    doc = collections.Counter()
    for term in terms:
        term = term.lower()
        # skip stopwords
        if term in sw_dictionary:
            continue
        if term in dictionary:
            dictionary[term] = dictionary[term]
        else:
            dictionary[term] = {'index': len(dictionary) + 1, 'doc_num': 0}
        doc[term] += 1
    labels.append(int(row['class']))
    docs.append(doc)

print len(dictionary)
total_docs = len(docs)
# input term => number of doc
# count idf
for term in dictionary:
    for doc in docs:
        if term in doc:
            dictionary[term]['doc_num'] += 1

courpus_vec = []
for doc in docs:
    doc_vec = []
    total_freq = sum(doc.values())
    for term in dictionary:
        if term in doc:
            freq = doc[term]
            tf = float(freq)/total_freq
            idf = math.log(float(total_docs)/dictionary[term]['doc_num'])
            tfidf = tf * idf
            doc_vec.append(tfidf)
        else:
            doc_vec.append(0)
    courpus_vec.append(doc_vec)

#print courpus_vec
#print labels
train_total = int(round(total_docs*float(3.0/4.0)))
train_input = courpus_vec[:train_total]
train_output = labels[:train_total]
test_input = courpus_vec[(train_total-total_docs):]
test_output = labels[(train_total-total_docs):]

logreg = linear_model.LogisticRegression()
logreg.fit(train_input, train_output)

# data baru
x_baru = test_input[0]
y_baru = test_output[0]
prediksi = logreg.predict([x_baru])

print 'hasil penentuan class dari model: ' + str(prediksi[0]) + ' class sebenarnya: ' + str(y_baru)
# initialiasi => model
# /v1/classification-comment?data= ...
# 











