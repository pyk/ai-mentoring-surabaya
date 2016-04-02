import collections
import re
import math

corpus = [
    "Mantaaap....Pak Presiden..habisi para pencuri ikan diwilayah kita......jangan kasi ampun.....sanksi keras akan membuat mereka jera!",
    "Insya alloh indonesia akan di sgani dan menjadi macan asia.. Kalau pemimpin ny sprti bapa presiden kita skarang. Lanjutkan pa kami alloh slalu brsma mu.. Amiin"
]

docs = []
# buat dictionary & count term freq
dictionary = collections.OrderedDict()
for doc in corpus:
    netralized = re.sub('[^a-zA-Z0-9\s]+', ' ', doc)
    netralized = netralized.strip()
    netralized = re.sub('\s+', ' ', netralized)
    terms = netralized.split(' ')
    d = collections.Counter()
    for term in terms:
        term = term.lower()
        if term in dictionary:
            dictionary[term] = dictionary[term]
        else:
            dictionary[term] = {'index': len(dictionary) + 1, 'doc_num': 0}
        d[term] += 1
    docs.append(d)

print len(dictionary)
total_docs = len(docs)

# count idf
for term in dictionary:
    for doc in docs:
        if term in doc:
            dictionary[term]['doc_num'] += 1

# tugas untuk peserta:
# buat representasi vektornya (materi ada di slide)
# yang di ketahui total_docs, 
# TF(D,t): freq t/total_freq di dokumen D
# IDF(t): log(total_docs/total dokumen dimana ada term t)
# TFIDF = TF * IDF

# jawaban tugas
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

print courpus_vec
