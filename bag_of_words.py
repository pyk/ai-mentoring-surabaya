import collections
import re
import pprint

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
            dictionary[term] = {'index': len(dictionary) + 1 }
        d[term] += 1
    docs.append(d)

dictionary_str = ""
for term in dictionary:
    dictionary_str += term + ', '
print dictionary_str

print len(dictionary)

# tugas untuk peserta:
# buat representasi vektornya
# (materi ada di slide)
# jawaban tugas
courpus_vec = []
for doc in docs:
    doc_vec = []
    for term in dictionary:
        if term in doc:
            doc_vec.append(doc[term])
        else:
            doc_vec.append(0)
    courpus_vec.append(doc_vec)

print courpus_vec



