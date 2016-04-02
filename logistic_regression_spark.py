# Kode dibawah ini kita menggunakan spark untuk menerapkan
# konsep yang kita dapat.
# Materi pendukung ada di slide

# jalankan pyspark lalu masukkan kode di bawah ini
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD

htf = HashingTF(10000)

# komentar spam
# NOTE: pastikan path file sama dengan di mesinmu
classNeg1Data = sc.textFile("/home/vagrant/go/src/github.com/pyk/ai-mentoring-surabaya/class_neg_1.data")
classNeg1DataLabeled = classNeg1Data.map(lambda text : LabeledPoint(1, htf.transform(text.split(" "))))
print classNeg1DataLabeled.count()
classNeg1DataLabeled.persist()

# komentar netral
# NOTE: pastikan path file sama dengan di mesinmu
class0Data = sc.textFile("/home/vagrant/go/src/github.com/pyk/ai-mentoring-surabaya/class_0.data")
class0DataLabeled = class0Data.map(lambda text : LabeledPoint(1, htf.transform(text.split(" "))))
print class0DataLabeled.count()
class0DataLabeled.persist()

# komentar berisi harapan, pembelaan, usulan
# NOTE: pastikan path file sama dengan di mesinmu
class1Data = sc.textFile("/home/vagrant/go/src/github.com/pyk/ai-mentoring-surabaya/class_1.data")
class1DataLabeled = class1Data.map(lambda text : LabeledPoint(1, htf.transform(text.split(" "))))
print class1DataLabeled.count()
class1DataLabeled.persist()

# komentar meminta kejelasan, aduan
# NOTE: pastikan path file sama dengan di mesinmu
class2Data = sc.textFile("/home/vagrant/go/src/github.com/pyk/ai-mentoring-surabaya/class_2.data")
class2DataLabeled = class2Data.map(lambda text : LabeledPoint(1, htf.transform(text.split(" "))))
print class2DataLabeled.count()
class2DataLabeled.persist()

# kita split datanya
trainNeg1, testNeg1 = classNeg1DataLabeled.randomSplit([0.8, 0.2])
train0, test0 = class0DataLabeled.randomSplit([0.8, 0.2])
train1, test1 = class1DataLabeled.randomSplit([0.8, 0.2])
train2, test2 = class2DataLabeled.randomSplit([0.8, 0.2])

# kita gabung train dan test jadi satu
trainh = sc.union([trainNeg1, train0, train1, train2])
testh = sc.union([testNeg1, test0, test1, test2])

# kita gunakan logistic regression untuk membuat model
model = LogisticRegressionWithSGD.train(trainh)
prediction_and_labels = testh.map(lambda point: (model.predict(point.features), point.label))
correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)

accuracy = correct.count() / float(testh.count())

print "Akurasi model " + str(accuracy * 100) + "%"
