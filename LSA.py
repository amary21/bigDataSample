#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:52:20 2020

@author: amary
@Email: taufik.amary@gmail.com
"""

# # contoh teks
# teks = ['Human machine interface for ABC computer applications',
#         'A survey of user opinion of computer system response time',
#         'The EPS user interface management system',
#         'System and human system enginering testing of EPS',
#         'Relation of user perceived response time to error measurement',
#         'The generation of random, binary, ordered trees',
#         'The intersection graph of paths in trees',
#         'Graph minors IV: widths of trees and well-quasi-ordering',
#         'Graph minors: a survey']

isiDetik = open("Detik.com.txt", "r")
isiCnn = open("cnn.txt", "r")
isiMerdeka = open("merdeka.txt", "r")
isiLiputan = open("liputan6.txt", "r")
isiKompas = open("kompas.txt", "r")

teksDetik = isiDetik.readlines()
teksCnn = isiCnn.readlines()
teksMerdeka = isiMerdeka.readlines()
teksLiputan = isiLiputan.readlines()
teksKompas = isiKompas.readlines()

teks = teksDetik + teksCnn + teksMerdeka + teksLiputan + teksKompas
print(teks)

# melakukan praproses
import re

cleanteks = []
for kalimat in teks:
    # mengubah ke lowercase
    kalimat = kalimat.lower()
    # menghapus tanda koma dan ':'
    kalimat = kalimat.replace('"', '')
    kalimat = re.sub("[.,:\n-]","", kalimat)
    # memisah kalimat menjadi kata-kata
    cleanteks.append(kalimat.split())
    
# tampilkan hasil praproses
print('cleanteks :\n',cleanteks)

# membuat kamus kumpulan data
from gensim import corpora

dictioanary = corpora.Dictionary(cleanteks)

# melihat hasil dictionary
print('\ndictioanary :\n',dictioanary)

# mengubah teks ke dalam bentuk document-term matrix
dtm = [dictioanary.doc2bow(text) for text in cleanteks]

# melihat hasil document-term matrix
print('\ndocument-term matrix :\n',dtm)

# melakukan proses topic extraction menggunakan LSI model (topic = 2)
from gensim.models.lsimodel import LsiModel
lsimodel = LsiModel(dtm, num_topics=5, id2word=dictioanary)

# melihat hasil topic
print('\nhasil topic :\n',lsimodel.show_topics(num_words=3))