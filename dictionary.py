# -*- coding: utf-8 -*-
import sys
import string
import csv
import ipdb
import math
import json
import random
import nltk.translate.bleu_score
import copy
import codecs
import tensorflow as tf
import numpy as np


output_file=codecs.open("data_rezult.csv", "w", "utf-8")
out=codecs.open("dict.txt", "w", "utf-8")
dicttime=codecs.open("dicttime.txt","w", "utf-8")

csvFile = codecs.open("datadata.csv",  "rb", "utf-8") # нужно прочитать оттуда данные!
ads=[]
grades=[]
words=[]
csvFile.readline()

for line in csvFile:
    list = line.split(';')
    str1 = ''
    for c in list[10]:
        if c not in ('(', '!', ',', '?', '.', '\n', '"'):
            if c.isdigit():
                c=" number "
            str1 = str1 + c
    str2 = str1.lower()
    tip=list[4].strip()
    #tip.decode("utf-8")
    if u"поисковая" in tip:
        str2=str2+" found"
    if u"тематическая" in tip:
        str2=str2+" tematic"
    pl=list[5].strip()
    if u"спецразмещение" in pl:
        str2=str2+" special"
    if u"прочее" in pl:
        str2=str2+" other"
    ads.append(str2)
    #print tip
    #print str2


    if len(list[7]) > 4:
        x = 0
    else:
        x = float(list[7])
    if len(list[6]) > 4:
        y = 0
    else:
        y = float(list[6])
    if y == 0:
        r = 0
    else:
        r = (x) / (y)
        round(r, 8)
    grades.append(r)

ads=filter(None, ads)
print np.shape(ads)

for i in range(0,len(ads)):
    k=ads[i].split(' ')
    for j in range(len(k)):
        k[j].lower()
    words.extend(k)
words=filter(None, words)



def read_file(filename):

    words=[]
    string_file = filename.readline()
    list_sentence=[]
    while(string_file):
        w=string_file.split('.')
        list_sentence.extend(w)
        string_file = filename.readline()

    for i in range(0, len(list_sentence)):
        w=list_sentence[i].split(' ')
        words.extend(w)

    for i in range(0,len(words)):
        if(words[i]=='\n'):
            words[i]=[]
        else:
            words[i]=words[i].lower()

    for i in range(0,len(words)):#duplicate
        c =words[i]
        for j in range(0,len(words)):
            if (words[j] == c and i < j):
                words[j]=[]
    words=filter(None, words)
    return words


def words_of_string(text):
    #function cuts the line into words
    #функция разрезает предложение на слова и возвращает список слов (без их повторений)
    all_words=[]
    k=text.split('.')
    for i in range(0, len(k)):
        w=k[i].split(' ')
        all_words.extend(w)
    for i in range(0,len(all_words)):#duplicate
        c =all_words[i]
        for j in range(0,len(all_words)):
            if (all_words[j] == c and i < j):
                all_words[j]=[]
    all_words=filter(None, all_words)
    return all_words

def popular_words(list_sentence,  N):
    #the creation of the dictionary the most popular words
    #arguments are a list of all the proposals and the length of the dictionary
    #функция составляет словарь длины N
    #на вход принимает список предложений, режет их с помощью предыдущей функции
    #проходит по всем словам, добавляем их в словарь, считаем сколько их в списке и берем N штук
    dic={}
    for stroka in list_sentence:
        li=words_of_string(stroka)
        for x in li:
            if len(x)>3:
                if x in dic:
                    dic[x]=dic[x]+1
                else:
                    dic[x]=1
    print "quantity of all words"
    print len(dic)
    b = dic.items()
    b.sort(key=lambda item: item[1], reverse=True)
    print len(b)
    for i in range(N, len(b)):
        b[i]=[]
    b=filter(None, b)
    print "Dictionary is ready"
    return b

if __name__== "__main__":

    def ld(p, encoding="utf-8"):
        # загрузка объекта
        with codecs.open(p, "rt", encoding=encoding) as f:
            return json.load(f)

    data = ld("user_file.json")  # dict with data for run our program
    print data['length of dictionary']
    length=int(data['length of dictionary'])

    dict=popular_words(ads, length)
    print np.shape(dict)
    for i in range(0, length):

        out.write((dict[i][0]))
        out.write('\n')
        dicttime.write((dict[i][0]))
        dicttime.write('\n')
        dicttime.write(str(dict[i][1]))
        dicttime.write('\n')
