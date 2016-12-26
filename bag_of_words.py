# -*- coding: UTF-8 -*-
import codecs
import glob
import random

def standart_string (_str):
    '''
    #функция очищает строку от лишних знаков пунктуации и пробелов
    :param _str: any string
    :return: lower string without disign, whitespace
    '''
    new_str = _str.lower()
    new_str = new_str.strip()
    new_str = new_str.replace("/", " ")
    new_str = new_str.replace(".", " ")
    new_str = new_str.replace("!", " ")
    new_str = new_str.replace("-", " ")
    new_str = new_str.replace(",", " ")
    new_str = new_str.replace("\t", " ")
    new_str = new_str.replace("\n", " ")
    new_str = new_str.replace("\r", " ")
    new_str = new_str.replace('"', " ")
    while new_str.find("  ") >= 0:
        new_str = new_str.replace("  ", " ")
    return new_str

all_words = ['try', 'cut', 'run']


def get_words_from_folder(folder_name):
    pass

def dictionary(words, lenght):
    '''

    :param words: list of all words of which we do Dictionary
    :param lenght: max len of dictionary
    :return: sorted list of words
    '''
    dic = {}
    for x in words:
        if x in dic:
            dic[x] = dic[x] + 1
        else:
            dic[x] = 1
    print ("количество всех слов")
    print (len(dic))
    b = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    for i in range(len(b)):
        if b[i][1] < 10:
            b[i] = ()
    b = [x for x in b if x]
    print("Количество слов, которые встречаются чащем, чем 10 раз = " + str(len(b)))
    if (len(b) < lenght):
        print ("Количество слов меньше, чем требуемая длина словаря")
        print ("Длина словаря="+str(len(b)))
    b = b[0:lenght]
    new_b = []
    for i in range(len(b)):
        new_b.append(b[i][0])
    print ("Словарь готов")
    #print (new_b)
    return new_b

#DIC = dictionary(all_words, 3625)

def gen_vector(word, dic):
    #функция делает вектор для слова по словарю
    lenght_of_dic = len(dic)
    vector = [0] * lenght_of_dic
    for i in range(lenght_of_dic):
        if word == dic[i]:
            vector[i] = 1
    return vector

def text2vectors(text, dic):
    '''
    функция делает вектора для текстаб по словарю указанной длинны
    если текст короткий - он дополнится нулевыми векторами
    если длиннее, то обрежется
    :param text:  data string
    :param dic:     dic with popular word, making before
    :return: needed quantity vectors
    '''
    vectors = []
    text = standart_string(text)
    words = text.split(" ")
    len_vector = len(dic)
    vector = [0] * len_vector
    for i in range(len_vector):
        if dic[i] in text:
            vector[i] = 1
    for i in range(len_vector):
        vector[i] = vector[i] / len(words)
    return vector

def get_words(text):
    #берем текст, удаляем лишнее и режем на слова
    text = standart_string(text)
    words = text.split(" ")
    return words

#path = "/home/nina/Загрузки/hem/*.txt"
def get_all_words(path_to_folder):
    #функция получает на вход путь к папке с файлами
    #функция возвращает список всех слов из всех файлов
    #в полученном списке куча шлака и повторяющихся слов
    #функция нужна, чтобы составить словарь
    files = glob.glob(path_to_folder)
    quantity_of_files = len(files)
    all_texts = []
    for i in range(quantity_of_files):  # список всех текстов в папке
        in_file = codecs.open(files[i], "r" ,encoding='utf-8', errors='ignore')
        #print (files[i])
        str1 = in_file.read()
        str1.encode("utf-8")
        all_texts.append(str1)
    all_words = []
    for i in range(len(all_texts)):
        words_one_text = get_words(all_texts[i])
        all_words.extend(words_one_text)
    #print (all_words[0:10])
    return all_words

#get_all_words(path)

def get_vectors_from_path(path, dic):
    #функция получает путь к файлу с текстом и словарь
    #функция делает список векторов для текста заданного длинны
    data = codecs.open(path, "r", encoding='utf-8', errors='ignore')
    data_new = data.read()
    vector = text2vectors(data_new, dic)
    return vector

def pell_mell(a,b):
    #mix list composed for 2 vectors
    c=zip(a,b)
    random.shuffle(c)
    new_a=[]
    new_b=[]
    for i in range(len(a)):
        new_a.append(c[i][0])
        new_b.append(c[i][1])
    return new_a, new_b

def pell_mell2(a, b):
    c = []
    for i in range(len(a)):
        c.append((a[i],b[i]))
    random.shuffle(c)
    new_a = []
    new_b = []
    for i in range(len(a)):
        new_a.append(c[i][0])
        new_b.append(c[i][1])
    return new_a, new_b