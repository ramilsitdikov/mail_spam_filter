# -*- coding: UTF-8 -*-
import codecs
import glob

data_file = "network.tsv"

def standart_string (_str):
    '''
    #функция очищает строку от лишних знаков пунктуации и пробелов
    :param _str: any string
    :return: lower string without disign, whitespace
    '''
    new_str = _str.lower()
    new_str = new_str.strip()
    new_str = new_str.replace("/", " ")
    new_str = new_str.replace("»", " ")
    new_str = new_str.replace("«", " ")
    new_str = new_str.replace(".", "")
    new_str = new_str.replace("!", "")
    new_str = new_str.replace("- ", "")
    new_str = new_str.replace("-", " ")
    new_str = new_str.replace(",", "")
    new_str = new_str.replace("\t", " ")
    new_str = new_str.replace("\n", " ")
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

def text2vectors(text, dic, max_len):
    '''
    функция делает вектора для текстаб по словарю указанной длинны
    если текст короткий - он дополнится нулевыми векторами
    если длиннее, то обрежется
    :param text:  data string
    :param dic:     dic with popular word, making before
    :param max_len: max len of all texts
    :return: needed quantity vectors
    '''
    vectors = []
    zero = [0]*len(dic)
    text = standart_string(text)
    words = text.split(" ")
    for i in range(len(words)):
        vect = gen_vector(words[i], dic)
        vectors.append(vect)
    while(len(vectors) < max_len):
        vectors.append(zero)
    vectors = vectors[0:max_len]
    return vectors

def get_words(text):
    #берем текст, удаляем лишнее и режем на слова
    text = standart_string(text)
    words = text.split(" ")
    return words

path = "/home/nina/Загрузки/hem/*.txt"
def get_all_words(path_to_folder):
    #функция получает на вход путь к папке с файлами
    #функция возвращает список всех слов из всех файлов
    #в полученном списке куча шлака и повторяющихся слов
    #функция нужна, чтобы составить словарь
    files = glob.glob(path_to_folder)
    quantity_of_files = len(files)
    all_texts = []
    for i in range(quantity_of_files):  # список всех текстов в папке
        in_file = open(files[i])
        str = ''
        for c in in_file:
            str = str + c
        all_texts.append(str)
    all_words = []
    for i in range(len(all_texts)):
        words_one_text = get_words(all_texts[i])
        all_words.extend(words_one_text)
    #print (all_words[0:10])
    return all_words

get_all_words(path)

def get_vectors_from_path(path, dic, max_len):
    #функция получает путь к файлу с текстом и словарь
    #функция делает список векторов для текста заданного длинны
    data = codecs.open(path, "r", "utf-8")
    vectors = text2vectors(data,dic, max_len)
    return vectors