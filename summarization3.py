import collections
import io
import math
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
stemmer = SnowballStemmer(language='russian')

PROGRAM = 1

output_filename = 'output.txt'
with io.open(output_filename,  encoding='utf-8', mode='w') as file:
    file.write('TF-IDF\n')

corpus = ['Tais/1.txt', 'Tais/2.txt', 'Tais/3.txt', 'Tais/4.txt',
          'Tais/5.txt', 'Tais/6.txt', 'Tais/7.txt', 'Tais/8.txt', 'Tais/9.txt']
filtered_corpus = []
tf_idf_corpus = []

for input_filename in corpus:
    # чтение из файла
    with io.open(input_filename, encoding='utf-8', mode='r') as file:
        contents = file.read()

    # разбиение на токены
    word_tokens = word_tokenize(contents)

    # нормализация токенов
    normalized_contents = []
    if (PROGRAM == 1):
        # нормализация pymorphy2
        for word in word_tokens:
            normalized_contents.append(morph.parse(word)[0].normal_form)
    else:
        # нормализация nltk
        for word in word_tokens:
            normalized_contents.append(stemmer.stem(word.lower()))

    # удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    punctuation = string.punctuation + '–«»'
    filtered_contents = [
        w for w in normalized_contents if not w.lower() in stop_words and w not in punctuation]
    filtered_corpus.append(filtered_contents)

    count_all = float(len(filtered_contents))

for filtered_contents in filtered_corpus:
    # http://nlpx.net/archives/57
    # TF-IDF_dictionary
    tf_idf = collections.Counter(filtered_contents)
    for i in tf_idf:
        tf_idf[i] = tf_idf[i]/count_all * \
            math.log10(len(filtered_corpus) /
                       sum([1.0 for j in filtered_corpus if i in j]))
    tf_idf_corpus.append(tf_idf)

    with io.open(output_filename,  encoding='utf-8', mode='a') as file:
        for key, value in tf_idf.items():
            file.write('\'{}.txt\' {} – {}\n'.format(
                filtered_corpus.index(filtered_contents)+1, value, key))
