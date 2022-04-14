import collections
import io
from itertools import islice
import math
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
stemmer = SnowballStemmer(language='russian')

compression = 0.5

PROGRAM = 1

output_filename = 'output4.txt'
with io.open(output_filename,  encoding='utf-8', mode='w') as file:
    file.write('Annotation\n')

corpus = ['Tais copy/1.txt', 'Tais copy/2.txt']
# corpus = ['Tais/1.txt']
filtered_corpus = []
tf_idf_corpus = []

for input_filename in corpus:
    # чтение из файла
    with io.open(input_filename, encoding='utf-8', mode='r') as file:
        contents = file.read()

    # разбиение на предложения
    sent_tokens = sent_tokenize(contents)
    sent_tf = {}

    words_ft = []

    for sentence in sent_tokens:
        # разбиение на токены
        word_tokens = word_tokenize(sentence)

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

        tf_idf = collections.Counter(filtered_contents)
        for i in tf_idf:
            tf_idf[i] = tf_idf[i]/count_all

        words_ft.append(tf_idf)

        sent_tf[sentence] = sum(tf_idf.values())

    sorted_by_tf = dict(
        sorted(sent_tf.items(), key=lambda item: item[1], reverse=True))

    n = int(len(sorted_by_tf) * compression)

    sorted_by_tf = dict(islice(sorted_by_tf.items(), n))

    annotation = [w for w in sent_tokens if w in sorted_by_tf]

    with io.open(output_filename,  encoding='utf-8', mode='a') as file:
        file.write('\n\n{} список слов \n'.format(input_filename))

        for sent in words_ft:
            for key, value in sent.items():
                file.write('{} – {}\n'.format(value, key))

        for key, value in sent_tf.items():
            file.write('{} – {}\n'.format(value, key))

        file.write('\n\n{} annotation \n'.format(input_filename))

        file.write((' '.join(annotation)))


#     with io.open(output_filename,  encoding='utf-8', mode='a') as file:
#         for key, value in tf_idf.items():
#             file.write('\'{}.txt\' {} – {}\n'.format(
#                 filtered_corpus.index(filtered_contents)+1, value, key))
