import io
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

input_filename = 'input1.txt'
output_filename = 'output.txt'

# чтение из файла
with io.open(input_filename, encoding='utf-8', mode='r') as file:
    contents = file.read()

# разбиение на токены
word_tokens = word_tokenize(contents)

# нормализация токенов
normalized_contents = []
for word in word_tokens:
    normalized_contents.append(morph.parse(word)[0].normal_form)

# удаление стоп-слов
stop_words = set(stopwords.words('russian'))
punctuation = string.punctuation + '–«»'
filtered_contents = [
    w for w in normalized_contents if not w.lower() in stop_words and w not in punctuation]

# составление частотного словаря
dictionary = {}
for word in filtered_contents:
    if word in list(dictionary):
        dictionary[word] += 1
    else:
        dictionary[word] = 1


with io.open(output_filename,  encoding='utf-8', mode='w') as file:
    file.write('Частотный словать ' + input_filename + '\n')
with io.open(output_filename,  encoding='utf-8', mode='a') as file:
    for key, value in dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True)).items():
        file.write('{}: {}\n'.format(key, value))
