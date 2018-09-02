import gensim
import os
import codecs
from glob import glob


class MySentences():
	def __init__(self, dirname_list):
		self.dirname_list = dirname_list

	def __iter__(self):
		for fname in self.dirname_list:
			for line in codecs.open(fname, 'r', 'utf-8'):
				pieces = line.strip().replace(' ', '')
				words = [w for w in pieces]
				yield words


root_path = os.getcwd() + os.sep
raw_corpus_path = root_path + "assets" + os.sep + "raw_corpus" + os.sep
cooked_corpus_path = root_path + "assets" + os.sep + "cooked_corpus" + os.sep
original_files_path = raw_corpus_path + "original_data" + os.sep

def gen_w2c():
	corpus = glob(original_files_path + "*" + os.sep + "*txtoriginal*")

	sentences = MySentences(corpus)

	model = gensim.models.Word2Vec(
		sentences, size=100, window=5, min_count=1, iter=100, workers=4)
	# model.save(os.path.join(cooked_corpus_path, 'vec.m'))

	model.wv.save_word2vec_format(os.path.join(cooked_corpus_path, 'vec.txt'), binary=False)
