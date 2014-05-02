#!/usr/bin/env python
from __future__ import unicode_literals, division
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
from nlp import *

def semcor_to_sentences():
	sentences = []
	for s in semcor.tagged_sents(tag="both"):
		nested_list = [[(word, tag, w.label()) for word, tag in w.pos()] for w in s]
		words, tags, senses = zip(*[triplet for sublist in nested_list for triplet in sublist])
		senses = [t if (t is None or not t.isupper()) else None for t in senses]
		sentence = Sentence(words=words, pos_tags=tags, senses=senses)
		sentences.append(sentence)
	return sentences


class Disambiguator(object):

	def __init__(self):
		pass
		
	def get_wordnet_pos(self, treebank_tag):

		if treebank_tag.startswith('J'):
			return wn.ADJ
		elif treebank_tag.startswith('V'):
			return wn.VERB
		elif treebank_tag.startswith('N'):
			return wn.NOUN
		elif treebank_tag.startswith('R'):
			return wn.ADV
		else:
			return None

	def polysemous_words(self, sentence):	
		wn_tags = [self.get_wordnet_pos(tag) for tag in sentence.pos_tags]
		for i in xrange(len(wn_tags)):
			tag = wn_tags[i]
			word = sentence.words[i]
			synsets = wn.synsets(word, pos=tag)
			if len(synsets) > 1:
				sentence.senses[i] = "UNKNOWN"
			elif len(synsets) == 1:
				sentence.senses[i] = synsets[0]
			else:
				sentence.senses[i] = None
		return sentence





	def polysemous_words(text):
		pass
