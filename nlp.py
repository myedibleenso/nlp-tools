#!/usr/bin/env python
from __future__ import unicode_literals, division
from nltk.corpus import wordnet as wn
import nltk
import re

###regex
url_pattern = r'(https?:\/\/.*[\r\n]*)'
punct_pattern = re.compile("(([)!?.'(*-8;:=<?>@DP[\]\\dop{}|]+\s?)+)")
punct = "!')(*-.8;:=<?>@DP[]\dop{}|"

###
token_pattern = r'''(?x)              # set flag to allow verbose regexps
				 https?\://([^ ,;:()`'"])+   # URLs
				| [<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]   # emoticons
				| [\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?   # emoticons, reverse orientation
				| ([A-Z]\.)+                # abbreviations, e.g. "U.S.A."
				| \w+(-\w+)*                # words with optional internal hyphens
				| \$?\d+(\.\d+)?%?          # currency and percentages
				| \.\.\.                    # ellipsis
				| @+[\w_]+                  # twitter-style user names
				| \#+[\w_]+[\w\'_\-]*[\w_]+ # hashtags
				| [.,;"'?():-_`]            # these are separate tokens
				'''

class Tokenizer(object):
	def __init__(self):
		pass

	def clean(self, text):
		"""
		"""
		clean_text = re.sub(url_pattern, " URL ", text, flags=re.MULTILINE)
		clean_text = re.sub("&amp;", "&", clean_text, flags=re.MULTILINE)
		return clean_text

	def allPunct(self, w):
		return True if all([c in punct for c in w]) else False

	def joinPunctuationSequence(self, sentence):
		"""
		join a punctuation sequence 
		into a single token
		"""
		#ensure sentence is a list
		sentence = sentence.split() if type(sentence) is not list else sentence
		text = ""
		for i in range(len(sentence)):	
			w = sentence[i]
			if self.allPunct(w) and (0 <= i-1 <= len(sentence)) and (i+1 < len(sentence)):
				prev_w = sentence[i-1]
				next_w = sentence[i+1]
				if self.allPunct(next_w):
					text += w
				else:
					text += "{0} ".format(w)
			else:
				text += "{0} ".format(w)

		text = text.split()
		return text

	def correct_tokenization(self, tokenized_text):
		"""
		correct tokenization.
		
		combine single element lists 
		of punctuation with previous line

		combine sequences of punctuation 
		into a single token (ex. emoticons)
		"""
		corrected = []
		for line in tokenized_text:
			if all([w in punct for w in line]):
				corrected[-1] = corrected[-1] + line if corrected else ""
			else:
				corrected.append(line)
		#combine punctuation sequences into a single token
		corrected = [self.joinPunctuationSequence(c) for c in corrected]
		return corrected

	def tokenize(self, text):
		"""
		"""
		text = self.clean(text)
		tokenized_text = [nltk.tokenize.word_tokenize(sent) for sent in nltk.tokenize.sent_tokenize(text)]
		#correct tokenization
		tokenized_text = self.correct_tokenization(tokenized_text)
		return tokenized_text if len(tokenized_text) > 1 else tokenized_text[0]


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

	def assign_senses(self, words, tags):	
		wn_tags = [self.get_wordnet_pos(tag) for tag in tags]
		senses = []
		for i in xrange(len(wn_tags)):
			tag = wn_tags[i]
			if tag:
				word = words[i]
				print "word: {0}\ttag: {1}".format(word, tag)
				synsets = wn.synsets(word, pos=tag)
				if len(synsets) > 1:
					senses.append("UNKNOWN")
				elif len(synsets) == 1:
					senses.append(synsets[0].name())
			else:
				senses.append(None)
		print senses
		return senses

	def get_frames(self, sense):
		synset = wn.synset(sense)
		frames = set()
		for lemma in synset.lemmas():
			for i in range(len(lemma.frame_ids())):
				frame_id = lemma.frame_ids()[i]
				frame = lemma.frame_strings()[i].replace(lemma.name(), "___")
				frames.add((frame_id, frame))
		return sorted(list(frames), key=lambda (frame_id, frame): frame_id)

	def semantic_profile(self, sense):
		synset = wn.synset(sense)
		name_value_pairs = []
		# All the available synset methods:
		print "definition:\t{0}".format(synset.definition())
		if synset.examples():
			print "examples:\n{0}".format("\n".join(["\t{0}".format(s) for s in synset.examples()]))
		if synset.lexname():
			print "lexname:\t{0}".format(synset.lexname())
		if synset.lemma_names():
			print "lemma names:\t{0}".format(", ".join(synset.lemma_names()))
		if synset.similar_tos():
			print "similar to:\t{0}".format(", ".join(synset.similar_tos()))
		if synset.attributes():
			print "attributes:\t{0}".format(", ".join(synset.attributes()))
		if synset.verb_groups():
			print "verb groups:\t{0}".format(", ".join(synset.verb_groups()))
		if synset.also_sees():
			print "also see:\t{0}".format(", ".join(synset.also_sees()))
		if synset.frame_ids():
			print "Frames:\n{0}".format("\n".join(["\t{0}\t{1}".format(frame_id, frame) for frame_id, frame in self.get_frames(sense)]))
		if synset.hypernyms():
			print "{0}:\t{1}".format("hypernyms", ', '.join([s.name() for s in synset.hypernyms()]))
		if synset.hyponyms():
			print "{0}:\t{1}".format("hyponyms", ', '.join([s.name() for s in synset.hyponyms()]))
		if synset.instance_hypernyms():
			print "{0}:\t{1}".format("instance hypernyms", ', '.join([s.name() for s in synset.instance_hypernyms()]))
		if synset.instance_hyponyms():
			print "{0}:\t{1}".format("instance hyponyms", ', '.join([s.name() for s in synset.instance_hyponyms()]))
		if synset.member_holonyms():
			print "{0}:\t{1}".format("member holonyms", ', '.join([s.name() for s in synset.member_holonyms()]))
		if synset.member_meronyms():
			print "{0}:\t{1}".format("member meronyms", ', '.join([s.name() for s in synset.member_meronyms()]))
		if synset.substance_holonyms():
			print "{0}:\t{1}".format("substance holonyms", ', '.join([s.name() for s in synset.substance_holonyms()]))
		if synset.substance_meronyms():
			print "{0}:\t{1}".format("substance meronyms", ', '.join([s.name() for s in synset.substance_meronyms()]))
		if synset.part_holonyms():
			print "{0}:\t{1}".format("part holonyms", ', '.join([s.name() for s in synset.part_holonyms()]))
		if synset.part_meronyms():
			print "{0}:\t{1}".format("part meronyms", ', '.join([s.name() for s in synset.part_meronyms()]))
		if synset.instance_hypernyms():
			print "{0}:\t{1}".format("entailments", ', '.join([s.name() for s in synset.entailments()]))
		if synset.instance_hyponyms():
			print "{0}:\t{1}".format("causes", ', '.join([s.name() for s in synset.causes()]))		

	def find_related(self, sense):
		definition = wn.synset(sense).definition()
		words = tokenizer.tokenize(definition)
		tags = tagger(words)
		return [(words[i], disambiguator.get_wordnet_pos(tags[i])) for i in range(len(tags)) if disambiguator.get_wordnet_pos(tags[i])]

	def get_lemmas(self, sense):
		#wn.synset
		pass

	def extract_from_known_definitions(self, sentence):
		known = [sentence.senses[i] for i in xrange(len(sentence.senses)) if sentence.senses[i] not in [None, "UNKNOWN"]]
		related = set()
		for k in known:
			for r in self.find_related(k):
				related.add(r)
		return related

	def disambiguate(self, sentence):
		pass

######################################
#tagger = nltk.pos_tag
tagger = lambda words: [t for (w,t) in nltk.pos_tag(words)]
tokenizer = Tokenizer()
disambiguator = Disambiguator()
######################################

class Sentence(object):
	def __init__(self, words, pos_tags=None, senses=None):
		#if not isinstance(words, list):
		#	raise TypeError("words must be a list")
		self.words = self.set_words(words)
		self.pos_tags = self.set_tags(pos_tags)
		self.senses = self.set_senses(senses)

	def set_words(self, words):
		return words if type(words) is list else tokenizer.tokenize(words)

	def set_tags(self, pos_tags):
		try:
			return pos_tags if pos_tags else tagger(self.words)
		except:
			return pos_tags if pos_tags else ["UNKNOWN"]*len(self.words)
	
	def set_senses(self, senses):
		try:
			return disambiguator.assign_senses(self.words, self.pos_tags)
		except:
			return senses if senses else [""] * len(self.words)

	def __repr__(self):
		return ' '.join("{w}__{p}__{s}".format(w=self.words[i],p=self.pos_tags[i],s=self.senses[i]) for i in xrange(len(self.words)))

	def to_string(self):
		return ' '.join("{w}__{p}__{s}".format(w=self.words[i],p=self.pos_tags[i],s=self.senses[i]) for i in xrange(len(self.words)))

	def tuples(self):
		"""
		(word, pos, sense)
		"""
		return [(self.words[i], self.pos_tags[i], self.senses[i]) for i in xrange(len(self.words))]

