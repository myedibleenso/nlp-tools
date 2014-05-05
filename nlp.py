#!/usr/bin/env python
from __future__ import unicode_literals, division
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from itertools import chain
from nltk.corpus import semcor
from collections import defaultdict
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

######################################

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
		sentence_length = len(sentence)
		text = ""
		for i in xrange(sentence_length):	
			w = sentence[i]
			if self.allPunct(w) and (0 <= i-1 <= sentence_length) and (i+1 < sentence_length):
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
		return self.correct_tokenization(tokenized_text)
		#return tokenized_text if len(tokenized_text) > 1 else tokenized_text[0]

#######################################################
class Disambiguator(object):

	def __init__(self):
		self.stop = True
		self.stem = True

		self.method = self.modified_lesk
	
	def flatten(self, somelist):
		"""
		credit:
		http://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
		"""
		for i, x in enumerate(somelist):
			while isinstance(somelist[i], list):
				somelist[i:i+1] = somelist[i]
		return somelist

	def semcor_sentences(self, labeled=True):
		sentences = []
		for s in semcor.tagged_sents(tag="both"):
			triplets = []
			for w in s:
				sense = self.clean_sense(w.label())
				triplets += [(sense, w, self.clean_pos(w, p)) for (w,p) in w.pos()]

			senses, words, tags = zip(*triplets)			

			if labeled:
				sentence = Sentence(words=words, pos_tags=tags, senses=self.clean_labels(senses))
			#for testing
			else:
				sentence = Sentence(words=words, pos_tags=tags)
			
			sentences.append(sentence)
		return sentences

	def clean_sense(self, sense):
		return Sentence.CLOSED_CLASS if (not sense or sense == sense.upper()) else sense
	
	def clean_pos(self, word, tag):
		return tag or word

	def clean_labels(self, senses):
		return [s if s != None else Sentence.CLOSED_CLASS for s in senses]
		return senses

	def get_wordnet_pos(self, treebank_tag):
		"""
		Map Penn treebank-style pos tag 
		to wordnet pos tag
		"""

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

	def sterilize_word(self, word):
		return word.replace("-","") if not wn.synsets(word) else word

	def assign_senses(self, words, tags, polysemous_tag="UNKNOWN", closed_class_tag="<closed-class>"):	
		"""
		Assign senses for non-polysemous words
		"""
		wn_tags = [self.get_wordnet_pos(tag) for tag in tags]
		senses = []
		for i in xrange(len(wn_tags)):
			tag = wn_tags[i]
			if tag:
				word = self.sterilize_word(words[i])
				synsets = wn.synsets(word, pos=tag)
				if len(synsets) > 1:
					senses.append(polysemous_tag)
				elif len(synsets) == 1:
					senses.append(synsets[0].name())
				#if there are no synsets, assume it is a closed-class word...
				else:
					senses.append(closed_class_tag)
			else:
				senses.append(closed_class_tag)
		return senses

	def polysemous_words(self, sentence):
		"""
		return list of polysemous words
		"""
		return [(sentence.words[i], self.get_wordnet_pos(sentence.pos_tags[i])) for i in xrange(len(sentence.senses)) if sentence.senses[i] == sentence.UNKNOWN]

	def get_frames(self, sense):
		"""
		Retrieves frames
		"""
		synset = wn.synset(sense)
		frames = set()
		for lemma in synset.lemmas():
			for i in xrange(len(lemma.frame_ids())):
				frame_id = lemma.frame_ids()[i]
				frame = lemma.frame_strings()[i].replace(lemma.name(), "___")
				frames.add((frame_id, frame))
		return sorted(list(frames), key=lambda (frame_id, frame): frame_id)

	def semantic_profile(self, sense):
		"""
		Print a summary of wordnet-derived semantic information 
		available for a given sense
		"""
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

	def senses_summary(self, word, pos):
		for s in wn.synsets(word, pos=pos): 
			examples = '\n\t\t'.join(e for e in s.examples())
			print "name:\t\t{0}\ndef:\t\t{1}\nexamples:\t{2}\n".format(s.name(), s.definition(), examples)

	def compare_synsets(self, synset1, synset2, metric=wn.wup_similarity, metric_options=dict()):
		"""
		compute similarity
		"""
		scores = list()
		for s1 in synset1:
			for s2 in synset2:
				scores.append((s1, s2, metric(s1, s2, **metric_options)))
		scores = sorted(scores, key=lambda (s1, s2, score): score)
		for (s1, s2, score) in scores:
			print "{0}:\t{1}".format(s1.name(), s1.definition())
			print "{0}:\t{1}".format(s2.name(), s2.definition())
			print "{0}:\t{1:.3}\n".format(metric.__name__, score)
		return max(scores, key=lambda x:x[-1])

	def find_related(self, sense):
		definition = wn.synset(sense).definition()
		words = self.flatten(tokenizer.tokenize(definition))
		tags = tagger(words)
		return [(words[i], disambiguator.get_wordnet_pos(tags[i])) for i in xrange(len(tags)) if disambiguator.get_wordnet_pos(tags[i])]

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

	def semantic_signature(self, ambiguous_word, pos=None):
		""" 
		"""
		synset_signatures = dict()
		
		for synset in wn.synsets(ambiguous_word, pos=pos):
			signature = []
			#definition
			signature += self.flatten(tokenizer.tokenize(synset.definition()))
			#examples
			signature += self.flatten([tokenizer.tokenize(e) for e in synset.examples()])
			#lemma names
			signature += synset.lemma_names()
			#lemma names of hypernyms and hyponyms
			signature += self.flatten([s.lemma_names() for s in synset.hypernyms() + synset.hyponyms()])
			
			# Optional: removes stopwords.
			if self.stop and signature:
				try:
					signature = [i for i in signature if i not in stopwords.words('english')]    
				except:
					pass
			# Matching exact words causes sparsity, so optional matching for stems.
			if self.stem and signature: 
				try:
					signature = [stemmer.stem(w) for w in signature]
				except:
					pass

			synset_signatures[synset] = set(signature) if signature else set()
		
		return synset_signatures

	def compare_overlaps(self, context, synset_signatures):
		""" 
		Calculates overlaps between the context sentence and the synset_signature
		and returns a ranked list of synsets from highest overlap to lowest.
		"""
		overlapping_synsets = [] # a tuple of (len(overlap), synset).
		context = set(context)
		for s in synset_signatures:
			overlaps = synset_signatures[s].intersection(context)
			overlapping_synsets.append((len(overlaps), s))

		# Rank synsets from highest to lowest overlap.
		ranked_synsets = sorted(overlapping_synsets, reverse=True)
		
		# Normalize scores. 
		total = sum(score for score, synset in ranked_synsets)

		return ranked_synsets if not total else [(score/total, synset) for score, synset in ranked_synsets]
		
	def simple_lesk(self, context, ambiguous_word, pos=None):
		"""
		Modified Lesk algorithm.
		based on the example by Liling Tan (https://github.com/alvations/pywsd)
		"""
		sense_dictionary = {s:set(self.flatten(tokenizer.tokenize(s.definition()))) for s in wn.synsets(ambiguous_word, pos=pos)}
		ranked_senses = self.compare_overlaps(context, sense_dictionary)
		
		return ranked_senses  

	def modified_lesk(self, context, ambiguous_word, pos=None):
		"""
		Adapted from Banerjee and Pederson (2002)		
		based on the example by Liling Tan (https://github.com/alvations/pywsd)
		"""
		# Get the signatures for each synset.
		synset_signatures = self.semantic_signature(ambiguous_word, pos)

		for s in synset_signatures:
			related_senses = list(set(s.member_holonyms() + s.member_meronyms() + 
									  s.part_meronyms() + s.part_holonyms() + 
									  s.similar_tos() + s.substance_holonyms() + 
									  s.substance_meronyms()))
			
			signature = [w for w in self.flatten([r.lemma_names() for r in related_senses]) if w not in stopwords.words('english')]

			#stem matches
			signature = [stemmer.stem(w) for w in signature]

			for w in signature: 
				synset_signatures[s].add(w)
		
		ranked_senses = self.compare_overlaps(context, synset_signatures)
		return ranked_senses

	def most_freq_sense(self, ambiguous_word, context=None, pos=None):
		"""
		Select the first synset with the specified PoS
		"""
		try:
			return wn.synsets(ambiguous_word, pos=pos)[0]
			return wn.synsets(ambiguous_word)[0]
		except:
			return "UNKNOWN"

	def disambiguate(self, sentence, method=None):
		"""
		disambiguate all polysemous words in a sentence
		"""

		ambiguous_words = [(sentence.words[i], self.get_wordnet_pos(sentence.pos_tags[i]), i) for i in xrange(sentence.length) if sentence.senses[i] == sentence.UNKNOWN] 
		
		#make sure we have some polysemous words
		if not ambiguous_words: 
			return sentence

		method = self.method if not method else method
		
		#display sentence to disambiguate
		print "\nDisambiguating \"{0}\" with {1}...".format(sentence.semantic_representation(), method.__name__)
		
		context_sentence = [stemmer.stem(w) for w in sentence.words] if self.stem else sentence.words
		
		score = "NA"
		disambiguated_senses = sentence.senses[:]
		for (word, tag, i) in ambiguous_words:
			
			try:
				score, best_sense = method(context=context_sentence, ambiguous_word=word, pos=tag)[0]
			except:
				best_sense = method(context=context_sentence, ambiguous_word=word, pos=tag)
			
			print "\nWord:\t\t{0}".format(word)
			print "Sense:\t\t{0}".format(best_sense.name())
			print "Definition:\t{0}".format(best_sense.definition())
			
			disambiguated_senses[i] = best_sense.name()
		
		disambiguated_sentence = Sentence(words=sentence.words[:], pos_tags=sentence.pos_tags[:], senses=disambiguated_senses)
		
		print "\n",disambiguated_sentence
		print disambiguated_sentence.semantic_representation()
		#return the new Sentence
		return disambiguated_sentence


######################################
class Performance(object):

	def __init__(self, gold_labels, experimental_labels):
		self.gold_labels = old_labels
		self.experimental_labels = experimental_labels
		self.accuracy = self.accuracy()

	def accuracy(self):
		total = len(self.gold_labels)
		right = sum(1 for i in xrange(total) if self.gold_labels[i] == self.experimental_labels[i])
		wrong = total - right
		return right/total if right else 1

################################################
class SemcorExperiment(object):
	
	def __init__(self):
		print "Generating sense-annotated semcor sentences..."
		self.gold = disambiguator.semcor_sentences()
		print "Generating experimental semcor sentences..."
		self.experimental = disambiguator.semcor_sentences(labeled=False)
		print "Indexing potential ambiguities..."
		self.test_indices = self.get_mismatch_indices()
		self.total = sum(len(indices) for indices in self.test_indices)
	
	def get_mismatch_indices(self):
		"""
		get indices of sense mismatches
		"""
		indices = list()
		for i in xrange(len(self.gold)):
			g = self.gold[i]
			e = self.experimental[i]
			print "index: {0}".format(i)
			print "gold:\n\twords: {0}\n\ttags: {1}\n\tsenses: {2}\n\tnum senses: {3}".format(g, ' '.join(g.pos_tags), ' '.join(g.senses), len(g.senses))
			print "experimental:\n\twords: {0}\n\ttags: {1}\n\tsenses: {2}\n\tnum senses: {3}".format(e, ' '.join(e.pos_tags), ' '.join(e.senses), len(e.senses))
			print
			if len(g.senses) != len(e.senses):
				raw_input("Mismatch!")
			mismatches = tuple(idx for idx in xrange(len(g.senses)) if g.senses[idx] != e.senses[idx])
			indices.append(mismatches)

		return indices

	def compare_labels(self, gold, experimental, indices=None):
		"""
		compare the labels of two sentences
		"""

		mismatches = 0 
		#list of triplets (word, gold sense, experimental sense)
		error_list = list()
		
		#check if we have indices
		indices = indices or xrange(len(gold.senses))

		for i in indices:
			if gold.senses[i] != experimental.senses[i]:
				mismatches += 1
				error_list.append((gold.words[i], gold.senses[i], experimental.senses[i]))
		
		return mismatches, error_list

	def semcor_baseline(self):
		"""
		Check accuracy of baseline WSD method (i.e. select most common sense)
		"""
		mismatches = 0 
		mismatch_dict = defaultdict(int)
		
		for i in xrange(len(self.test_indices)):
			g = self.gold[i]
			e = self.experimental[i]
			indices = self.test_indices
			new_e = disambiguator.disambiguate(sentence=e, method=disambiguator.most_freq_sense)
			error_count, errors = self.compare_labels(gold=g, experimental=e, indices=indices[i])
			#increment error count
			mismatches += error_count
			#store any errors
			if errors:
				for e in errors:
					mismatch_dict[e]+=1

		accuracy = mismatches/self.total if mismatches else 1
		#report accuracy and errors
		return accuracy, mismatch_dict


######################################
######################################
#tagger = nltk.pos_tag
tagger = lambda words: [t for (w,t) in nltk.pos_tag(words)]
tokenizer = Tokenizer()
disambiguator = Disambiguator()
stemmer = PorterStemmer()
######################################
######################################

class Sentence(object):
	
	UNKNOWN = "UNKNOWN"
	CLOSED_CLASS = "<closed-class>"
	
	def __init__(self, words, pos_tags=None, senses=None):
		#if not isinstance(words, list):
		#	raise TypeError("words must be a list")

		self.words = self.set_words(words)
		self.length = len(self.words)
		self.pos_tags = self.set_tags(pos_tags)
		self.senses = self.set_senses(senses)

	def set_words(self, words):
		if type(words) in [list, tuple]:
			return words if type(words) is tuple else tuple(words)

		sentences = tokenizer.tokenize(words)
		if len(sentences) > 1:
			raise Exception("More than one sentence detected in text!")
		else:
			return sentences[0]

	def set_tags(self, pos_tags):
		try:
			return pos_tags if pos_tags else tagger(self.words)
		except:
			return pos_tags if pos_tags else [self.UNKNOWN]*self.length
	
	def set_senses(self, senses):
		if type(senses) is list and self.length == len(senses):
			return senses

		try:
			return disambiguator.assign_senses(words=self.words, tags=self.pos_tags, polysemous_tag=self.UNKNOWN, closed_class_tag=self.CLOSED_CLASS)
		except:
			return senses if senses else [self.UNKNOWN] * self.length

	def semantic_representation(self):
		representation = self.senses[:]
		for i in xrange(self.length):
			if self.senses[i] == self.UNKNOWN:
				representation[i] = self.words[i].upper()
			elif self.senses[i] == self.CLOSED_CLASS:
				representation[i] = self.words[i]
		return " ".join(representation)

	def __str__(self):
		return ' '.join(self.words)

	def to_string(self):
		return ' '.join("{w}__{p}__{s}".format(w=self.words[i],p=self.pos_tags[i],s=self.senses[i]) for i in xrange(self.length))

	def tuples(self):
		"""
		(word, pos, sense)
		"""
		return [(self.words[i], self.pos_tags[i], self.senses[i]) for i in xrange(self.length)]

if __name__ == '__main__':
	examples = ["I hammered 40 nails and now my hand hurts.", 
	            "That stereo has great bass.", 
	            "I caught a huge bass."]
	for example in examples:
		s = Sentence(example)
		disambiguator.polysemous_words(s)
		disambiguator.disambiguate(s)

