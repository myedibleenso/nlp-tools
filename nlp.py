#!/usr/bin/env python
from __future__ import unicode_literals, division
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from itertools import chain
from nltk.corpus import semcor
from collections import defaultdict, Counter
import unittest
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
	UNSPECIFIED = "<unspecified>"

	def __init__(self):
		self.verbose = True
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

	def yap(self, msg):
		if self.verbose:
			print msg

	def is_single(self, w, t, sense):
		wn_t = disambiguator.get_wordnet_pos(t)
		return True if (len(wn.synsets(w,pos=wn_t)) == 1 and (sense != Disambiguator.UNSPECIFIED) and (sense != Sentence.CLOSED_CLASS)) else False

	def is_labeled_poly(self, w, t, sense):
		wn_t = disambiguator.get_wordnet_pos(t)
		return True if (len(wn.synsets(w,pos=wn_t)) > 1 and (sense != Disambiguator.UNSPECIFIED) and (sense != Sentence.CLOSED_CLASS)) else False

	def semantic_corpora_statistics(self, gold, experimental):
		"""
		"""
		all_tags = sum(s.length for s in gold)
		
		experimental_polysemous = sum(s.senses.count(Sentence.UNKNOWN) for s in experimental) / all_tags
		experimental_single_senses = sum(len([sense for sense in s.senses if (sense != Sentence.UNKNOWN) and (sense != Sentence.CLOSED_CLASS)]) for s in experimental) / all_tags
		experimental_closed_tags = sum(s.senses.count(Sentence.CLOSED_CLASS) for s in experimental) / all_tags
		
		gold_polysemous = sum(sum([1 for i in xrange(s.length) if iself.s_labeled_poly(s.words[i], s.pos_tags[i], s.senses[i])]) for s in gold) / all_tags
		gold_single_senses = sum(sum(1 for i in xrange(s.length) if self.is_single(s.words[i], s.pos_tags[i], s.senses[i])) for s in gold) / all_tags
		gold_closed = sum(s.senses.count(Sentence.CLOSED_CLASS) for s in gold) / all_tags
		gold_unspecified = sum(s.senses.count(Disambiguator.UNSPECIFIED) for s in gold) / all_tags

		print "Total senses:\t{0}".format(all_tags)

		print "Experimental % polysemous:\t{0:.4}".format(experimental_polysemous)
		print "Experimental % single sense:\t{0:.4}".format(experimental_single_senses)
		print "Experimental % closed class:\t{0:.4}".format(experimental_closed_tags)

		print "Gold % polysemous:\t{0:.4}".format(gold_polysemous)
		print "Gold % single sense:\t{0:.4}".format(gold_single_senses)
		print "Gold % closed class:\t{0:.4}".format(gold_unspecified)
		print "Gold % closed class:\t{0:.4}".format(gold_unspecified)

	def semcor_sentences(self, labeled=True):
		sentences = []
		for s in semcor.tagged_sents(tag="both"):
			triplets = []
			for w in s:
				sense = w.label()
				triplets += [(self.check_sense(sense=sense, word=w, tag=self.clean_pos(w, p)), w, self.clean_pos(w, p)) for (w,p) in w.pos()]

			senses, words, tags = zip(*triplets)			

			if labeled:
				sentence = Sentence(words=words, pos_tags=tags, senses=self.clean_labels(senses))
			#for testing
			else:
				sentence = Sentence(words=words, pos_tags=tags)
			
			sentences.append(sentence)
		return sentences

	def write_sentences(self, fname, sentences):
		with open(fname, 'w') as out:
			for s in sentences:
				out.write( "{0}\n".format(" ".join("__".join(t) for t in s.tuples())))

	def get_synsets(self, word, tag=None):
		"""
		retrieve synsets for the specified PoS
		"""
		#self.yap("getting synsets for {0}...".format(word))
		return wn.synsets(word, pos=tag) if tag != wn.ADJ else self.get_adj_synsets(word)
	
	def get_adj_synsets(self, word):
		"""
		Get adjective synsets
		"""
		#self.yap("getting adj synsets for {0}...".format(word))
		return [s for s in wn.synsets(word) if s.pos() == wn.ADJ_SAT or s.pos() == wn.ADJ]
	
	def check_sense(self, sense, word, tag):
		
		# if there isn't a sense or the sense if the PoS...
		if not sense or sense == tag:
			return Sentence.CLOSED_CLASS

		# see if the sense exists in wordnet DB
		try:
			if wn.synset(sense):
				has_sense = True
		except:
			has_sense = False

		# retrieve all possible synsets for a word...
		synsets = self.get_synsets(word)

		# case 1: If the specified synset doesn't exist 
		#         and the word has no synsets...
		if not synsets and not has_sense:
			return Sentence.CLOSED_CLASS

		# case 2: If the specified synset exists...
		elif has_sense:
			return sense

		# case 3: Only one synset?
		if len(synsets) == 1:
			return synsets[0].name()

		# case 4: Something might be wrong 
		#         with the synset formatting...
		else:
			return self.check_sense_pos(sense, word, tag)

	def check_sense_pos(self, sense, word, tag):
		"""
		"""

		# retrieve all synsets and synset names for the word
		synsets = self.get_synsets(word)
		synset_names = [s.name() for s in synsets]

		# get the wordnet PoS for the given treebank tag
		wn_tag = self.get_wordnet_pos(tag)
		
		formatted_sense = ".{0}.".format(wn_tag).join(sense.split("."))

		# did the formatting change work?
		if formatted_sense in synset_names:
			return formatted_sense

		# if the word is an adjective, it might have an ADJ_SAT match
		if wn_tag == wn.ADJ:
			adj_SAT_sense = ".s.".join(sense.split("."))
			if adj_SAT_sense in synset_names:
				return adj_SAT_sense
		
		# Check the wn PoS:
		wn_tag = self.get_wordnet_pos(tag)
		candidates = [s.name() for s in synsets if s.pos() == wn_tag]
		
		# if only one match...
		if len(candidates) == 1:
			return candidates[0]
	
		# if multiple matches...
		elif len(candidates) > 1:
			return Disambiguator.UNSPECIFIED

		# if there are multiple matching senses...
		return self.UNSPECIFIED

	def clean_pos(self, word, tag):
		"""
		"""
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
		#self.yap("sterilizing {0}...".format(word))
		return word.replace("-","") if not wn.synsets(word) else word

	def assign_senses(self, words, tags):	
		"""
		Assign senses for non-polysemous words
		"""
		polysemous_tag = Sentence.UNKNOWN
		closed_class_tag = Sentence.CLOSED_CLASS

		wn_tags = [self.get_wordnet_pos(tag) for tag in tags]
		senses = []
		for i in xrange(len(wn_tags)):
			wordnet_tag = wn_tags[i]
			if wordnet_tag:
				word = self.sterilize_word(words[i])
				synsets = self.get_synsets(word, tag=wordnet_tag)
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
		return [(sentence.words[i], self.get_wordnet_pos(sentence.pos_tags[i])) for i in xrange(len(sentence.senses)) if sentence.senses[i] == Sentence.UNKNOWN]

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
		for s in self.get_synsets(word, tag=pos): 
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
		
		for synset in self.get_synsets(ambiguous_word, tag=pos):
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
		sense_dictionary = {s:set(self.flatten(tokenizer.tokenize(s.definition()))) for s in self.get_synsets(ambiguous_word, tag=pos)}
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
			return self.get_synsets(ambiguous_word, tag=pos)[0]
			return wn.synsets(ambiguous_word)[0]
		except:
			return Sentence.CLOSED_CLASS

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
		self.yap("\nDisambiguating \"{0}\" with {1}...".format(sentence.semantic_representation(), method.__name__))
		
		context_sentence = [stemmer.stem(w) for w in sentence.words] if self.stem else sentence.words
		
		score = "NA"
		disambiguated_senses = sentence.senses[:]
		for (word, tag, i) in ambiguous_words:
			
			try:
				score, best_sense = method(context=context_sentence, ambiguous_word=word, pos=tag)[0]
			except:
				best_sense = method(context=context_sentence, ambiguous_word=word, pos=tag)
			
			self.yap("\nWord:\t\t{0}".format(word))
			try:
				self.yap("Sense:\t\t{0}".format(best_sense.name()))
				self.yap("Definition:\t{0}".format(best_sense.definition()))		
				disambiguated_senses[i] = best_sense.name()
			except:
				disambiguated_senses = best_sense
		
		disambiguated_sentence = Sentence(words=sentence.words[:], pos_tags=sentence.pos_tags[:], senses=disambiguated_senses)
		
		self.yap("\n{0}".format(disambiguated_sentence))
		self.yap(disambiguated_sentence.semantic_representation())
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
		return right/total

################################################
class SemcorExperiment(object):
	
	def __init__(self):
		self.ignore_unspecified = True
		print "Generating sense-annotated semcor sentences..."
		self.gold = disambiguator.semcor_sentences()
		print "Generating experimental semcor sentences..."
		self.experimental = disambiguator.semcor_sentences(labeled=False)
		print "Counting mismatches..."
		self.raw_total = None
		self.total = None 
		self.count_mismatches()
	
	def count_mismatches(self):
		"""
		"""
		raw_total_mismatches = 0
		selective_mismatches = 0
		for i in xrange(len(self.gold)):
			g = self.gold[i]
			e = self.experimental[i]
			for idx in xrange(len(g.senses)):
				if g.senses[idx] != e.senses[idx]:
					raw_total_mismatches += 1
					if g.senses[idx] != Disambiguator.UNSPECIFIED:
						selective_mismatches += 1

		self.raw_total = raw_total_mismatches
		self.total = selective_mismatches

	def compare_labels(self, gold, experimental):
		"""
		compare the labels of two sentences
		"""
		mismatches = 0 
		#list of triplets (word, gold sense, experimental sense)
		error_list = list()
		
		# find selective mismatches...
		if self.ignore_unspecified:
			for i in xrange(len(gold.senses)):
				if (gold.senses[i] != experimental.senses[i]) and (gold.senses[i] != Disambiguator.UNSPECIFIED):
					mismatches += 1
					wn_tag = disambiguator.get_wordnet_pos(gold.pos_tags[i]) or Sentence.CLOSED_CLASS
					error_list.append((gold.words[i], wn_tag, gold.senses[i], experimental.senses[i]))
		
		# find all mismatches...
		else:
			for i in xrange(len(gold.senses)):
				if gold.senses[i] != experimental.senses[i]:
					mismatches += 1
					wn_tag = disambiguator.get_wordnet_pos(gold.pos_tags[i]) or Sentence.CLOSED_CLASS
					error_list.append((gold.words[i], wn_tag, gold.senses[i], experimental.senses[i]))
			
		return mismatches, error_list

	def wsd_performance(self, method=None):
		"""
		Checks accuracy of baseline WSD method by default
		"""
		method = method or disambiguator.most_freq_sense

		mismatches = 0 
		mismatch_dict = defaultdict(int)

		verbosity = disambiguator.verbose
		#temporarily silence disambiguator
		#disambiguator.verbose = False
		for i in xrange(len(self.gold)):
			g = self.gold[i]
			e = self.experimental[i]
			#context, ambiguous_word, pos=None
			new_e = disambiguator.disambiguate(sentence=e, method=method)
			#compare sense labels of disambiguated sentence and gold sentence
			error_count, errors = self.compare_labels(gold=g, experimental=new_e)
			#increment error count
			mismatches += error_count
			#store any errors
			if errors:
				for e in errors:
					mismatch_dict[e] += 1

		#restore disambiguator verbosity
		#disambiguator.verbose = verbosity

		total = self.total if self.ignore_unspecified else self.raw_total

		right = total - mismatches
		accuracy = right / total

		#report accuracy and errors
		return accuracy, Counter(mismatch_dict)

	def write_results(self, fname, error_dict, threshold=1):
		"""
		"""
		with open(fname, 'w') as out:
			for (e, c) in [(error, count) for (error, count) in sorted(error_dict.items(), key=lambda x: x[-1], reverse=True) if count >= threshold]:
   				out.write("{0}\t{1}\n".format('\t'.join(e),c))

######################################
######################################
#tagger = nltk.pos_tag
tagger = lambda words: [t for (w,t) in nltk.pos_tag(words)]
tokenizer = Tokenizer()
disambiguator = Disambiguator()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
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
			return disambiguator.assign_senses(words=self.words, tags=self.pos_tags)
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

 
class Tests(unittest.TestCase):

	def test_sense_assignment(self):
		text = "Her reply stung me , but this was too important to let my hurt make any difference ."
		tags = "PRP$ NN VB PRP , CC DT VB RB JJ TO VB PRP$ JJ VB DT NN .".split()
		correct_senses = [Sentence.CLOSED_CLASS, Sentence.UNKNOWN, Sentence.UNKNOWN, Sentence.CLOSED_CLASS, 
						  Sentence.CLOSED_CLASS, Sentence.CLOSED_CLASS, Sentence.CLOSED_CLASS, Sentence.UNKNOWN,
 						  Sentence.UNKNOWN, Sentence.UNKNOWN, Sentence.CLOSED_CLASS, Sentence.UNKNOWN, 
 						  Sentence.CLOSED_CLASS, Sentence.UNKNOWN, Sentence.UNKNOWN, Sentence.CLOSED_CLASS, 
 						  Sentence.UNKNOWN, Sentence.CLOSED_CLASS]
		s = Sentence(text, pos_tags=tags)
		self.assertEqual(s.senses, correct_senses)


##############################################################

def demo():
	examples = ["I hammered 40 nails and now my hand hurts.", 
				"That stereo has great bass.", 
				"I caught a huge bass."]
	for example in examples:
		s = Sentence(example)
		disambiguator.polysemous_words(s)
		disambiguator.disambiguate(s)

def run_tests():
	"""unit tests"""
	print
	print "*"*30
	print "*", " "*10, "Tests", " "*9, "*"
	print "*"*30
	unittest.main()

##############################################################

if __name__ == '__main__':
	#demo capabilities
	demo()
	#run unit tests
	unittest.main()

