nlp-tools
=========

#Description

Building on the shoulders of giants...


#Getting started 

- _To do!_


====

#Features

##Tokenization

- sentence and word tokenization has improved performance over `nltk` for social media data (emoticons)
  - Tokenizer supports emoticons
 
##Storage class
- Sentence class modeled after CoreNLP representation

====

#To Do

##Tokenizer class:

- Add method for flattening sentences into words


##Tagger:

- category mapping (all punct to PUNCT?) 
- Stanford dependencies (might be tricky interfacing with the jvm...)?

##Disambiguator (WSD system):  

- Add backoff for overlap scores of 0 (choose first sense?)
- Modify context IR for Lesk-based metrics (hyponyms, hypernyms for all known senses, etc)
- `word2vec` from semcor data (any other sources?) and sentence similarity measures?
- Intelligent negation modeling?
- Context window expansion (ex. +/- n sentences when available)?
- Include supervised methods?
- LDA, LSA, etc

##Sentence class:

 - Include normalized sentiment scores for words (pos neg)?


#References

- Lesk (1986) based models for WSD adapted from Liling Tan's [`pywsd`](https://github.com/alvations/pywsd) implementation.
 <p>

```
@misc{pywsd14,
 	author = {Liling Tan},
 	title = {Pywsd: Python Implementations of Word Sense Disambiguation (WSD) Technologies},
 	howpublished = {https://github.com/alvations/pywsd}},
 	year = {2014}
}
```

- [`NLTK 3`](https://github.com/nltk/nltk) is the nlp backbone supporting this.
   <p>

```
@inproceedings{Loper:2002:NNL:1118108.1118117,
    author = {Loper, Edward and Bird, Steven},
    title = {NLTK: The Natural Language Toolkit},
    booktitle = {Proceedings of the ACL-02 Workshop on Effective Tools and Methodologies for Teaching Natural Language Processing and Computational Linguistics - Volume 1},
    series = {ETMTNLP '02},
    year = {2002},
    location = {Philadelphia, Pennsylvania},
    pages = {63--70},
    numpages = {8},
    url = {http://dx.doi.org/10.3115/1118108.1118117},
    doi = {10.3115/1118108.1118117},
    acmid = {1118117},
    publisher = {Association for Computational Linguistics},
    address = {Stroudsburg, PA, USA},
} 
```

