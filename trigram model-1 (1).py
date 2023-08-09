import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing - Fall 2022 
Daniel Bauer
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    result = []
    start = ('START')
    end = ('STOP')

    #insert a Start at the beginning
    sequence.insert(0, start)

    # fill in the correct amount of starts for the n-gram group
    for i in range((n - 2)):
        sequence.insert(0, start)

    # add the end to the end of n-gram
    sequence.append(end)

    # create a list of each n-gram and then convert it to tuples to add to the final list
    for i in range(len(sequence)):
        temp_list = []
        for j in range(n):
            if i + j < len(sequence):
                temp_list.append(sequence[i + j])

    #check edge cases to make sure that all are n-grams
        if len(temp_list) == n:
            tupleOfList = tuple(temp_list)
            result.append(tupleOfList)
    # return final result
    return result


class TrigramModel(object):

    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.totalWords = 0

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = {}  # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        ##Your code here

        # find all of the counts
        for sentence in corpus:
            uniGrams = get_ngrams(sentence, 1)
            biGrams = get_ngrams(sentence, 2)
            triGrams = get_ngrams(sentence, 3)
            for uGram in uniGrams:
                #We do not keep track of Start for the uniGram counts
                if uGram != tuple(('START',)):
                    # Make sure to add Stop to the amount of total token count
                    self.unigramcounts[uGram] = self.unigramcounts.get(uGram, 0) + 1
                    self.totalWords += 1

            for bGram in biGrams:
                self.bigramcounts[bGram] = self.bigramcounts.get(bGram, 0) + 1
            for tGram in triGrams:
                self.trigramcounts[tGram] = self.trigramcounts.get(tGram, 0) + 1

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        bigram = tuple(trigram[0:2])
        bigramCounts = self.bigramcounts.get(bigram, 0)
        if bigramCounts != 0:
            return self.trigramcounts.get(trigram, 0) / bigramCounts
        else:
            # if the bigram is not in the system then base it only off of the unigram word
            return self.raw_unigram_probability(trigram[2])

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        unigram = tuple(bigram[0])
        unigramCounts = self.unigramcounts.get(unigram, 0)
        if unigramCounts != 0:
            value = self.bigramcounts.get(bigram, 0) / self.unigramcounts.get(unigram, 0)
            return value
        else:
            return self.unigramcounts.get(unigram, 0)

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        return self.unigramcounts.get(unigram, 0) / self.totalWords

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0

        bigram = tuple(trigram[1:3])

        list = []
        list.append(trigram[2])
        unigram = tuple(list)

        result = lambda1 * self.raw_trigram_probability(trigram) + \
                 lambda2 * self.raw_bigram_probability(bigram) + \
                 lambda3 * self.raw_unigram_probability(unigram)

        return result

    def sentence_logprob(self, sentence):
        trigrams = get_ngrams(sentence, 3)
        finalProb = 0

        for tgram in trigrams:
            stp = self.smoothed_trigram_probability(tgram)
            if stp == 0:
                print(tgram)
            logProb = math.log2(stp)
            finalProb += logProb
        return finalProb

    def perplexity(self, corpus):
        runningPerplexity = 0
        count = 0

        for sentence in corpus:
            #This is for the end marker that is missing in each sentence
            count += 1

            sentenceLogProbability = self.sentence_logprob(sentence)
            runningPerplexity += sentenceLogProbability
            for word in sentence:
                count += 1

        l = runningPerplexity / count
        perplexity = 2 ** (-1 * l)
        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)
    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f),
                                             model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f),
                                             model2.lexicon))
        total += 1
        if pp1 <= pp2:
            correct += 1

    for f in os.listdir(testdir2):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f),
                                             model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f),
                                             model2.lexicon))
        total += 1
        if pp2 <= pp1:
            correct += 1


    return correct / total


if __name__ == "__main__":
    #model = TrigramModel(sys.argv[1])
    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    #dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print('pp', pp)
    # Essay scoring experiment:
    acc = essay_scoring_experiment('./hw1_data/ets_toefl_data/train_high.txt',
                                   './hw1_data/ets_toefl_data/train_low.txt',
                                   './hw1_data/ets_toefl_data/test_high',
                                   './hw1_data/ets_toefl_data/test_low')
    print(acc)
