import pdb
import settings
import gensim
from gensim.models.doc2vec import Doc2Vec
import os
import collections
import smart_open
import random

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

class Doc2VecModel:

	def __init__(self,train_file,test_file,name='doc2vec',seed=None):
		self.train_corpus = list(read_corpus(train_file))
		self.test_corpus = list(read_corpus(test_file, tokens_only=True))
		self.name = name
		self.model_name = settings.PERSIST_LOCATION+self.name+'_'+str(settings.VECTOR_SIZE)+'.mdl'
		if not os.path.exists(self.model_name):
			self.model = Doc2Vec(size=settings.VECTOR_SIZE,min_count=settings.MIN_FREQ,seed=seed,workers=settings.WORKER_THREADS,dm_concat=1,iter=settings.N_ITERATION)
		else:
			print "Loaded Pre Trained Model"
			self.model = Doc2Vec.load(self.model_name)

	def train_model(self,force_train=False):
		print "Training Model"
		if force_train or not os.path.exists(self.model_name):
			self.model.build_vocab(self.train_corpus)
			self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.iter)
			#Saving the model
			self.model.save(self.model_name)
		else:
			print "Pre Trained model is used instead of training"

	def trim_training_data(self):
		self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

	def get_vector(self,document):
		doc_token = gensim.utils.simple_preprocess(line)
		return self.model.infer_vector(doc_token)