import argparse
import gensim
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases,Phraser
import numpy as np
from collections import Counter
import random
import sys
import json

from get_data import get_documents

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser(description='calculate wordembeddings from ES and ngram matrix')
parser.add_argument('-i','--index',type=str,default='main_corpus',help='elasticsearch index (default: test_small')
parser.add_argument('-k','--key',nargs="+",default=["class"],help='field of elastic search to access (default: class)')
parser.add_argument('-v','--value',nargs="+",default=["pbv"],help='the value of the field in elastic search (default: pbv)')
parser.add_argument('-d','--data_type',type=str,default='lemma',help='the type of data to access in content (default: lemma)')
parser.add_argument('-l','--language',type=str,default='de',help='the lang of the words to extract')
parser.add_argument('-f','--infile',default=None,help='if not none file with tokenized list of words')
parser.add_argument('-cwb','--cwb_input',default=False,type=bool,help='if not none, extract the texts from the input stream')
parser.add_argument('-s','--cwb_s_attribute',default='text',help='the xml attribute from the cwb input to split the texts at')
parser.add_argument('-mn','--model_name')
parser.add_argument('-p','--phrases',default=True,help="If phrases True, phrases will be precomputed")
parser.add_argument('-max','--max_vocab_size',default=100000,help='the max vocab size, if None, the entire vocab is used')
parser.add_argument('-o','--outdir',default="embedding_models",help='directory to store the computed we model')

args = parser.parse_args()

def handle_cwb(data):
	"""
	Function reads in the stdin from CWB and splits it into lists of text which are each stored in a results dict
	"""
	print('splitting cwb data')
	texts = {}
	text = []
	id_ = "0"
	source= "0"
	for line in data[1:]:
		if line.startswith('</'+args.cwb_s_attribute+'_id'):
			texts[id_+"_"+source] = text
			text = []
		elif line.startswith('<'+args.cwb_s_attribute) == False:
			text.append(line.rstrip())
		elif line.startswith('<'+args.cwb_s_attribute+'_id'):
			print(line)
			id_ = line.split(" ")[1].rstrip('>\n')
		# the '_decade' string can be replaced with another s attribute of text
		#i.e. year or source (otherwise if not specified when calling the code, it will be '00')
		elif line.startswith('<'+args.cwb_s_attribute+'_decade'):
			print(line)
			source = line.split(" ")[1].rstrip('>\n')
		else:
			#sollte nur linien mit <text> matchen und </text_id > ...)
			continue

	return texts


def get_texts():
	results = []
	search_terms = zip(args.key,args.value)
	for (key,value) in search_terms:
		print(key,value)
		results += get_documents(args.index,str(key),str(value),args.data_type,args.language)

	print(len(results))

	return results

def read_file(filename):
	infile = open(filename,'r')
	texts = [lemma for lemma in infile if len(lemma) > 1 and lemma.isspace() == False]

	return texts


def compute_wordembeddings(texts,phrases=args.phrases):
	#documents = ["the mayor of new york was there", "machine learning can be useful sometimes","new york mayor was present"]
	
	docs = [doc for id_,doc in sorted(texts.items())]
	ids = sorted(texts.keys())
	print('...computing phrases')
	phrases = Phrases(docs,min_count=5,threshold=10,delimiter=b"_")
	bigram_phraser = Phraser(phrases)
	phrased_texts = {}
	for i in range(0,len(texts)-1):
		tokens = bigram_phraser[texts[ids[i]]]
		phrased_texts[ids[i]] = tokens 

	phr_docs = [doc for id_,doc in sorted(phrased_texts.items())]
	tri_phrases = Phrases(phr_docs,min_count=5,threshold=5,delimiter=b"_")
	trigram_pharser = Phraser(tri_phrases)

	tri_phrased_texts = {}
	count = 1
	for i in range(0,len(texts)-1):
		tokens = trigram_pharser[phrased_texts[ids[i]]]
		tri_phrased_texts[ids[i]]=tokens

		if count % 100 == 0:
			print('...',count)
		count += 1

	with open(args.model_name+'_trigrams.json','w') as out_corpus:
		json.dump(tri_phrased_texts,out_corpus)

	tri_phrased_it = [doc for id_,doc in sorted(tri_phrased_texts.items())]
	print('...computing WE')
	model = Word2Vec(tri_phrased_it,window=6,sorted_vocab=1,min_count=5,sample=1e-5,sg=1,workers=15) #special sample für populismus
	
	model.save(args.outdir.rstrip('/')+"/"+args.model_name+'.model')
	print('done!')

def main():

	if args.infile != None:
		results = read_file(args.infile)
		
	elif args.cwb_input == True:
		print("reading input from cwb")
		data = sys.stdin.readlines()
		results = handle_cwb(data)

	else:
		results = get_texts()
		
	print('number of texts to compute WE for:',len(results))
	print('...computing WE')

	compute_wordembeddings(results)
	

if __name__ == '__main__':
	main()