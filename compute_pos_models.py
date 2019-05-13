import argparse
import gensim
from gensim.models import Word2Vec
import numpy as np
from collections import Counter
import random
import sys
import os
import re

from get_data import get_pos_doc


parser = argparse.ArgumentParser(description='calculate wordembeddings from ES and ngram matrix')
parser.add_argument('-i','--index',type=str,default='main_corpus',help='elasticsearch index (default: test_small')
parser.add_argument('-k','--key',nargs="+",default=["class"],help='field of elastic search to access (default: class)')
parser.add_argument('-v','--value',nargs="+",default=["pbv"],help='the value of the field in elastic search (default: pbv)')
parser.add_argument('-d','--data_type',type=str,default='lemma',help='the type of data to access in content (default: lemma)')
parser.add_argument('-l','--language',type=str,default='de',help='the lang of the words to extract')
parser.add_argument('-f','--infile',default=None,help='if not none file with tokenized list of words')
parser.add_argument('-cwb','--cwb_input',default=False,type=bool,help='if not none, extract the texts from the input stream')
parser.add_argument('-s','--cwb_s_attribute',default='text_id',help='the xml attribute from the cwb input to split the texts at')
parser.add_argument('-mn','--model_name')
parser.add_argument('-o','--outdir',default="pos_models",help='directory to store the computed we model')

args = parser.parse_args()

def handle_cwb(data):
	"""
	Function reads in the stdin from CWB and splits it into lists of text which are each stored in a results list 
	"""
	print('splitting cwb data')
	texts = []
	text = []
	for line in data[1:]:
		if line.startswith('</'+args.cwb_s_attribute+'>'):
			texts.append(text)
			text = []
		elif line.startswith('<'+args.cwb_s_attribute+'>') == False:
			text.append(("_").join(line.rstrip().split()))
		else:
			#sollte nur linien mit <text> matchen
			continue
	return texts



def get_texts():
	results = []
	search_terms = zip(args.key,args.value)
	for (key,value) in search_terms:
		print(key,value)
		results += get_pos_doc(args.index,str(key),str(value),args.data_type,args.language)

	print(len(results))

	return results

def read_file(filename):
	"""
	Fileinupt looks as follows:
	lemma\tpos
	-> will be transformed into [lenmm_pos,lemma_pos,lemma_pos,lemma_pos....]
	"""
	infile = open(filename,'r')
	texts = [("_").join(line.rstrip().split()) for line in infile if len(line) > 1 and line.isspace() == False]

	return texts


def compute_wordembeddings(texts):
	#model = Word2Vec(texts,min_count=10,window=10,sg=1,workers=15)

	model = Word2Vec(texts,window=10,sorted_vocab=1,min_count=10,sample=1e-5,sg=1,workers=15) #special sample für populismus
	model.save(args.outdir.rstrip('/')+"/"+args.model_name+'.model')


def main():

	if os.path.isdir(args.outdir) == False:
		os.mkdir(args.outdir)

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