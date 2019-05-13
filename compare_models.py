import argparse
import gensim
from gensim.models import Word2Vec
import numpy as np
import pandas as pd 

import plotly.tools as tls
import plotly.plotly as py
import plotly.graph_objs as go

import cufflinks as cf

parser = argparse.ArgumentParser(description='Compare cosine similarity of two models')
parser.add_argument('-m1','--model1',help='path of the first model to use')
parser.add_argument('-m2','--model2',help='path of the second model to use')


args = parser.parse_args()

def get_nearest_neighbours(token,model,n=100):
	try:
		if token.isupper():
			token = token.lower()
		nn = {word.rstrip():float(sim) for (word,sim) in model.most_similar(token,topn=n)}

	except KeyError:
		nn = None
	return nn


def compute_differences(model1path,model2path):

	leere_sig = ["Europa_NE","Elite_NN","System_NN","Korrektheit_NN","Lügenpresse_NN","souverän_ADJA","unabhängig_ADJA","Widerstand_NN","wir_PPER","Volk_NN"]
	model1 = gensim.models.keyedvectors.KeyedVectors.load(model1path)
	model2 = gensim.models.keyedvectors.KeyedVectors.load(model2path)
	word_dist = {}
	for i,word in enumerate(model1.wv.index2word[:50000]):
	#for i,word in enumerate(leere_sig):

		lemma = word.split("_")[0]
		pos = word.split("_")[1]
		print(word)
		count1 = model1.wv.vocab[word].count
		try:
			count2 = model2.wv.vocab[word].count
		except KeyError:
			count2 = 0

		distances2 = []
		distances1 = []
		dist_diff = []
		oov_count = 0
		nn1_sim = None
		nn2_sim = None

		word_dist[word] = {"nn1":None,"m1_count":count1,"m2_count":count2,"nn2":None,"nn1_mean":None,"nn1_median":None,"nn1_std":None,"nn2_mean":None,"nn2_median":None,"nn2_std":None,"oov_count":None,"nn_overlap":None,"std_1-2":None,"pos":pos,"word":lemma}
	
		
		try:
			vec2 = model2.wv[word] #check if word exists in other model
			nn1 = get_nearest_neighbours(word,model1)
			nn2 = get_nearest_neighbours(word,model2)
			
			intersection = []
			if nn1 != None:
				nn1_sim = list(nn1.values())
				nn1_dist = [1-sim for sim in nn1_sim]
				
				for nn in nn1.keys():

					try: # checks if nn is in model 2
						sim = model2.similarity(word,nn)
						distances1.append(1-nn1[nn])
						distances2.append(1-sim)		
						
					except KeyError:
						oov_count += 1
				if nn2 != None:
					nn2_sim = list(nn2.values())
					nn2_dist = [1-sim for sim in nn2_sim]
					intersection = set(nn1.keys()).intersection(nn2.keys())
					word_dist[word].update({"nn_overlap":len(intersection),
					"nn2_mean":np.mean(nn2_sim),
					"nn2_median":np.median(nn2_sim),
					"nn2_std":np.std(nn2_sim),
					"std_1-2":np.std(nn1_sim)-np.std(nn2_sim),
					"nn2_dist":sum(nn2_dist),
					"nn2_dist_mean":np.mean(nn2_dist),
					"pnn_dist_in_sbz":sum(distances2),
					"pnn_dist_in_pnn":sum(distances1),
					"pnn_dist_in_sbz_mean":np.mean(distances2),
					"pnn_dist_in_sbz_std":np.std(distances2),
					"pnn_dist_in_sbz_median":np.median(distances2)})
				
				word_dist[word].update({
				"nn1":nn1,
				"nn2":nn2,
				"nn1_mean":np.mean(nn1_sim),
				"nn1_median":np.median(nn1_sim),
				"nn1_std":np.std(nn1_sim),
				"nn1_dist":sum(nn1_dist),
				"nn1_dist_mean":np.mean(nn1_dist),
				"oov_count":oov_count
				})
			


		except KeyError:
			print(word)
		except ZeroDivisionError:
			continue

	return word_dist

def visualize_models(data):
	data = data.sort_values('pnn_dist_in_sbz',ascending=False)
	data = data[:3000]

	data.iplot(x='m1_count',y='m2_count',z='pnn_dist_in_sbz',mode='markers',categories="pos",colorscale="spectral",text='word',title='pines vs. spiegelzeit v3',kind="3dscatter",xTitle='pines frequency (absolute)',yTitle='sbz frequency (absolute)',zTitle='pnn in sbz sum')
			
def main():
	word_stats = compute_differences(args.model1,args.model2)
	df = pd.DataFrame.from_dict(word_stats,orient="index")
	df.sort_values('pnn_dist_in_sbz',ascending=False)
	df = df.dropna(thresh=10)
	df.to_csv('march_orient_pines.csv',sep="\t",encoding="utf-8")
	#df = pd.read_csv('v3_pi_vs_sbz.csv',sep="\t",encoding="utf-8")
	

	#visualize_models(df)



if __name__ == '__main__':
	main()