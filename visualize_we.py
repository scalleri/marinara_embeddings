import argparse
import gensim
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import os


parser = argparse.ArgumentParser(description='visualize wordembeddings')
parser.add_argument('-md','--model_dir',default="embedding_models/",help="directory with the models to visualize")
parser.add_argument('-tp','--tensorflow_path',default="tensorboard_test/",help="path to the tensorboard files")

args = parser.parse_args()

def visualize_wordembeddings():
	#based on: https://github.com/sudharsan13296/visualise-word2vec/blob/master/Word2vec%20Embeddings.ipynb
	#and https://stackoverflow.com/questions/45020971/visualizing-multiple-embedding-with-tensorflow?rq=1

	if os.path.isdir(args.tensorflow_path) == False:
		os.mkdir(args.tensorflow_path)

	
	sub_features = []
	for model_file in os.listdir(args.model_dir):
		if model_file.endswith('.model'):

			cluster_meta = False
			print(model_file)

			model = gensim.models.keyedvectors.KeyedVectors.load(args.model_dir+model_file)
			max_size = len(model.wv.vocab)-1
			#max_size = 200000
			w2v = np.zeros((max_size,model.layer1_size))
			#Number of tensors (64465) do not match the number of lines in metadata (64461).
			
			
			#check if clusterfile exists
			#if os.path.isfile('/home/call/medienling/we_models'+model_file+'.cluster_500.txt'):
			if os.path.isfile('/home/call/n_hot_embeddings/show_models/geburtsberichte_we_cluster_2000.txt'):
				cluster_dict = {}
				cluster_meta = True
				#with open('/home/call/medienling/we_models'+model_file+'.cluster_500.txt') as cluster_info:
				with open('/home/call/n_hot_embeddings/show_models/geburtsberichte_we_cluster_2000.txt') as cluster_info:
					for line in cluster_info:
						row = line.rstrip('\n').split('\t')
						cluster_name = ("|").join([row[i] for i in range(2,len(row)-2,2)])
						cluster_dict[row[0]] = row[1]+"_"+cluster_name


			empty_vectors = 0
			with open(args.tensorflow_path+model_file+"metadata.tsv","w+") as file_metadata:
				if cluster_meta == True:
					file_metadata.write('Word\tClusterlabel\n')
				for i,word in enumerate(model.wv.index2word[:max_size]):
					if word.isspace() != True:
						w2v[i] = model.wv[word]
						if cluster_meta == True and word in cluster_dict:
							file_metadata.write(word + '\t' + cluster_dict[word] + '\n')
						else:
							file_metadata.write(word + '\n')
					else:
						empty_vectors += 1 # count the number of 'rows' that have to be deleted from the w2v, bc we skip lines

			

			print(empty_vectors)
			if empty_vectors != 0:
				w2v = np.delete(w2v,np.s_[-empty_vectors:],0)
					
			print(max_size-empty_vectors)
			print(w2v.shape)
			with tf.device("/cpu:0"):
				embedding = tf.Variable(w2v, trainable=False, name=model_file)
				sub_features.append(embedding)
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	tf.global_variables_initializer().run()
	path = args.tensorflow_path

	writer = tf.summary.FileWriter(path, sess.graph)
	config = projector.ProjectorConfig()

	for model in sub_features:
		print(model.name)
		embedding = config.embeddings.add()
		embedding.tensor_name = model.name
		embedding.metadata_path = model.name[:-2]+"metadata.tsv"

	projector.visualize_embeddings(writer, config)

	saver.save(sess, path+'/model.ckpt', global_step=max_size)

def main():
	visualize_wordembeddings()


if __name__ == '__main__':
	main()