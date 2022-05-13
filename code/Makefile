experimento_wisard_1:
	python3 classifier.py --file /home/jolima/Documentos/multilabel-classification/dataset/kaggle_dataset.csv --n_iter 3 --min_ram 3 --max_ram 5 --min_ter 3 --max_ter 5 --n_sample 10
preprocessamento_tfidf:
	python3 preprocessing_tfidf.py --file ../dataset/kaggle_dataset.csv --n_sample 1000 --featureSource 1
preprocessamento_doc2vec:
	python3 preprocessing_doc2vec.py --file ../dataset/kaggle_dataset.csv --n_sample 2000 --featureSource 1	
binarizacao_doc2vec:
	python3 binarization.py --file /home/jolima/Documentos/multilabel-classification/code/vectors_doc2vec_spacy --term_size 8
experimento_wisard_doc2vec_pws:
	python3 classifier_v2.py --file /home/jolima/Documentos/multilabel-classification/code/vectors_doc2vec_spacy.csv --n_iter 3 --min_ram 24 --max_ram 36 --min_ter 16 --max_ter 24 --n_sample 1000 --preprocess 1 --method 0
experimento_wisard_doc2vec_br:
	python3 classifier_v2.py --file /home/jolima/Documentos/multilabel-classification/code/vectors_doc2vec_spacy.csv --n_iter 3 --min_ram 32 --max_ram 32 --min_ter 32 --max_ter 32 --n_sample 1000 --preprocess 1 --method 1
experimento_wisard_tfidf_br:
	python3 classifier_v2.py --file ../code/vectors_tfidf.csv --n_iter 3 --min_ram 16 --max_ram 16 --min_ter 16 --max_ter 16 --n_sample 10 --preprocess 0 --method 1 --featureSource 1

preprocessing:
	python3.8 classifierv.py 

	