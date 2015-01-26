import re
import random
from math import exp, log
from datetime import datetime
from operator import itemgetter

def clean(s):
	"""
		Returns a cleaned, lowercased string
	"""
	return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()

def get_data_tsv(loc_dataset,opts):
	"""
	Running through data in an online manner
	Parses a tsv file for this competition and yields label, identifier and features
	output:
			label: int, The label / target (set to "1" if test set)
			id: string, the sample identifier
			features: list of tuples, in the form [(hashed_feature_index,feature_value)]
	"""
	for e, line in enumerate(open(loc_dataset,"rb")):
		if e > 0:
			r = line.strip().split("\t")
			id = r[0]
			
			if opts["clean"]:
				try:
					r[2] = clean(r[2])
				except:
					r[1] = clean(r[1])
			
			if len(r) == 3: #train set
				features = [(hash(f)%opts["D"],1) for f in r[2].split()]
				label = int(r[1])
			else: #test set
				features = [(hash(f)%opts["D"],1) for f in r[1].split()]
				label = 1
				
			if opts["2grams"]:
				for i in xrange(len(features)-1):
					features.append((hash(str(features[i][0])+str(features[i+1][0]))%opts["D"],1))
			yield label, id, features
			
def dot_product(features,weights):
	"""
	Calculate dot product from features and weights
	input:
			features: A list of tuples [(feature_index,feature_value)]
			weights: the hashing trick weights filter, note: length is max(feature_index)
	output:
			dotp: the dot product
	"""
	dotp = 0
	for f in features:
		dotp += weights[f[0]] * f[1]
	return dotp	

def train_tron(loc_dataset,opts):
	start = datetime.now()
	print("\nPass\t\tErrors\t\tAverage\t\tNr. Samples\tSince Start")
	
	#Initializing the weights
	if opts["random_init"]:
		random.seed(3003)
		weights = [random.random()] * opts["D"]
	else:
		weights = [0.] * opts["D"]
	
	#Running training passes
	for pass_nr in xrange(opts["n_passes"]):
		error_counter = 0
		for e, (label, id, features) in enumerate( get_data_tsv(loc_dataset,opts) ):
			
			dp = dot_product(features, weights) > 0.5
			error = label - dp # error is 1 if misclassified as 0, error is -1 if misclassified as 1
			
			if error != 0:
				error_counter += 1
				# Updating the weights
				for index, value in features:
					weights[index] += opts["learning_rate"] * error * log(1.+value)

		#Reporting stuff
		print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (pass_nr+1,error_counter,round(1 - error_counter /float(e+1),5),e+1,datetime.now()-start))
		
		#Oh heh, we have overfit :)
		if error_counter == 0 or error_counter < opts["errors_satisfied"]:
			print("%s errors found during training, halting"%error_counter)
			break
	return weights
	
def test_tron(loc_dataset,weights,opts):
	"""
		output:
				preds: list, a list with [id,prediction,dotproduct,0-1normalized dotproduct]
	"""
	start = datetime.now()
	print("\nTesting online\nErrors\t\tAverage\t\tNr. Samples\tSince Start")
	preds = []
	error_counter = 0
	for e, (label, id, features) in enumerate( get_data_tsv(loc_dataset,opts) ):

		dotp = dot_product(features, weights)
		dp = dotp > 0.5
		if dp > 0.5: # we predict positive class
			preds.append( [id, 1, dotp ] )
		else:
			preds.append( [id, 0, dotp ] )
		
		if label - dp != 0:
			error_counter += 1
			
	print("%s\t\t%s\t\t%s\t\t%s" % (error_counter,round(1 - error_counter /float(e+1),5),e+1,datetime.now()-start))
		
	#normalizing dotproducts between 0 and 1 TODO: proper probability (bounded sigmoid?), online normalization
	max_dotp = max(preds,key=itemgetter(2))[2]
	min_dotp = min(preds,key=itemgetter(2))[2]
	for p in preds:
		p.append((p[2]-min_dotp)/float(max_dotp-min_dotp)) #appending normalized to predictions

	#Reporting stuff
	print("Done testing in %s"%str(datetime.now()-start))	
	return preds

if __name__ == "__main__":
	
	#Setting options
	opts = {}
	opts["D"] = 2 ** 25
	opts["learning_rate"] = 0.1
	opts["n_passes"] = 80 # Maximum number of passes to run before halting
	opts["errors_satisfied"] = 0 # Halt when training errors < errors_satisfied
	opts["random_init"] = False # set random weights, else set all 0
	opts["clean"] = True # clean the text a little
	opts["2grams"] = True # add 2grams

	#training and saving model into weights
	weights = train_tron("d:\\labeledTrainData.tsv",opts)
	
	#testing and saving predictions into preds
	preds = test_tron("d:\\testData.tsv",weights,opts)

	#writing kaggle submission
	with open("d:\\popcorn.csv","wb") as outfile:
		outfile.write('"id","sentiment"'+"\n")
		for p in sorted(preds):
			outfile.write("%s,%s\n"%(p[0],p[3]))