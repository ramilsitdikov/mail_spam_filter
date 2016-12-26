from ada_boost import *
import time
from functions import *
import pickle

test = "enron"

print("Loading dataset...")
dataset = load("./dataset/%s.pickle" % test)
print("Successful load dataset with %s/%s/%s elements" % (len(dataset.training_set),len(dataset.validation_set),len(dataset.testing_set)))

fout = open("validation_output.csv", "w")
fout.write("number_of_rounds;max_depth;min_examples;min_entropy;training_time;precision1;recall1;fscore1;precision2;recall2;fscore2;accuracy1;accuracy2\n")
fout.close()
print
print("Validating classifier...")
best_fscore = 0.0
for number_of_rounds in [1]:
    for max_depth in [8]:
        for min_examples in [20]:
            for min_entropy in [0.0]:

                start = time.time()
                clf = AdaBoost(number_of_rounds, max_depth, min_examples, min_entropy)
                clf.train(dataset.training_set, dataset.training_labels)
                training_time = time.time() - start

                training_results = clf.classify(dataset.training_set)
                acc1, precision1, recall1, fscore1 = measure(training_results, dataset.training_labels)

                validation_results = clf.classify(dataset.validation_set)
                acc2, precision2, recall2, fscore2 = measure(validation_results,dataset.validation_labels)

                fout = open("validation_output.csv","a")
                line = "%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (number_of_rounds, max_depth, min_examples, min_entropy, training_time, precision1, recall1, fscore1, precision2, recall2, fscore2, acc1, acc2)
                line.replace(".",",")
                fout.write(line)
                fout.close()

                print "Params:",(number_of_rounds, max_depth, min_examples, min_entropy)
                print "Time:",training_time
                print "Training set:", (precision1, recall1, fscore1, acc1)
                print "Validation set:", (precision2, recall2, fscore2, acc2)
                if fscore2 > best_fscore:
                    best_fscore = fscore2
                    print "Dumping classifier..."
                    with open('data.pickle', 'wb') as f:
                        pickle.dump(clf, f)
                    print "Successful dump classifier"
                print "Best fscore:", best_fscore
                print
    #             break
    #         break
    #     break
    # break

print("Successful finish!")
