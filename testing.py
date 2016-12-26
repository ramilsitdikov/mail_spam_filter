import pickle
from vectors import DIC
from ada_boost import *
from bag_of_words import get_vectors_from_path


vectors = get_vectors_from_path("test.txt", DIC)
#clf2 = AdaBoost()
with open('data.pickle', 'rb') as f:
    clf2= pickle.load(f)
answer = clf2.classify([vectors])
print answer

