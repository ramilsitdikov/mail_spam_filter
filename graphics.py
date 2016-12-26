import matplotlib.pyplot as plt
import pylab

def fscore_line(infile):
    train_score = []
    test_score = []
    koord = []
    k = 0
    i = 0
    file = open(infile, "r")
    line = file.readline()
    line = file.readline()
    while line:
        if k ==0:
            line = line.replace('"', '')
            line = line.replace("'", "")
            elements = line.split(';')
            train_score.append( float(elements[7]))
            test_score.append(float(elements[10]))
            koord.append(i)
            i +=1
            line = file.readline()
            k = k + 1
        else:
            line = file.readline()
            k = 0
    file.close()
    plt.title('F-score on learning:')
    line1 = pylab.plot([train_score], label="fscore_on_train")
    line2 = pylab.plot([test_score], label="fscore_on_test")
    pylab.plot(koord, train_score)
    pylab.plot(koord, test_score)
    pylab.legend(("fscore on train", "fscore on test"))
    pylab.show()

def acc_line(infile):
    train_acc = []
    test_acc = []
    koord = []
    k = 0
    i = 0
    file = open(infile, "r")
    line = file.readline()
    line = file.readline()
    while line:
        if k ==0:
            line = line.replace('"', '')
            line = line.replace("'", "")
            elements = line.split(';')
            train_acc.append(float(elements[11]))
            test_acc.append(float(elements[12]))
            koord.append(i)
            i += 1
            line = file.readline()
            k = k + 1
        else:
            line = file.readline()
            k = 0
    file.close()
    plt.title('accuracy on learning:')
    line1 = pylab.plot([train_acc], label="accuracy on train")
    line2 = pylab.plot([test_acc], label="accuracy on test")
    pylab.plot(koord, train_acc)
    pylab.plot(koord, test_acc)
    pylab.legend(("accuracy on train", "accuracy on test"))
    pylab.show()

x1 = [2, 4, 8]
y1 = [0.847039473684, 0.916118421053, 0.945723684211]
x2 = [2, 4, 8]
y2 = [0.819943622269, 0.914728682171, 0.947498238196]

pylab.plot(x2, y2)
pylab.plot(x1, y1)
pylab.legend(( "accuracy on train", "accuracy on test"))
pylab.show()

# fscore_line("validation_output_v1.csv")

# acc_line("validation_output_all.csv")