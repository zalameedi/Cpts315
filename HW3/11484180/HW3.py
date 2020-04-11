"""
Programmer : Zeid Al-Ameedi
Date : 03-31-2020
Description : CPTS 315 Homework 3. Instructions can be found in the document hw3.pdf.
Collab : Stackoverflow, David Henshaw, Janna Doppa and Piazza (CPTS315 course website).
Uploaded : www.github.com/zalameedi
"""

import numpy


class Perceptron():
    def __init__(self):
        self.traindatafile = './fortunecookiedata/traindata.txt'
        self.trainlabelfile = './fortunecookiedata/trainlabels.txt'
        self.testingfile = './fortunecookiedata/testdata.txt'
        self.testinglabelfile = './fortunecookiedata/testlabels.txt'
        self.stopwordsfile = './fortunecookiedata/stoplist.txt'
        self.outputfile = './output.txt'
        self.learning_rate = 1
        self.num_iterations = 20
        self.num_C = 26
        self.ocrTrain = './OCR-data/ocr_train.txt'
        self.ocrTest = './OCR-data/ocr_test.txt'

    def run(self):
        lines, training_words = self.init_train_data()
        self.del_stop_words(training_words)
        given_dt, vocabulary = self.sort_vocab(training_words)
        fortunes = self.ft_vectors(given_dt, lines, vocabulary)
        train_labels = self.rd_tr_labels()
        test_data = self.creat_test_vectors(given_dt, vocabulary)
        test_labels = self.getting_test_labels()
        el_mistakes, el_test, el_train = self.t_simple_perceptron(given_dt, fortunes, test_data,
                                                                  test_labels, train_labels)
        avg_test_acc, avg_train_acc = self.t_avg_perceptron(given_dt, fortunes, test_data,
                                                            test_labels, train_labels)
        self.gen_report(avg_test_acc, avg_train_acc, el_mistakes, el_test, el_train)
        ocr_train, ocr_train_label = self.rd_train_labels(self.ocrTrain)
        features, ocr_test, ocr_test_label, train_data = self.extract_test_labels(self.ocrTest, ocr_train)
        index_to_letter, letter_to_index, test_data = self.test_ft_vectors(features, ocr_test, ocr_train_label,
                                                                           test_data)
        self.simple_perceptron(self.num_C, self.num_iterations, self.learning_rate, self.outputfile, features, index_to_letter,
                               letter_to_index,
                               ocr_test_label, ocr_train_label, test_data, train_data)

    def accuracy(self, weight, examples, labels):
        c = 0
        my_set = numpy.shape(examples)
        for index in range(0, my_set[0]):
            p_label = numpy.dot(examples[index], numpy.transpose(weight))
            if ((p_label[0] > 0 and labels[index] > 0) or \
                    (p_label[0] <= 0 and labels[index] < 0)):
                c += 1
        return c / my_set[0]

    def init_train_data(self):
        training_words = set()
        f = open(self.traindatafile, 'r')
        lines = f.read().split('\n')
        for line in lines:
            words = line.split(' ')
            for word in words:
                training_words.add(word)
        f.close()
        return lines, training_words

    def del_stop_words(self, training_words):
        f = open(self.stopwordsfile, 'r')
        stop_words = f.read().split('\n')
        for word in stop_words:
            training_words.discard(word)
        f.close()
        del stop_words

    def sort_vocab(self, training_words):
        vocabulary = dict()
        given_dt = len(training_words)
        for index, word in enumerate(sorted(training_words)):
            vocabulary[word] = index
        return given_dt, vocabulary

    def generate_results(self, ITERATIONS, OUT_FILE, avg_test_acc, avg_train_acc, el_mistakes, el_test, el_train):
        f = open(OUT_FILE, 'a')
        for i in range(1, ITERATIONS + 1):
            f.write('iteration-' + str(i) + ' ' + str(el_mistakes[i - 1]) + '\n')
        for i in range(1, ITERATIONS + 1):
            f.write('iteration-' + str(i) + ' ' + str(el_train[i - 1]) + ' ' + str(el_test[i - 1]) + '\n')
        f.write(str(el_train[ITERATIONS - 1]) + ' ' + str(el_test[ITERATIONS - 1]) + '\n')
        f.write(str(avg_train_acc) + ' ' + str(avg_test_acc) + '\n')
        f.close()

    def ft_vectors(self, M, lines, vocabulary):
        fortunes = numpy.zeros((len(lines), M + 1))
        self.feature_vector_extracter(M, lines, fortunes, vocabulary)
        del lines
        return fortunes

    def rd_tr_labels(self):
        f = open(self.trainlabelfile, 'r')
        train_labels = f.read().split('\n')
        for index, label in enumerate(train_labels):
            if int(label) == 0:
                train_labels[index] = -1
            else:
                train_labels[index] = 1
        f.close()
        return train_labels

    def creat_test_vectors(self, M, vocabulary):
        f = open(self.testingfile, 'r')
        lines = f.read().split('\n')
        lines.pop()
        test_data = numpy.zeros((len(lines), M + 1))
        self.feature_vector_extracter(M, lines, test_data, vocabulary)
        f.close()
        return test_data

    def feature_vector_extracter(self, M, lines, test_data, vocabulary):
        for index, line in enumerate(lines):
            words = line.split(' ')
            for word in words:
                if vocabulary.get(word) is not None:
                    test_data[index][vocabulary[word]] = 1
            test_data[index][M] = 1

    def getting_test_labels(self):
        f = open(self.testinglabelfile, 'r')
        test_labels = f.read().split('\n')
        test_labels.pop()
        for index, label in enumerate(test_labels):
            if int(label) == 0:
                test_labels[index] = -1
            else:
                test_labels[index] = 1
        f.close()
        return test_labels

    def t_simple_perceptron(self, M, fortunes, test_data, test_labels, train_labels):
        w = numpy.zeros((1, M + 1))
        max_m = list()
        train_z = list()
        test_z = list()
        my_set = numpy.shape(fortunes)
        self.cont_t_SP(fortunes, max_m, my_set, test_data, test_labels, test_z, train_labels, train_z, w)
        return max_m, test_z, train_z

    def cont_t_SP(self, fortunes, max_m, my_set, test_data, test_labels, test_z, train_labels, train_z, w):
        for i in range(1, self.num_iterations + 1):
            mistakes = 0
            for index in range(0, my_set[0]):
                predicted = numpy.dot(fortunes[index], numpy.transpose(w))
                if predicted[0] * train_labels[index] <= 0:
                    mistakes += 1
                    w = w + self.learning_rate * train_labels[index] * fortunes[index]
            max_m.append(mistakes)
            train_z.append(self.accuracy(w, fortunes, train_labels))
            test_z.append(self.accuracy(w, test_data, test_labels))

    def t_avg_perceptron(self, tot_mistakes, fortunes, test_data, test_labels, train_labels):
        w = numpy.zeros((1, tot_mistakes + 1))
        u = numpy.zeros((1, tot_mistakes + 1))
        c = 1
        S = numpy.shape(fortunes)
        for i in range(1, self.num_iterations + 1):
            for index in range(0, S[0]):
                predicted = numpy.dot(fortunes[index], numpy.transpose(w))
                if predicted[0] * train_labels[index] <= 0:
                    w = w + self.learning_rate * train_labels[index] * fortunes[index]
                    u = u + c * self.learning_rate * train_labels[index] * fortunes[index]
                c += 1
        w = w - u * (1 / c)
        avg_train_acc = self.accuracy(w, fortunes, train_labels)
        avg_test_acc = self.accuracy(w, test_data, test_labels)
        return avg_test_acc, avg_train_acc

    def hash_set(self, ocr_train_label):
        letter_to_index = dict()
        index_to_letter = dict()
        letters = sorted(list(set(ocr_train_label)))
        for index, letter in enumerate(letters):
            letter_to_index[letter] = index
            index_to_letter[index] = letter
        return index_to_letter, letter_to_index

    def gen_report(self, avg_test_acc, avg_train_acc, el_mistakes, test_z, train_z):
        f = open(self.outputfile, 'w')
        for i in range(1, self.num_iterations + 1):
            f.write('iteration-' + str(i) + ' ' + str(el_mistakes[i - 1]) + '\n')
        for i in range(1, self.num_iterations + 1):
            f.write('iteration-' + str(i) + ' ' + str(train_z[i - 1]) + ' ' + str(test_z[i - 1]) + '\n')
        f.write(str(train_z[self.num_iterations - 1]) + ' ' + str(test_z[self.num_iterations - 1]) + '\n')
        f.write(str(avg_train_acc) + ' ' + str(avg_test_acc) + '\n\n')
        f.close()

    def accuracy2(self, weight, examples, labels, d):
        c = 0
        my_set = numpy.shape(examples)
        for i in range(0, my_set[0]):
            p_label = numpy.zeros((1, self.num_C))
            for j in range(0, self.num_C):
                p_label[0][j] = numpy.dot(examples[i], numpy.transpose(weight[j]))
            if d[numpy.argmax(p_label)] == labels[i]:
                c += 1
        return c / my_set[0]

    def rd_train_labels(self, OCR_TRAINING_FILE):
        ocr_train = list()
        ocr_train_label = list()
        f = open(OCR_TRAINING_FILE, 'r')
        lines = f.read().split('\n')
        for line in lines:
            elements = line.split('\t')
            if (len(elements) > 3) and (elements[3] == '_'):
                ocr_train.append(elements[1].lstrip('im'))
                ocr_train_label.append(elements[2])
        f.close()
        return ocr_train, ocr_train_label

    def extract_test_labels(self, OCR_TESTING_FILE, ocr_train):
        ocr_test = list()
        ocr_test_label = list()
        f = open(OCR_TESTING_FILE, 'r')
        lines = f.read().split('\n')
        for line in lines:
            elements = line.split('\t')
            if (len(elements) > 3) and (elements[3] == '_'):
                ocr_test.append(elements[1].lstrip('im'))
                ocr_test_label.append(elements[2])
        f.close()
        features, train_data = self.train_ft_vectors(ocr_train)
        return features, ocr_test, ocr_test_label, train_data

    def train_ft_vectors(self, ocr_train):
        features = len(ocr_train[0])
        train_data = numpy.zeros((len(ocr_train), features + 1))
        for index, example in enumerate(ocr_train):
            for i, digit in enumerate(example):
                train_data[index][i] = int(digit)
            train_data[index][features] = 1
        return features, train_data

    def test_ft_vectors(self, features, ocr_test, ocr_train_label, test_data):
        test_data = numpy.zeros((len(ocr_test), features + 1))
        for index, example in enumerate(ocr_test):
            for i, digit in enumerate(example):
                test_data[index][i] = int(digit)
            test_data[index][features] = 1
        del ocr_test
        index_to_letter, letter_to_index = self.hash_set(ocr_train_label)
        return index_to_letter, letter_to_index, test_data

    def simple_perceptron(self, CLASSES, ITERATIONS, LEARNING_RATE, OUT_FILE, features, index_to_letter,
                          letter_to_index, ocr_test_label, ocr_train_label, test_data, train_data):
        w = numpy.zeros((CLASSES, features + 1))
        el_mistakes = list()
        el_train = list()
        el_test = list()
        my_set = numpy.shape(train_data)
        self.simpPerceptronHelper(CLASSES, ITERATIONS, LEARNING_RATE, my_set, el_mistakes, el_test, el_train,
                                  index_to_letter, letter_to_index, ocr_test_label, ocr_train_label, test_data,
                                  train_data, w)
        self.average_perceptron(CLASSES, ITERATIONS, LEARNING_RATE, OUT_FILE, my_set, el_mistakes, el_test, el_train,
                                features, index_to_letter, letter_to_index, ocr_test_label, ocr_train_label, test_data,
                                train_data)

    def simpPerceptronHelper(self, num_C, ITERATIONS, LEARNING_RATE, S, el_mistakes, el_test, el_train,
                             index_to_letter, letter_to_index, ocr_test_label, ocr_train_label, test_data, train_data,
                             w):
        for i in range(1, ITERATIONS + 1):
            mistakes = 0
            for j in range(0, S[0]):
                predicted = numpy.zeros((1, num_C))
                for k in range(0, num_C):
                    predicted[0][k] = numpy.dot(train_data[j], numpy.transpose(w[k]))
                p_index = numpy.argmax(predicted)
                a_index = letter_to_index[ocr_train_label[j]]
                if p_index != a_index:
                    mistakes += 1
                    w[p_index] = w[p_index] - self.learning_rate * train_data[j]
                    w[a_index] = w[a_index] + self.learning_rate * train_data[j]
            el_mistakes.append(mistakes)
            el_train.append(self.accuracy2(w, train_data, ocr_train_label, index_to_letter))
            el_test.append(self.accuracy2(w, test_data, ocr_test_label, index_to_letter))

    def average_perceptron(self, num_C, ITERATIONS, LEARNING_RATE, OUT_FILE, S, el_mistakes, el_test, el_train,
                           features, index_to_letter, letter_to_index, ocr_test_label, ocr_train_label, test_data,
                           train_data):
        w = numpy.zeros((num_C, features + 1))
        u = numpy.zeros((num_C, features + 1))
        c = 1
        w = self.avgPerceptronHelper(S, c, letter_to_index, num_C, ocr_train_label, train_data, u, w)
        avg_train_acc = self.accuracy2(w, train_data, ocr_train_label, index_to_letter)
        avg_test_acc = self.accuracy2(w, test_data, ocr_test_label, index_to_letter)
        self.generate_results(self.num_iterations, self.outputfile, avg_test_acc, avg_train_acc, el_mistakes, el_test, el_train)

    def avgPerceptronHelper(self, S, c, letter_to_index, num_C, ocr_train_label, train_data, u, w):
        for i in range(1, self.num_iterations + 1):
            for j in range(0, S[0]):
                predicted = numpy.zeros((1, num_C))
                for k in range(0, num_C):
                    predicted[0][k] = numpy.dot(train_data[j], numpy.transpose(w[k]))
                p_index = numpy.argmax(predicted)
                a_index = letter_to_index[ocr_train_label[j]]
                if p_index != a_index:
                    w[p_index] = w[p_index] - self.learning_rate * train_data[j]
                    w[a_index] = w[a_index] + self.learning_rate * train_data[j]
                    u[p_index] = u[p_index] - c * self.learning_rate * train_data[j]
                    u[a_index] = u[a_index] + c * self.learning_rate * train_data[j]
                c += 1
        w = w - u * (1 / c)
        return w


def main():
    print("Perceptron being built. . .(please wait)")
    p = Perceptron()
    p.run()
    print("\n\nProgram complete. Please see {0} for the report.".format(p.outputfile))


if __name__ == '__main__':
    main()
