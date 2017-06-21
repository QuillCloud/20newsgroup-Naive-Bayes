import os
import re
import nltk
import random
import math
from sklearn.metrics import f1_score

class Prepare():
    def __init__(self, News_Dir):
        self.News_Dir = News_Dir
        self.index_Dir = News_Dir+"_dic"
        self.count_Dir = News_Dir+"_count"
        self.news_group = {}
        self.cv_fold = 1

    # get words in News, get their stems and drop those in stopwords list
    # write words into new files in new directory call 'XXX_dic'
    def read_and_create_index(self):
        # if new directory 'XXX_dic' already exists, skip this function
        if not os.path.exists(self.index_Dir):
            print("start create {}".format(self.index_Dir))
            os.makedirs(self.index_Dir)
        else:
            print("{} already exists".format(self.index_Dir))
            print()
            return None
        # the stopwords list need to be download
        nltk.download('stopwords')

        # start create 'XXX_dic' directory
        # read words in original dataset, get words' stems and drop stopwords
        # write them into new files in 'XXX_dic' directory
        group_number = 0
        for group_name in os.listdir(self.News_Dir):
            if group_name == ".DS_Store":
                continue
            group_number += 1
            group_path = self.News_Dir + '/' + group_name
            index_group_path = self.index_Dir + '/' + str(group_number)
            if not os.path.exists(index_group_path):
                os.makedirs(index_group_path)
            file_number = 0
            for news_name in os.listdir(group_path):
                file_number += 1
                news_path = group_path + '/' + news_name
                index_path = index_group_path + '/' + str(group_number) + "-" + str(file_number)
                fw = open(index_path, 'w')
                content = open(news_path, 'r', encoding='utf-8',errors='ignore').readlines()
                for line in content:
                    word_list = get_words(line)
                    for word in word_list:
                        fw.write('{}\n'.format(word))
                fw.close()
            print("Group {} done".format(group_name))
        print("Create {} complete".format(self.index_Dir))
        print()
        return None

    # get total specific words in dataset (occur more than 5 times)
    # and count the specific words for each News files
    # write in new directory call 'XXX_count'
    def write_total_dic(self):
        # if directory 'XXX_count' already exists, skip this function
        if not os.path.exists(self.count_Dir):
            print("start create {}".format(self.count_Dir))
            os.makedirs(self.count_Dir)
        else:
            print("{} already exists".format(self.count_Dir))
            print()
            return None

        # get total words in dataset, store in 'word_dic'
        word_dic = {}
        for group_name in os.listdir(self.index_Dir):
            group_path = self.index_Dir + '/' + group_name
            if group_name == ".DS_Store":
                continue
            for news in os.listdir(group_path):
                news_path = group_path + '/' + news
                for line in open(news_path).readlines():
                    word = line.strip('\n')
                    if word in word_dic:
                        word_dic[word] += 1
                    else:
                        word_dic[word] = 1
        new_word_dic = {}
        print("{} different words and counts of words are {}".format(len(word_dic), sum(word_dic.values())))
        print("Get words that occur more than 5 times, so the specific words are")
        # get specific words in dataset store in 'new_word_dic'
        for word in word_dic:
            if word_dic[word] > 4:
                new_word_dic[word] = word_dic[word]
        print("{} different words and counts of words are {}".format(len(new_word_dic), sum(new_word_dic.values())))

        # start create 'XXX_count' directory
        # count each specific words in News files
        # and write them into new files in 'XXX_count' directory
        for group_name in os.listdir(self.index_Dir):
            if not os.path.exists(self.count_Dir):
                os.makedirs(self.count_Dir)
            if group_name == ".DS_Store":
                continue
            group_path = self.index_Dir + '/' + group_name
            count_path = self.count_Dir + '/' + group_name
            if not os.path.exists(count_path):
                os.makedirs(count_path)
            for news in os.listdir(group_path):
                word_dic = {}
                news_path = group_path + '/' + news
                count_news_path = count_path + '/' + news
                for line in open(news_path).readlines():
                    word = line.strip('\n')
                    if word in new_word_dic:
                        if word in word_dic:
                            word_dic[word] += 1
                        else:
                            word_dic[word] = 1
                f = open(count_news_path, 'w')
                for word in word_dic:
                    f.write('{} {}\n'.format(word, word_dic[word]))
                f.close()
            print("Group {} done".format(group_name))
        print("Create {} complete".format(self.count_Dir))
        print()
        return None

    # set n fold cross-validation
    # divide files in 'XXX_count' directory into n parts equally
    # and store them into 'self.news_group'
    def set_cv_fold(self, n, seed_n):
        print("set cv to {}-fold".format(n))

        print("random seed is {}".format(seed_n))
        print()
        random.seed(seed_n)
        self.cv_fold = n
        for group_name in os.listdir(self.count_Dir):
            if group_name == ".DS_Store":
                continue
            self.news_group[group_name] = []
            group_path = self.count_Dir + '/' + group_name
            group_news_list = []
            for news_name in os.listdir(group_path):
                group_news_list.append(news_name)
            random.shuffle(group_news_list)
            self.news_group[group_name] = chunk(group_news_list, self.cv_fold)
        return None

# implementing the Multinomial Naive Bayes
class Multinomial_NB():
    def __init__(self, count_dir):
        self.train_set = {}
        self.test_set = {}
        self.NB_word_P = {}
        self.NB_group_p = {}
        self.count_dir = count_dir
        self.total_word = {}

    # train the data, in given training data set,
    # calculate P(g_j) for each group store in 'self.NB_group_p'
    # and P(w_i│g_j) for each specific word in that group store in 'self.NB_word_P'
    def nb_multinomial_train(self, train_set):
        # read files in training set, get counts of words of each file
        self.train_set = train_set
        for group in self.train_set:
            group_index_path = self.count_dir + '/' + group
            self.NB_word_P[group] = {}
            for news in self.train_set[group]:
                news_index_path = group_index_path + '/' + news
                word_count = {}
                with open(news_index_path, 'r', encoding='utf-8', errors='ignore') as f:
                    while 1:
                        line = f.readline().split()
                        if not line:
                            break
                        if line[0] not in word_count:
                            word_count[line[0]] = int(line[1])
                        else:
                            word_count[line[0]] += int(line[1])
                for word in word_count:
                    if word not in self.NB_word_P[group]:
                        self.NB_word_P[group][word] = word_count[word]
                    else:
                        self.NB_word_P[group][word] += word_count[word]
                    if word not in self.total_word:
                        self.total_word[word] = word_count[word]
                    else:
                        self.total_word[word] += word_count[word]

        # calcualte P(g_j) and P(w_i│g_j)
        # to avoid out range of 'float' value type
        # using math.log with P(g_j) and P(w_i│g_j)
        group_total = {}
        total = sum(self.total_word.values())
        for group in self.NB_word_P:
            group_total[group] = 0
            for word in self.total_word:
                if word not in self.NB_word_P[group]:
                    self.NB_word_P[group][word] = 0
                group_total[group] += self.NB_word_P[group][word]
            #print(group_total[group])
            print("Group {} counts of words are {}".format(group, group_total[group]))
            self.NB_group_p[group] = math.log(float(group_total[group]) / total)
        print("counts of words in total training set is {}".format(total))
        #print(total)
        print()
        for i in self.NB_word_P:
            for word in self.NB_word_P[i]:
                self.NB_word_P[i][word] = math.log(float(self.NB_word_P[i][word] + 0.01) / \
                                          (group_total[i] + 0.01*len(self.NB_word_P[i])))
        return None

    # using given testing set to evaluate the performance of trained model
    def nb_multinomial_test(self, test_set):

        # read the files in testing set, get counts of words in files
        self.test_set = test_set
        test_news = {}
        news_group = {}
        c = 0
        for group in self.test_set:
            group_index_path = self.count_dir + '/' + group
            for news in self.test_set[group]:
                c += 1
                test_news[news] = {}
                news_group[news] = group
                news_index_path = group_index_path + '/' + news
                with open(news_index_path, 'r', encoding='utf-8', errors='ignore') as f:
                    while 1:
                        line = f.readline().split()
                        if not line:
                            break
                        test_news[news][line[0]] = int(line[1])

        # calculate probabilities of every group, and chose largest one
        # since using math.log with P(g_j) and P(w_i│g_j) in train part
        # using operation plus in stead of multiply
        # print Accuracy and F1 Score
        right = 0
        wrong = 0
        y_true = []
        y_pre = []
        for news in test_news:
            largest_p = float("-inf")
            prediction = ""
            for group in self.NB_word_P:
                p = self.NB_group_p[group]
                for word in self.NB_word_P[group]:
                    if word in test_news[news]:
                        p = p + (self.NB_word_P[group][word]*test_news[news][word])
                if p > largest_p:
                    prediction = group
                    largest_p = p
            y_true.append(int(news_group[news]))
            y_pre.append(int(prediction))
            if prediction == news_group[news]:
                right += 1
            else:
                wrong += 1
        #print(right / (wrong + right))
        #print(f1_score(y_true, y_pre, average='macro'))
        print("Accuracy: {}".format(right/(wrong + right)))
        print("F1 Score: {}".format(f1_score(y_true, y_pre, average='macro')))
        return None


# I focus on implementing Multinomial model
# this part is Bernoulli model
# just use for checking if my choice is right
# please skip this part
class Bernoulli_NB():
    def __init__(self, count_dir):
        self.train_set = {}
        self.test_set = {}
        self.NB_word_P = {}
        self.NB_group_p = {}
        self.count_dir = count_dir
        self.total_word = {}

    def nb_bernoulli_train(self, train_set):
        self.train_set = train_set
        files_in_group = {}
        for group in self.train_set:
            group_index_path = self.count_dir + '/' + group
            self.NB_word_P[group] = {}
            files_in_group[group] = 0
            for news in self.train_set[group]:
                files_in_group[group] += 1
                news_index_path = group_index_path + '/' + news
                word_count = {}
                with open(news_index_path, 'r', encoding='utf-8', errors='ignore') as f:
                    while 1:
                        line = f.readline().split()
                        if not line:
                            break
                        if line[0] not in word_count:
                            word_count[line[0]] = 1
                        if line[0] not in self.total_word:
                            self.total_word[line[0]] = 1
                for word in word_count:
                    if word not in self.NB_word_P[group]:
                        self.NB_word_P[group][word] = 1
                    else:
                        self.NB_word_P[group][word] += 1

        total = sum(files_in_group.values())
        for group in self.NB_word_P:
            for word in self.total_word:
                if word not in self.NB_word_P[group]:
                    self.NB_word_P[group][word] = 0
            self.NB_group_p[group] = float(files_in_group[group]) / float(total)

        for group in self.NB_word_P:
            for word in self.NB_word_P[group]:
                self.NB_word_P[group][word] = float(self.NB_word_P[group][word] + 1) \
                                              / float(files_in_group[group] + 2)

    def nb_bernoulli_test(self, test_set):
        self.test_set = test_set
        test_news = {}
        news_group = {}
        for group in self.test_set:
            group_index_path = self.count_dir + '/' + group
            for news in self.test_set[group]:
                test_news[news] = {}
                news_group[news] = group
                news_index_path = group_index_path + '/' + news
                with open(news_index_path, 'r', encoding='utf-8', errors='ignore') as f:
                    while 1:
                        line = f.readline().split()
                        if not line:
                            break
                        test_news[news][line[0]] = 1
        right = 0
        wrong = 0
        for news in test_news:
            largest_p = 0
            prediction = ""
            for group in self.NB_word_P:
                p = self.NB_group_p[group]
                for word in self.NB_word_P[group]:
                    if word in test_news[news]:
                        p = p * self.NB_word_P[group][word]
                    else:
                        p = p * (1 - self.NB_word_P[group][word])
                if p > largest_p:
                    prediction = group
                    largest_p = p
            if prediction == news_group[news]:
                right += 1
            else:
                wrong += 1
        print("Accuracy: {}".format(right / (wrong + right)))
        return None


# split the line and get words in line
# drop the stop word and get word's stem
def get_words(line):
    ps = nltk.PorterStemmer()
    sp = re.compile('[^a-zA-Z]')
    st = nltk.corpus.stopwords.words('english')
    words = [ps.stem(word.lower()) for word in sp.split(line) if len(word) > 0 and word.lower() not in st]
    return words

# split a list into n parts
def chunk(lst,n):
    return [ lst[i::n] for i in range(n) ]



