import CS9417Proj

# prepare the data
a = CS9417Proj.Prepare("20news-18828")
a.read_and_create_index()
a.write_total_dic()

# set cross-validation fold to 10
n = 10
# Here I set random seed to 10
seed = 10
a.set_cv_fold(n, seed)

# using Multinomial Naive Bayes
# for every loop
# using 1 folder for testing, 9 folders for training
for k in range(n):
    train_set = {}
    test_set = {}
    for i in a.news_group:
        train_set[i] = []
        for j in range(len(a.news_group[i])):
            if j == k:
                test_set[i] = a.news_group[i][j]
            else:
                train_set[i] += a.news_group[i][j]
    print("{} start:".format(k+1))
    b = CS9417Proj.Multinomial_NB(a.count_Dir)
    b.nb_multinomial_train(train_set)
    print("{} results: ".format(k+1))
    b.nb_multinomial_test(test_set)
    print()



