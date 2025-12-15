import pandas as pd
import assignment1_functions as af

kmtest = pd.read_csv('kmtest.csv', sep=r'\s+',encoding='utf-8-sig', header=None, dtype=float)
kmtest_normalized = (kmtest - kmtest.mean()) / kmtest.std()

for K in range(2,6):
    centers, labels = af.simplekmeans(kmtest, K)
    af.kmtest_plot(kmtest.values, centers, labels, f'kmtest-with-K-value-{K}.png')
    centers_n, labels_n = af.simplekmeans(kmtest_normalized, K)
    af.kmtest_plot(kmtest_normalized.values, centers_n, labels_n, f'kmtest-normalized-with-K-value-{K}.png')