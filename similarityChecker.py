#!/usr/bin/python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import PyPDF2
import glob
import re
import argparse
import sys


def calc_tfidf_matrix(path):
    # Create Corpus
    corpus = []
    for pdf in glob.glob(path+"/*.pdf"):
        pdfObj = open(pdf, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfObj)
        numPages = pdfReader.numPages
        wholeText = '';
        for i in range(numPages):
            pageObj = pdfReader.getPage(i)
            text = pageObj.extractText()
            text = text.replace('\n', ' ').replace('\r',' ').replace('.', '').replace(',','')
            text = ''.join([i for i in text if not i.isdigit()])
            text = re.sub(r'[^\w]', ' ', text)
            text = " ".join(text.split()) # remove extra whitespace
            text = text.lower()
            wholeText = wholeText + text + ' '
            # print(wholeText)
        corpus.append((pdf, wholeText))

    # Generate TF-IDF Matrix
    vect = TfidfVectorizer()
    tfidf_matrix =  vect.fit_transform([content for pdf, content in corpus])
    return corpus, tfidf_matrix


def find_similar(tfidf_matrix, index, top_n):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]


def check_args(args=None):
    parser = argparse.ArgumentParser(description='Calculate Similarity Index of Documents')
    parser.add_argument('-P', '--path',
                        help='Path to corpus',
                        required=True,
                        )
    parser.add_argument('-T', '--top',
                        help='Define number of top n identical document',
                        type=int,
                        default=10
                        )
    parser.add_argument('-I', '--index',
                        help='Define corpus index',
                        type=int,
                        required=True
                        )
    result = parser.parse_args()
    return result.path, result.top, result.index


def main():
    path, top, index = check_args(sys.argv[1:])
    mat = calc_tfidf_matrix(path)
    corpus = mat[0]
    tfidf_matrix = mat[1]

    print('The selected document is ' + corpus[index][0]);
    print('\n')
    print('======== Similarity Index Results ========')

    for index, score in find_similar(tfidf_matrix, index, top):
        print(score, corpus[index][0])


if __name__ == '__main__':
    main()
