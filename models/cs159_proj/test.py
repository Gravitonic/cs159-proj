from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2
from nltk.translate.api import AlignedSent
from nltk.metrics import precision, recall

from files import files
from alignments import german_alignments

bitext = []
bitext.append(AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']))
bitext.append(AlignedSent(['das', 'haus', 'ist', 'ja', 'groß'], ['the', 'house', 'is', 'big']))
bitext.append(AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']))
bitext.append(AlignedSent(['das', 'haus'], ['the', 'house']))
bitext.append(AlignedSent(['das', 'buch'], ['the', 'book']))
bitext.append(AlignedSent(['ein', 'buch'], ['a', 'book']))

ibm2 = IBMModel2(bitext, 5)

als0 = AlignedSent(['das', 'haus', 'ist', 'ja', 'groß'], ['the', 'house', 'is', 'big'])
ibm2.align(als0)
print(als0.alignment)

als1 = AlignedSent(['das', 'haus', 'ist', 'ein', 'buch'], ['the', 'house', 'is', 'a', 'book'])
ibm2.align(als1)
print(als1.alignment)
ibm2.align(AlignedSent(['nova', 'wordae'], ['new', 'word']))
