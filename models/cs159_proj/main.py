from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2
from nltk.translate.api import AlignedSent
from nltk.metrics import precision, recall

from files import files
from alignments import french_alignments

ITERATIONS = 9

CORPUS_SIZES = [20, 100, 1000, 10000]

for size in CORPUS_SIZES:
    bitext1 = []
    bitext2 = []

    with (
        open(files['fr-en']['fr'][size]) as f,
        open(files['fr-en']['en'][size]) as e,
    ):
        for f_line, e_line in zip(f, e):
            f_sent = f_line.strip().split()
            e_sent = e_line.strip().split()
            bitext1.append(AlignedSent(f_sent, e_sent))
            bitext2.append(AlignedSent(f_sent, e_sent))

    ibm1 = IBMModel1(bitext1, ITERATIONS)
    ibm2 = IBMModel2(bitext2, ITERATIONS)

    count = [0, 0]
    total_precision: list[float] = [0, 0]
    total_recall: list[float] = [0, 0]

    for i, alignment in enumerate(french_alignments):
        p1 = precision(alignment, bitext1[i].alignment)
        r1 = recall(alignment, bitext1[i].alignment)
        p2 = precision(alignment, bitext2[i].alignment)
        r2 = recall(alignment, bitext2[i].alignment)
        if p1 is not None and r1 is not None:
            total_precision[0] += p1
            total_recall[0] += r1
            count[0] += 1
        if p2 is not None and r2 is not None:
            total_precision[1] += p2
            total_recall[1] += r2
            count[1] += 1




    print("CORPUS SIZE:", size)
    print("average precision (ibm1):", total_precision[0] / count[0])
    print("average recall (ibm1):", total_recall[0] / count[0])
    print("average precision (ibm2):", total_precision[1] / count[1])
    print("average recall (ibm2):", total_recall[1] / count[1])
