from Bio.Align import PairwiseAligner

# Your sequences
sequence1 = "CGATTGACTCTCCACGCTGTCCCTAACCATGACCGTCGAAG"
sequence2 = "CGATTGACTCTCCTTCGACGGTCATGTACTAGATCAGAGG"
sequence3 = "CGATTGACTCTCCCTCTGATCTAGTAGTTAGGACAGCGTG"

# Create a PairwiseAligner object
aligner = PairwiseAligner()

# Perform global alignments
alignments = aligner.align(sequence1, sequence2)
print(alignments[0])

alignments = aligner.align(sequence1, sequence3)
print(alignments[0])

alignments = aligner.align(sequence2, sequence3)
print(alignments[0])
