import re
from Bio import Entrez
import time
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import ray
Entrez.email = "benyamin.jami76@gmail.com"  # Replace with your email address

# @ray.remote(num_cpus=2)
def fetch_blast_data(protein_seq, i):
    # Protein sequence to search with BLAST
    sequence = '>sequence1\n{0}'.format(protein_seq)

    # Perform the BLAST search with BLOSUM62 matrix and E-value threshold of 1e-50
    print(i)
    result_handle = NCBIWWW.qblast(program="blastp", database="nr_clustered", sequence=sequence, expect=0.7, matrix_name="BLOSUM62", alignments=0, descriptions=100)

    # Parse the BLAST results
    blast_records = NCBIXML.parse(result_handle)

    sequences = []
    # Iterate through the results and fetch the protein sequences
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                if hsp.expect < 1e-50:
                    # Extract the accession number from the alignment title
                    accession = alignment.accession

                    # Fetch the protein sequence using Entrez
                    handle = Entrez.efetch(db="protein", id=accession, rettype="fasta", retmode="text")
                    protein_sequence = handle.read().strip()
                    handle.close()

                    # Print the protein sequence
                    sequences.append(protein_sequence)
                    

    # Close the result handle after parsing the results
    result_handle.close()
    return (i, sequences)



# ray.init(num_cpus=4)

futures = []
antigens = []
with open('antigens.txt', 'r') as f:
    for i, line in enumerate(f):
        antigens.append(line)

        # fetch_blast_data(line.replace('\n',''), i)
antigens = list(set(antigens))
for i, ag in enumerate(antigens):
    # futures.append(fetch_blast_data.remote(ag.replace('\n',''), i))
    fetch_blast_data(ag.replace('\n',''), i)


while len(futures) > 0:
    done_ids, futures = ray.wait(futures, num_returns=1)
    for done_id in done_ids:
        i, done_task = ray.get(done_id)
        with open(str(i)+'.fasta', 'w') as f:
            for seq in done_task:
                f.write(seq+'\n')
        print(f'Remaining {len(futures)}. Finished {str(done_task)}')
    time.sleep(1.0)

