import torch
from typing import List
from sentence_transformers import SentenceTransformer
from utils import embed_perturbed_text
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MagicWordFinder:
    def __init__(self, model: SentenceTransformer, S: List[str], max_reps=16):
        # The embedding model
        self.model = model
        # The corpus S
        self.S = S
        # Compute the embedding of each token and store the result in a matrix E
        self.tokenizer = model.tokenizer
        self.E = model[0].auto_model.embeddings.word_embeddings.weight

        self.cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        # Embed all texts in S
        self.S_embs = model.encode(S, convert_to_tensor=True)
        # Compute the center of mass of the embeddings
        self.e_star = torch.mean(torch.from_numpy(model.encode(S)).to(device), dim=0)
        # How many times should we try to repeat each candidate magic word?
        self.max_reps = max_reps
    
    # Find magic words using alg. 3 from the paper
    def find_magic_words(self, m=1, # the magic words we find are m tokens long
                          k=15, # no. candidates to try for each token position 
                          k_0=10, # no. magic words to return
                          epochs=1, # no. passes over the dataset (in the paper this is always set to 1)
                          ):
        # Initialize the magic word embeddings
        h = self.model.get_sentence_embedding_dimension()
        t_star = torch.zeros((h, m), device=device)
        # Iterate over all sentences in the corpus
        for _ in range(epochs):
            for s_j in self.S:
                t = torch.randn((h, m), device=device, requires_grad=True)
                # Run the model on the perturbed text
                perturbed_text_embs = embed_perturbed_text(self.model, s_j, t)
                # Compute loss -- the loss indicates how close the perturbed text embeddings
                # are to the center of mass of the corpus embeddings, so a higher loss is better
                l_plus = perturbed_text_embs @ self.e_star.unsqueeze(1)
                # Backprop through the loss
                l_plus.backward()
                # Gradient Ascent
                t_star += t.grad 

        # Get k candidates for the positive magic words
        # Each entry contains k candidates for the m-th token
        # shape (k, m)
        cands = torch.topk(self.E @ t_star, k, dim=0).indices.T

        # Narrow down the candidates using alg. 1 from the paper
        return self.__narrow_down(product(*cands.tolist()), k_0)
    
    def __narrow_down(self, cands: product, k_0: int):
        # Scores for each candidate
        cs = []
        # Sentences from the corpus with the magic words appended
        sents = []
        # Convert the candidates from an itertools.product to a list so that we'll be able to iterate over them multiple times
        cands = list(cands)
        h = self.model.get_sentence_embedding_dimension()

        # Precompute embed(s_j + r * t_i) for all sentences s_j, candidates t_i, and 1 <= r <= max_reps
        for s_j in self.S:
            s_j_tag = self.tokenizer(s_j)["input_ids"]
            # Remove the EOS token
            eos_token = s_j_tag[-1]
            s_j_tag = s_j_tag[:-1]
            # Append the candidate magic word t_i r times
            for t_i in cands:
                for r in range(1, self.max_reps + 1):
                    sents.append(self.tokenizer.decode(s_j_tag + list(t_i) * r + [eos_token]))
        # Embed each sentence to get a (len(S) * len(cands) * max_reps, h)-shaped tensor
        sents_emb = self.model.encode(sents, convert_to_tensor=True)
        # Reshape the tensor
        sents_emb = sents_emb.reshape((len(self.S), len(cands), self.max_reps, h))

        # Compute the optimal number of repetitions for each candidate
        for i in range(len(cands)):
            cs.append(max([self.__calc_narrow_down_score(i, r, sents_emb) for r in range(1, self.max_reps + 1)]))

        # Find the indices of the top-k_0 candidates within the candidates
        cands_idxs = torch.topk(torch.tensor(cs), k_0, dim=0).indices.flatten().tolist()
        # Use them to index the candidates
        cands = [list(cands[idx]) for idx in cands_idxs]

        return cands

    def __calc_narrow_down_score(self, i, r, sents_emb):
        score = 0

        # Iterate over each pair (s_j, s_k) of sentences from the corpus
        # and sum the cosine similarity between s_j + t_i * r and s_k
        # where t_i is the candidate magic word and r is the repetition count
        for j in range(len(self.S)):
            for k in range(len(self.S)):
                score += self.cos_sim(sents_emb[j, i, r - 1, :], sents_emb[k, i, r - 1, :])
        
        return score