import torch
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embed a perturbed sentence -- before embedding, we append a perturbation to the sentence's token embeddings
def embed_perturbed_text(model: SentenceTransformer, sentence: str, perturbations: torch.Tensor):    
    # Embed the sentence's tokens (shape (seq_len, h))
    # Initial embedding matrix of the model
    E = model[0].auto_model.embeddings.word_embeddings.weight
    tokens = model.tokenizer(sentence)["input_ids"][0]
    token_embs = E[tokens].unsqueeze(0)
    # Concatenate w/perturbations (shape (seq_len + m, h))
    perturbed_sentence_emb = torch.cat([token_embs, perturbations.T])
    # Create an attention mask
    attn_mask = torch.ones(1, perturbed_sentence_emb.shape[0], device=device)
    # Run the core module on the input embeddings
    outputs = model[0].auto_model.forward(inputs_embeds=perturbed_sentence_emb.unsqueeze(0), attention_mask=attn_mask)
    # Apply the final pooling layer
    outputs = model[1].forward({"token_embeddings": outputs["last_hidden_state"], "attention_mask": attn_mask})

    return outputs["sentence_embedding"]

