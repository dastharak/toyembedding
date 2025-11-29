#Embedding-as-Logits (Embedding dimension = V)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime as dt

# ----------------- Config -----------------
seed = 41
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

V = 20              # Vocab size for demo
epochs = 50
lr = 0.045 # big lr
use_gpu = False
device = 'cuda' if torch.cuda.is_available() & use_gpu else 'cpu'
print(f"device:{device}")
# ----------------- Toy dataset -----------------
# cyclic bigrams: i -> (i+1) % V
# We assume that the input tokens 0..19 represent some real characters
inputs = torch.arange(V, dtype=torch.long)
targets = (inputs + 1) % V
# Print the inputs and targets
print(f"\n______________________")
print(f"|{'Inputs'.center(10)}|{'Targets'.center(10)}|")
print(f"______________________")
for inp,targ in zip(inputs,targets):
    print(f"|{str(inp.item()).center(10,)}|{str(targ.item()).center(10)}|")
print(f"______________________")
# Treat as a single batch of size V (simplicity)
inputs = inputs.to(device)
targets = targets.to(device)

# ----------------- Model -----------------
class EmbAsLogits(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # embedding: maps token -> vector of length vocab_size (used as logits)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # initialize small random weights from normal dist.
        nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.01)

    def printEmbeddingTable(self,):
        print("=== Basic Token Embedding Table Summary ===")
        print(f"     Shape: {self.token_embedding_table.weight.shape}")
        print(f"    Device: {self.token_embedding_table.weight.device}")
        print(f"     Dtype: {self.token_embedding_table.weight.dtype}")
        print(f"RequirGrad: {self.token_embedding_table.weight.requires_grad}")
        print(f"      Norm: {torch.norm(self.token_embedding_table.weight).item():.6f}")
        print(f"      Mean: {self.token_embedding_table.weight.mean().item():.6f}")
        print(f"       Std: {self.token_embedding_table.weight.std().item():.6f}")
        print(f"       Min: {self.token_embedding_table.weight.min().item():.6f}")
        print(f"       Max: {self.token_embedding_table.weight.max().item():.6f}")
        # some pretty-print params
        torch.set_printoptions(precision=4, sci_mode=False, linewidth=200, threshold=10_000)
        np.set_printoptions(precision=4, suppress=True, linewidth=200)
                
        if V <= 5 : #for small matrix print the whole thing
            with torch.no_grad():
                print(self.token_embedding_table.weight.data[0:5])
        else:
            token_ids = torch.tensor([0, 1, 2, 3, 4],device=device) # show first 5 token_ids
            print(f"\n{'__'*10}{'__'*(40)}")
            print(f"|{'Token ID'.center(10)}|{'Weights'.center(95)}")
            print(f"{'__'*10}{'__'*(40)}")
            torch.set_printoptions(precision=6,edgeitems=3,linewidth=100,sci_mode=False)
            for tokid in token_ids:
                embed_vect = self.token_embedding_table(tokid)
                print(f"|{str(tokid.item()).center(10,)}|{embed_vect.detach()[0:3]}...{embed_vect.detach()[-4:-1]}")

    def forward(self, x):
        # x: (N,) -> logits: (N, V)
        logits = self.token_embedding_table(x)  # gather rows
        return logits

embedding = EmbAsLogits(V)
model = embedding.to(device)
embedding.printEmbeddingTable()

# before training
print(f"After initializing the model randomly")
with torch.no_grad():
    final_logits = model(inputs)
    final_pred = final_logits.argmax(dim=1)
    print(f"\n|{'Token ID'.center(10)}|{'PredNext'.center(10)}|{'Target'.center(10)}|")
    print(f"{'__'*16}")
    for i in range(V):
        print(f"|{str(i).center(10)}|{str(final_pred[i].item()).center(10)}|{str(targets[i].item()).center(10)}|")

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# parameter count
param_count = sum(p.numel() for p in model.parameters())
print(f"\nModel : Embedding-as-Logits, parameter count: {param_count:,}  (≈ V*V = {V*V:,})")
st = dt.now()
print(f"Training started at {st} for {epochs} epochs")
# ----------------- Training -----------------
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(inputs)  # (V, V)
    loss = F.cross_entropy(logits, targets)  # softmax+CE
    loss.backward()
    optimizer.step()
    print('.',end='')
    if epoch % 20 == 0 or epoch <= 5:
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == targets).float().mean().item()
        print(f"\nEpoch {epoch:3d}  loss={loss.item():.6f}  acc={acc*100:5.2f}%",end='')

print(f"\nTraining ended at {dt.now()} time taken {dt.now()-st}")
embedding.printEmbeddingTable()

# After Training
print(f"Final Model")
with torch.no_grad():
    final_logits = model(inputs)
    final_pred = final_logits.argmax(dim=1)
    print(f"\n|{'Token ID'.center(10)}|{'PredNext'.center(10)}|{'Target'.center(10)}|")
    print(f"{'__'*16}")
    for i in range(V):
        print(f"|{str(i).center(10)}|{str(final_pred[i].item()).center(10)}|{str(targets[i].item()).center(10)}|")
