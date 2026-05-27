import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1: idx + 1 + self.block_size], dtype=torch.long)
        return x, y


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def _generate_mask(self, sz, device):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)
        return mask

    def forward(self, idx):
        b, t = idx.size()
        tok = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        # transformer expects (seq_len, batch, embed)
        tok = tok.transpose(0, 1)
        mask = self._generate_mask(t, idx.device)
        out = self.transformer(tok, mask=mask)
        out = out.transpose(0, 1)
        out = self.ln_f(out)
        logits = self.head(out)
        return logits


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return [stoi[ch] for ch in text]


def decode(indices, itos):
    return ''.join(itos[i] for i in indices)


def generate(model, start_text, stoi, itos, length=200, device='cpu'):
    model.eval()
    indices = torch.tensor([stoi[ch] for ch in start_text], dtype=torch.long, device=device).unsqueeze(0)
    generated = indices.tolist()[0]
    with torch.no_grad():
        for _ in range(length):
            inp = indices[:, -model.block_size:]
            logits = model(inp)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated.append(int(next_id[0, 0].item()))
            indices = torch.cat([indices, next_id], dim=1)
    return decode(generated, itos)


def train(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    data_path = Path(args.data_path)
    assert data_path.exists(), f"Data file {data_path} not found"

    text = data_path.read_text(encoding='utf-8')
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    dataset = CharDataset(data, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = TinyTransformer(vocab_size=len(stoi), block_size=args.block_size, n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training on device: {device}. Vocab size: {len(stoi)}. Data length: {len(data)} chars.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = 0.95 * running_loss + 0.05 * loss.item() if running_loss else loss.item()
            pbar.set_postfix(loss=running_loss)

        sample = generate(model, start_text=args.start_text, stoi=stoi, itos=itos, length=args.sample_length, device=device)
        print(f"\nSample after epoch {epoch}:\n{sample}\n")

        ckpt = {
            'model_state': model.state_dict(),
            'vocab': {'stoi': stoi, 'itos': itos},
            'args': vars(args)
        }
        torch.save(ckpt, args.checkpoint)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='alice_in_wonderland.txt')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--block_size', type=int, default=128)
    p.add_argument('--n_embd', type=int, default=128)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--sample_length', type=int, default=400)
    p.add_argument('--start_text', type=str, default='Alice')
    p.add_argument('--checkpoint', type=str, default='checkpoint.pt')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    train(args)
    print(f"Training finished in {time.time() - start:.1f}s")
