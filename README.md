# MiniGPT
# Criação de um mini GPT em Python
# =====================================================
# MiniGPT do zero 
# =====================================================

# ----- Célula 1: Bibliotecas e dispositivo -----
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("Dispositivo:", device)


# ----- Célula 2: Texto utilizado para o treinamento -----
text = """
Era uma vez um gato que vivia em uma pequena casa.
O gato gostava de caminhar pelo jardim.
Todos os dias o gato observava os pássaros.
Quando o sol aparecia, o gato saía de casa.
O cachorro também vivia naquela casa.
O cachorro corria pelo jardim e brincava com o gato.
O gato gostava de peixe e o cachorro gostava de carne.
À noite, os dois animais voltavam para casa.
O gato dormia perto da janela e o cachorro dormia perto da porta.
No dia seguinte, tudo começava novamente.
""" * 200

print("Número de caracteres:", len(text))
print()
print(text[:500])


# ----- Célula 3: Construção do vocabulário -----
chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return ''.join(itos[i] for i in ids)


print("Vocabulário:")
print(chars)
print()
print("|V| =", vocab_size)


# ----- Célula 4: Testando a tokenização -----
example = "gato"

encoded = encode(example)
decoded = decode(encoded)

print("Texto:", example)
print("Tokens:", encoded)
print("Reconstrução:", decoded)


# ----- Célula 5: Transformando todo o texto em um tensor -----
data = torch.tensor(
    encode(text),
    dtype=torch.long
)

print("Shape:", data.shape)
print("Primeiros 50 tokens:")
print(data[:50])


# ----- Célula 6: Separando treinamento e validação -----
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Tokens de treino:", len(train_data))
print("Tokens de validação:", len(val_data))


# ----- Célula 7: Hiperparâmetros -----
d = 32   # dimensão do embedding
T = 64   # tamanho máximo do contexto
L = 2    # número de blocos Transformer
h = 4    # número de cabeças de atenção

batch_size = 32

print("|V| =", vocab_size)
print("d =", d)
print("T =", T)
print("L =", L)
print("h =", h)
print("batch_size =", batch_size)


# ----- Célula 8: Construção dos lotes -----
def get_batch(split):

    source = train_data if split == "train" else val_data

    starts = torch.randint(
        0,
        len(source) - T - 1,
        (batch_size,)
    )

    X = torch.stack([
        source[i:i + T]
        for i in starts
    ])

    Y = torch.stack([
        source[i + 1:i + T + 1]
        for i in starts
    ])

    return X, Y


# ----- Célula 9: Visualizando um batch -----
X, Y = get_batch("train")

print("X.shape =", X.shape)
print("Y.shape =", Y.shape)

print()
print("Primeira entrada:")
print(decode(X[0].tolist()))

print()
print("Primeiro alvo:")
print(decode(Y[0].tolist()))


# ----- Célula 10: Uma cabeça de atenção -----
class Head(nn.Module):

    def __init__(self, d, head_dim, T):
        super().__init__()

        self.key = nn.Linear(
            d,
            head_dim,
            bias=False
        )

        self.query = nn.Linear(
            d,
            head_dim,
            bias=False
        )

        self.value = nn.Linear(
            d,
            head_dim,
            bias=False
        )

        mask = torch.tril(
            torch.ones(T, T)
        )

        self.register_buffer(
            "mask",
            mask
        )

        self.head_dim = head_dim

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        T_current = x.size(1)

        mask = self.mask[:T_current, :T_current]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        A = F.softmax(
            scores,
            dim=-1
        )

        out = A @ V

        return out


# ----- Célula 11: Multi-Head Attention -----
class MultiHeadAttention(nn.Module):

    def __init__(self, d, h, T):
        super().__init__()

        assert d % h == 0

        head_dim = d // h

        self.heads = nn.ModuleList([
            Head(
                d,
                head_dim,
                T
            )
            for _ in range(h)
        ])

        self.projection = nn.Linear(
            d,
            d
        )

    def forward(self, x):

        outputs = [
            head(x)
            for head in self.heads
        ]

        out = torch.cat(
            outputs,
            dim=-1
        )

        out = self.projection(out)

        return out


# ----- Célula 12: Feed-Forward Network -----
class FeedForward(nn.Module):

    def __init__(self, d):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d)
        )

    def forward(self, x):
        return self.net(x)


# ----- Célula 13: Bloco Transformer -----
class TransformerBlock(nn.Module):

    def __init__(self, d, h, T):
        super().__init__()

        self.ln1 = nn.LayerNorm(d)

        self.attention = MultiHeadAttention(
            d,
            h,
            T
        )

        self.ln2 = nn.LayerNorm(d)

        self.ffn = FeedForward(d)

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


# ----- Célula 14: Construindo o MiniGPT -----
class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d,
        T,
        L,
        h
    ):
        super().__init__()

        self.T = T

        self.token_embedding = nn.Embedding(
            vocab_size,
            d
        )

        self.position_embedding = nn.Embedding(
            T,
            d
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d,
                h,
                T
            )
            for _ in range(L)
        ])

        self.ln_final = nn.LayerNorm(d)

        self.lm_head = nn.Linear(
            d,
            vocab_size
        )

    def forward(self, idx):

        B, T_current = idx.shape

        token_emb = self.token_embedding(idx)

        positions = torch.arange(
            T_current,
            device=idx.device
        )

        position_emb = self.position_embedding(
            positions
        )

        x = token_emb + position_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits


# ----- Célula 15: Instanciando o modelo -----
model = MiniGPT(
    vocab_size=vocab_size,
    d=d,
    T=T,
    L=L,
    h=h
).to(device)

print(model)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Número de parâmetros: {num_parameters:,}"
)


# ----- Célula 16: Primeiro forward pass -----
X, Y = get_batch("train")

X = X.to(device)
Y = Y.to(device)

logits = model(X)

print("X:", X.shape)
print("logits:", logits.shape)
print("Y:", Y.shape)


# ----- Célula 17: Calculando a loss inicial -----
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.reshape(-1, vocab_size),
    Y.reshape(-1)
)

print("Loss inicial:", loss.item())


# ----- Célula 18: Otimizador -----
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

max_steps = 3000


# ----- Célula 19: Treinamento -----
for step in range(max_steps):

    X, Y = get_batch("train")

    X = X.to(device)
    Y = Y.to(device)

    logits = model(X)

    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 200 == 0:
        print(
            f"step {step:4d} | "
            f"loss {loss.item():.4f}"
        )


# ----- Célula 20: Função de geração -----
@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens
):
    model.eval()

    for _ in range(max_new_tokens):

        # Mantém somente os últimos T tokens
        idx_cond = idx[:, -model.T:]

        # Predições para todas as posições
        logits = model(idx_cond)

        # Queremos somente a última posição
        logits = logits[:, -1, :]

        # Converte logits em probabilidades
        probs = F.softmax(
            logits,
            dim=-1
        )

        # Sorteia o próximo token
        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        # Adiciona o novo token à sequência
        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

    model.train()

    return idx


# ----- Célula 21: Investigando a próxima previsão -----
prompt = "ca"

idx = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)

with torch.no_grad():

    logits = model(idx)

    # Pegamos somente a última posição:
    # é ela que prevê o próximo caractere.
    next_logits = logits[:, -1, :]

    probs = F.softmax(
        next_logits,
        dim=-1
    )

    probs = probs[0]

    top_probs, top_ids = torch.topk(
        probs,
        k=10
    )

    for p, token_id in zip(top_probs, top_ids):

        ch = itos[token_id.item()]

        print(
            repr(ch),
            f"{p.item():.4f}"
        )


# ----- Célula 22: Geração de texto -----
prompt = "O g"

idx = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)

generated = generate(
    model,
    idx,
    max_new_tokens=300
)

result = decode(
    generated[0].tolist()
)

print(result)
