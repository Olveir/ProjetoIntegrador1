import torch
import torch.nn as nn
from torch.optim import AdamW
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from tqdm import tqdm
import numpy as np

# Configurações
num_classes_total = 10
batch_size = 16
epochs = 10
learning_rate = 3e-5
max_length = 256
csv_file = "classificacao_simccit.csv"

# Tokenizador e modelo em português
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenizer.add_tokens(["[CTX]", "[NOCTX]"])
model = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=num_classes_total)
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Mapeamento das categorias
categorias_interacoes = {
    "APR": 0, "EMP": 1, "FAC": 2, "INF": 3, "INT": 4,
    "OVT": 5, "REC": 6, "SREL": 7, "SREF": 8, "REP": 9
}

# Leitura e preparação dos dados
df = pd.read_csv(csv_file)
df = df.dropna(subset=["Fala_Terapeuta", "Categoria"])
df["Fala_Cliente"] = df["Fala_Cliente"].fillna("")

def montar_texto(row):
    if row['Fala_Cliente'].strip():
        return f"[CTX] Cliente: {row['Fala_Cliente']} Terapeuta: {row['Fala_Terapeuta']}", 1
    else:
        return f"[NOCTX] Terapeuta: {row['Fala_Terapeuta']}", 0

df[['Interacao', 'Tem_Contexto']] = df.apply(montar_texto, axis=1, result_type='expand')
dataset = list(zip(df["Interacao"].tolist(), df["Categoria"].tolist(), df['Tem_Contexto'].tolist()))
random.shuffle(dataset)

# Split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Cálculo dos pesos das classes
labels = [categorias_interacoes[r] for _, r, _ in dataset]
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
pesos_tensor = torch.tensor(pesos, dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=pesos_tensor)

# Dataset customizado
class InteracaoDataset(Dataset):
    def __init__(self, data, tokenizer, label_map):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        texto, rotulo, contexto = self.data[idx]
        inputs = self.tokenizer(texto, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(self.label_map[rotulo])
        item['contexto'] = torch.tensor(contexto)
        return item

# Loaders
train_dataset = InteracaoDataset(train_data, tokenizer, categorias_interacoes)
test_dataset = InteracaoDataset(test_data, tokenizer, categorias_interacoes)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Otimizador e agendador
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
total_steps = len(train_loader) * epochs
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# Treinamento com early stopping
best_loss = float("inf")
patience, trigger = 2, 0

print("Iniciando treinamento...\n")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}", leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items() if k != 'contexto'}
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Loss médio na época {epoch+1}: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger = 0
        torch.save(model.state_dict(), "melhor_modelo_pt.pt")
    else:
        trigger += 1
        if trigger >= patience:
            print("Early stopping ativado.")
            break

# Avaliação
model.load_state_dict(torch.load("melhor_modelo_pt.pt"))
model.eval()
all_preds, all_labels, all_contexts = [], [], []

with torch.no_grad():
    for batch in test_loader:
        contexto = batch['contexto']
        batch = {k: v.to(device) for k, v in batch.items() if k != 'contexto'}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_contexts.extend(contexto.numpy())

# Relatórios
idx_to_label = {v: k for k, v in categorias_interacoes.items()}
nomes_categorias = [idx_to_label[i] for i in sorted(idx_to_label)]

print("\nRelatório de Classificação Geral:")
print(classification_report(all_labels, all_preds, target_names=nomes_categorias, digits=3))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=nomes_categorias, yticklabels=nomes_categorias, cmap="Blues")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão Geral")
plt.tight_layout()
plt.show()

# Relatório com/sem contexto
com_ctx = [(y, p) for y, p, c in zip(all_labels, all_preds, all_contexts) if c == 1]
sem_ctx = [(y, p) for y, p, c in zip(all_labels, all_preds, all_contexts) if c == 0]

for nome, grupo in [("Com Contexto", com_ctx), ("Sem Contexto", sem_ctx)]:
    if len(grupo) == 0:
        print(f"\nNenhum exemplo '{nome.lower()}'.")
        continue
    y_ref, y_pd = zip(*grupo)
    print(f"\nRelatório de Classificação - {nome}:")
    print(classification_report(y_ref, y_pd, target_names=nomes_categorias, digits=3))