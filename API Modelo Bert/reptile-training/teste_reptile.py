import torch
from transformers import BertTokenizer, BertForSequenceClassification

# === Parâmetros ===
modelo_path = "melhor_modelo_pt.pt"
max_length = 256

# === Categorias ===
categorias_interacoes = {
    "APR": 0, "EMP": 1, "FAC": 2, "INF": 3, "INT": 4,
    "OVT": 5, "REC": 6, "SREL": 7, "SREF": 8, "REP": 9
}
idx_to_categoria = {v: k for k, v in categorias_interacoes.items()}

# === Carregar Tokenizador e Modelo ===
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenizer.add_tokens(["[CTX]", "[NOCTX]"])

model = BertForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=len(categorias_interacoes)
)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(modelo_path, map_location=torch.device('cpu')))
model.eval()

# === Função para montar o texto ===
def montar_texto(fala_cliente, fala_terapeuta):
    if fala_cliente.strip():
        return f"[CTX] Cliente: {fala_cliente} Terapeuta: {fala_terapeuta}"
    else:
        return f"[NOCTX] Terapeuta: {fala_terapeuta}"

# === Loop de Inferência ===
print("\n=== Classificador de Interações Terapêuticas ===")
print("Digite 'sair' como fala do terapeuta para encerrar.\n")

while True:
    fala_cliente = input("Fala do cliente (opcional): ").strip()
    fala_terapeuta = input("Fala do terapeuta: ").strip()

    if fala_terapeuta.lower() == "sair":
        print("Encerrando...")
        break

    texto = montar_texto(fala_cliente, fala_terapeuta)

    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    print(f"\nTexto usado para predição:\n{texto}")
    print(f"👉 Categoria prevista: {idx_to_categoria[pred]}")
    print("-" * 60)
