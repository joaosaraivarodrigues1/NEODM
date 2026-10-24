"""
Prepara o dataset de datas para o NEODataModel.

Lê dados.csv, divide as DATAS em treino/validação/teste, tokeniza por caractere e grava:
- train.bin, val.bin, test.bin: uma linha por exemplo, todas com o mesmo tamanho (padding)
- train.csv, val.csv, test.csv: as linhas originais de cada conjunto, usadas na avaliação
- meta.pkl: vocabulário e tamanhos
"""
import os
import csv
import pickle
import random
import numpy as np

pasta = os.path.dirname(__file__)
SEP = '='       # separa a entrada da resposta
FIM = '\n'      # marca o fim da resposta
PAD = '<pad>'   # enchimento até todas as linhas terem o mesmo tamanho

# 1. ler o CSV gerado por gerar_dados.py
with open(os.path.join(pasta, 'dados.csv'), encoding='utf-8', newline='') as f:
    linhas = list(csv.DictReader(f))
print(f"linhas lidas: {len(linhas):,}")

# 2. dividir POR DATA: os 10 formatos de uma data vão todos para o mesmo conjunto
datas = sorted({l['target'] for l in linhas})
random.Random(1337).shuffle(datas)  # embaralha sempre igual (seed fixa)
n_train = int(0.8 * len(datas))
n_val = int(0.1 * len(datas))
conjunto = {}
for i, d in enumerate(datas):
    if i < n_train:
        conjunto[d] = 'train'
    elif i < n_train + n_val:
        conjunto[d] = 'val'
    else:
        conjunto[d] = 'test'

# 3. montar o texto de cada exemplo e o vocabulário de caracteres
textos = [l['input'] + SEP + l['target'] + FIM for l in linhas]
chars = sorted(set(''.join(textos)))
itos = {0: PAD}                                   # id 0 reservado para o padding
itos.update({i + 1: c for i, c in enumerate(chars)})
stoi = {c: i for i, c in itos.items()}
vocab_size = len(itos)
seq_len = max(len(t) for t in textos)            # tamanho da maior linha
print(f"vocabulário: {vocab_size} tokens | maior sequência: {seq_len} caracteres")

# 4. codificar cada conjunto, completar com padding e gravar
for split in ['train', 'val', 'test']:
    indices = [i for i, l in enumerate(linhas) if conjunto[l['target']] == split] # cria 3 listas temporarias
    arr = np.zeros((len(indices), seq_len), dtype=np.uint16)  # começa tudo com 0 = PAD
    for linha, i in enumerate(indices):
        ids = [stoi[c] for c in textos[i]]
        arr[linha, :len(ids)] = ids
    arr.tofile(os.path.join(pasta, f'{split}.bin'))

    with open(os.path.join(pasta, f'{split}.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=linhas[0].keys())
        writer.writeheader()
        writer.writerows(linhas[i] for i in indices)

    n_datas = len({linhas[i]['target'] for i in indices})
    print(f"{split}: {len(indices):,} exemplos de {n_datas:,} datas")

# 5. salvar o vocabulário e os tamanhos para o train.py e o evaluate.py
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'seq_len': seq_len,
    'pad_id': stoi[PAD],
    'sep_id': stoi[SEP],
    'fim_id': stoi[FIM],
}
with open(os.path.join(pasta, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# conferência: mostra o primeiro exemplo do treino, caractere por caractere
primeiro = np.fromfile(os.path.join(pasta, 'train.bin'), dtype=np.uint16)[:seq_len]
print("exemplo:", [itos[i] for i in primeiro])
