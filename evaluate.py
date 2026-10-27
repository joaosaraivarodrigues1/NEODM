"""
Avalia um checkpoint do NEODataModel na tarefa de datas.

Para cada linha de um conjunto (teste, por padrão), entrega ao modelo "entrada=",
deixa ele completar até o fim de linha e compara com a data ISO correta.

Uso:
$ python evaluate.py --out_dir=out-datas --split=test
"""
import os
import csv
import pickle
from collections import defaultdict
import torch
from model import NEODataConfig, NEODataModel

# -----------------------------------------------------------------------------
out_dir = 'out-datas'   # pasta do checkpoint
split = 'test'          # 'train', 'val' ou 'test'
batch_size = 1024       # quantas entradas gerar ao mesmo tempo
device = 'cuda'         # 'cuda' ou 'cpu'
n_exemplos = 5          # quantos exemplos imprimir no final
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# carregar o modelo treinado
checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
model = NEODataModel(NEODataConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.eval()            # desliga o dropout
model.to(device)    

# carregar o vocabulário e as linhas do conjunto
data_dir = os.path.join('data', checkpoint['config']['dataset'])
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
with open(os.path.join(data_dir, f'{split}.csv'), encoding='utf-8', newline='') as f:
    linhas = list(csv.DictReader(f))

# agrupar as linhas pelo tamanho da entrada: num lote, todas precisam ter o mesmo tamanho
grupos = defaultdict(list)
for linha in linhas:
    grupos[len(linha['input'])].append(linha)

# gerar a resposta de cada linha
tamanho_resposta = len('2023-12-25T\n')   # data ISO + marca T/F + fim de linha
with torch.no_grad():
    for tamanho, grupo in grupos.items():
        for i in range(0, len(grupo), batch_size):
            lote = grupo[i:i + batch_size]
            x = torch.tensor([[stoi[c] for c in l['input'] + '='] for l in lote], device=device)
            y = model.generate(x, tamanho_resposta, greedy=True, stop_token=meta['fim_id'])
            for linha, ids in zip(lote, y[:, x.size(1):].tolist()):
                texto = ''.join(itos[t] for t in ids)
                linha['previsto'] = texto.split('\n')[0]   # corta no fim de linha

# medir e mostrar: o acerto geral mistura partes diferentes; o que importa é o acerto por grupo

acertos = sum(l['previsto'] == l['target'] for l in linhas)
print(f"{split}: acerto exato {acertos / len(linhas):.2%} ({acertos:,} de {len(linhas):,})")

for coluna in ['parte', 'tipo']:
    print(f"\npor {coluna}:")
    for valor in sorted({l[coluna] for l in linhas}):
        grupo = [l for l in linhas if l[coluna] == valor]
        acertos = sum(l['previsto'] == l['target'] for l in grupo)
        print(f"  {valor:<12} {acertos / len(grupo):7.2%}  ({len(grupo):,} linhas)")



erros = [l for l in linhas if l['previsto'] != l['target']]
print(f"\n{n_exemplos} exemplos" + (" de erro:" if erros else ":"))
for l in (erros or linhas)[:n_exemplos]:
    print(f"  {l['input']:<42} previsto: {l['previsto']:<13} correto: {l['target']}")
