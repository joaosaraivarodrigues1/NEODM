"""
Avalia um checkpoint do NEODataModel na tarefa de datas.

Para cada linha de um conjunto (teste, por padrão), entrega ao modelo "entrada=",
deixa ele completar até o fim de linha e compara com a data ISO correta.

Uso:
$ python evaluate.py --out_dir=runs/baseline --split=test

Além de imprimir, grava na pasta da run:
- resultado_{split}.json: acertos por parte, parte x tipo e por campo, mais a config da run
- erros_{split}.csv: só as linhas em que o modelo errou
"""
import os
import csv
import json
import pickle

from collections import defaultdict
import torch
from model import NEODataConfig, NEODataModel

# -----------------------------------------------------------------------------
out_dir = 'runs/baseline' # pasta do checkpoint [alterado] era out-datas
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


# -----------------------------------------------------------------------------
# gravar o resultado na pasta da run, para o notebook comparar as runs

def grupo_de(l):
    """Parte do teste, com o anos_fora separado: 2100 (século novo) e 209x (década pouco vista)."""
    if l['parte'] == 'anos_fora':
        return 'anos_fora_2100' if l['ano'] == '2100' else 'anos_fora_209x'
    return l['parte']

def taxa(grupo, acertou):
    """Fração das linhas do grupo em que acertou(linha) é verdadeiro."""
    return sum(acertou(l) for l in grupo) / len(grupo)

def precisao_recall_F(grupo):
    """Precisão e recall da marca F (data inválida), a classe que queremos detectar.
    Olha o último caractere gerado, para medir o veredito mesmo com os dígitos desalinhados."""
    vp = sum(l['previsto'][-1:] == 'F' and l['target'][-1] == 'F' for l in grupo)  # previu F, era F
    fp = sum(l['previsto'][-1:] == 'F' and l['target'][-1] == 'T' for l in grupo)  # previu F, era T
    fn = sum(l['previsto'][-1:] != 'F' and l['target'][-1] == 'F' for l in grupo)  # era F, não previu F
    return {
        'precisao': vp / (vp + fp) if vp + fp else None,  # dos que marcou F, quantos eram F
        'recall': vp / (vp + fn) if vp + fn else None,    # dos que eram F, quantos marcou F
        'vp': vp, 'fp': fp, 'fn': fn,
    }


# a resposta tem posições fixas: AAAA-MM-DDT. Cada campo é um pedaço do texto.
CAMPOS = {'ano': slice(0, 4), 'mes': slice(5, 7), 'dia': slice(8, 10), 'marca': slice(10, 11)}

por_grupo = defaultdict(list)
for l in linhas:
    por_grupo[grupo_de(l)].append(l)


resultado = {
    'run': os.path.basename(os.path.normpath(out_dir)),
    'split': split,
    'config': checkpoint['config'],
    'params': sum(p.numel() for p in model.parameters()),
    'best_val_loss': float(checkpoint['best_val_loss']),
    'iter_do_checkpoint': checkpoint['iter_num'],
    'acerto': taxa(linhas, lambda l: l['previsto'] == l['target']),
    'por_parte': {},
    'parte_tipo': {},
    'por_campo': {},
    'marca_F': {},
}
for nome, grupo in sorted(por_grupo.items()):
    resultado['por_parte'][nome] = taxa(grupo, lambda l: l['previsto'] == l['target'])
    resultado['por_campo'][nome] = {
        campo: taxa(grupo, lambda l: l['previsto'][fatia] == l['target'][fatia])
        for campo, fatia in CAMPOS.items()
    }
    resultado['marca_F'][nome] = precisao_recall_F(grupo)
    resultado['parte_tipo'][nome] = {
        tipo: taxa([l for l in grupo if l['tipo'] == tipo], lambda l: l['previsto'] == l['target'])
        for tipo in sorted({l['tipo'] for l in grupo})
    }


with open(os.path.join(out_dir, f'resultado_{split}.json'), 'w', encoding='utf-8') as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

colunas = ['input', 'previsto', 'target', 'parte', 'tipo', 'formato', 'ruido', 'ano', 'mes', 'dia']
with open(os.path.join(out_dir, f'erros_{split}.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=colunas, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(erros)
print(f"\nresultado gravado em {out_dir}: resultado_{split}.json e erros_{split}.csv ({len(erros):,} erros)")


