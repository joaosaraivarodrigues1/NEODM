"""
Gera o dataset de datas: dados.csv, com a divisão em treino, validação e teste já decidida.

Cada linha é uma data escrita em algum formato (input) e a resposta em ISO seguida de
T (data válida) ou F (data inválida), por exemplo 2023-12-25T ou 2023-04-31F.
O modelo transcreve o que está escrito mesmo quando a data não existe e só no fim dá o veredito.

Colunas: input, target, formato, ruido, tipo, valida, conjunto, parte, ano, mes, dia
- conjunto: train, val ou test
- parte (só no test): comum, ruido, formato, anos_dentro ou anos_fora
- tipo: valida, fronteira, dia_perto, dia_longe, mes_perto ou mes_longe
"""
import os
import csv
import calendar
import random
from datetime import date
from collections import Counter

# -----------------------------------------------------------------------------
# 0. decisões do experimento: tudo o que define o dataset fica aqui
SEED = 1337
ANOS = range(1900, 2101)
ANOS_DENTRO = {1905, 1925, 1945, 1965, 1985, 2005, 2025, 2045, 2065, 2085}  # nunca vistos, dentro do intervalo
ANOS_FORA = set(range(2091, 2101))   # nunca vistos, depois do último ano do treino
FORMATO_RESERVADO = 'ponto'          # só aparece no teste
N_DIA_IMPOSSIVEL = 10_000            # chaves com dia 00 ou de 32 a 99
N_MES_IMPOSSIVEL = 30_000            # chaves com mês 00 ou de 13 a 99; 3x o dia, porque só usam os 3 formatos numéricos
PROP_PERTO = 0.5                     # metade das inválidas fáceis perto do limite (evita o atalho do 1º dígito)
REPETIR_FRONTEIRA = 7                # sobreamostragem da fronteira, só no treino
PROP_RUIDO = 0.20                    # fração das linhas de treino e validação com ruído
# -----------------------------------------------------------------------------

sorteio = random.Random(SEED)
pasta = os.path.dirname(os.path.abspath(__file__))


MESES = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]
MESES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun",
               "jul", "ago", "set", "out", "nov", "dez"]

UNIDADES = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove",
            "dez", "onze", "doze", "treze", "catorze", "quinze", "dezesseis", "dezessete",
            "dezoito", "dezenove"]
DEZENAS = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta",
           "oitenta", "noventa"]

def extenso(n):
    """Escreve um dia de 0 a 99 por extenso. O dia 1 é 'primeiro', como se fala."""
    if n == 1:
        return "primeiro"
    if n < 20:
        return UNIDADES[n]
    dezena, unidade = divmod(n, 10)
    return DEZENAS[dezena] if unidade == 0 else f"{DEZENAS[dezena]} e {UNIDADES[unidade]}"

    
# os 10 formatos anteriores + dia_extenso. Os que usam nome de mês só servem para mês de 1 a 12.
FORMATOS = {
    "extenso":        lambda a, m, d: f"{d} de {MESES[m-1]} de {a}",
    "extenso_zero":   lambda a, m, d: f"{d:02d} de {MESES[m-1]} de {a}",
    "extenso_caps":   lambda a, m, d: f"{d} de {MESES[m-1].capitalize()} de {a}",
    "extenso_upper":  lambda a, m, d: f"{d} DE {MESES[m-1].upper()} DE {a}",
    "abreviado":      lambda a, m, d: f"{d} {MESES_ABREV[m-1]} {a}",
    "abreviado_pt":   lambda a, m, d: f"{d} de {MESES_ABREV[m-1]}. de {a}",
    "dia_extenso":    lambda a, m, d: f"{extenso(d)} de {MESES[m-1]} de {a}",
    "barra":          lambda a, m, d: f"{d:02d}/{m:02d}/{a}",
    "barra_sem_zero": lambda a, m, d: f"{d}/{m}/{a}",
    "hifen":          lambda a, m, d: f"{d:02d}-{m:02d}-{a}",
    "ponto":          lambda a, m, d: f"{d:02d}.{m:02d}.{a}",
}
NUMERICOS = {"barra", "barra_sem_zero", "hifen", "ponto"}

def formatos_permitidos(tipo):
    """Formatos usados no treino, na validação e no teste comum: todos menos o reservado."""
    nomes = [f for f in FORMATOS if f != FORMATO_RESERVADO]
    if tipo in ('mes_perto', 'mes_longe'):
        nomes = [f for f in nomes if f in NUMERICOS]   # não existe nome para o mês 13
    return nomes

def alvo(a, m, d, tipo):
    """Resposta certa: a data escrita em ISO, mesmo que não exista, mais T ou F."""
    return f"{a:04d}-{m:02d}-{d:02d}" + ('T' if tipo == 'valida' else 'F')



# ruído recuperável: suja a forma sem mudar a data. Nunca troca dígito por dígito.
TROCAS = {'o': '0', 'O': '0', '0': 'O', 'l': '1', '1': 'l', 'S': '5', '5': 'S', 'B': '8', '8': 'B'}
SEM_ACENTO = {'ç': 'c', 'Ç': 'C', 'ê': 'e'}

def ruido(texto):
    """Faz uma alteração sorteada no texto. Devolve (texto alterado, nome da alteração)."""
    pos_troca = [i for i, c in enumerate(texto) if c in TROCAS]
    pos_acento = [i for i, c in enumerate(texto) if c in SEM_ACENTO]
    pos_letra = [i for i, c in enumerate(texto) if c.isalpha()]
    pos_espaco = [i for i, c in enumerate(texto) if c == ' ']
    opcoes = ['espaco'] + ['troca'] * bool(pos_troca) + ['acento'] * bool(pos_acento) \
             + ['caixa'] * bool(pos_letra)
    nome = sorteio.choice(opcoes)
    if nome == 'troca':
        i = sorteio.choice(pos_troca)
        return texto[:i] + TROCAS[texto[i]] + texto[i+1:], nome
    if nome == 'acento':
        i = sorteio.choice(pos_acento)
        return texto[:i] + SEM_ACENTO[texto[i]] + texto[i+1:], nome
    if nome == 'caixa':
        i = sorteio.choice(pos_letra)
        return texto[:i] + texto[i].swapcase() + texto[i+1:], nome
    # espaço extra: dobra um espaço que já existe, ou põe um no começo ou no fim
    if pos_espaco:
        i = sorteio.choice(pos_espaco)
        return texto[:i] + ' ' + texto[i:], nome
    return (' ' + texto if sorteio.random() < 0.5 else texto + ' '), nome

    
# -----------------------------------------------------------------------------
# 1. chaves: quais (ano, mes, dia) existem no experimento, e de que tipo
chaves = {}
for a in ANOS:
    for m in range(1, 13):
        ultimo = calendar.monthrange(a, m)[1]   # 28, 29, 30 ou 31; já sabe do bissexto
        for d in range(1, ultimo + 1):
            chaves[(a, m, d)] = 'valida'
        for d in range(ultimo + 1, 32):
            chaves[(a, m, d)] = 'fronteira'     # 29/02 em ano não bissexto, 30/02, 31/04...

n = 0
while n < N_DIA_IMPOSSIVEL:
    if sorteio.random() < PROP_PERTO:
        dia, tipo = sorteio.choice([0] + list(range(32, 40))), 'dia_perto'
    else:
        dia, tipo = sorteio.randint(40, 99), 'dia_longe'
    chave = (sorteio.choice(ANOS), sorteio.randint(1, 12), dia)
    if chave not in chaves:
        chaves[chave] = tipo
        n += 1

n = 0
while n < N_MES_IMPOSSIVEL:
    if sorteio.random() < PROP_PERTO:
        mes, tipo = sorteio.choice([0] + list(range(13, 20))), 'mes_perto'
    else:
        mes, tipo = sorteio.randint(20, 99), 'mes_longe'
    chave = (sorteio.choice(ANOS), mes, sorteio.randint(1, 31))
    if chave not in chaves:
        chaves[chave] = tipo
        n += 1

# -----------------------------------------------------------------------------
# 2. divisão: cada chave vai inteira para um conjunto (todos os seus formatos juntos)
divisao = {}
for chave in chaves:
    a = chave[0]
    if a in ANOS_DENTRO:
        divisao[chave] = ('test', 'anos_dentro')
    elif a in ANOS_FORA:
        divisao[chave] = ('test', 'anos_fora')
    else:
        r = sorteio.random()
        if r < 0.8:
            divisao[chave] = ('train', '-')
        elif r < 0.9:
            divisao[chave] = ('val', '-')
        else:
            divisao[chave] = ('test', 'comum')


            
# -----------------------------------------------------------------------------
# 3. linhas: cada chave escrita nos formatos permitidos, conforme o seu conjunto
linhas = []

def nova_linha(chave, tipo, formato, texto, ruido_nome, conjunto, parte):
    a, m, d = chave
    linhas.append({
        'input': texto, 'target': alvo(a, m, d, tipo), 'formato': formato,
        'ruido': ruido_nome, 'tipo': tipo, 'valida': 'T' if tipo == 'valida' else 'F',
        'conjunto': conjunto, 'parte': parte, 'ano': a, 'mes': m, 'dia': d,
    })

for chave, tipo in chaves.items():
    conjunto, parte = divisao[chave]
    textos_da_chave = set()
    for formato in formatos_permitidos(tipo):
        texto = FORMATOS[formato](*chave)
        if texto in textos_da_chave:            # ex.: extenso e extenso_zero são iguais do dia 10 em diante
            continue
        textos_da_chave.add(texto)
        if conjunto in ('train', 'val'):
            copias = REPETIR_FRONTEIRA if (conjunto == 'train' and tipo == 'fronteira') else 1
            for _ in range(copias):
                if sorteio.random() < PROP_RUIDO:
                    sujo, r = ruido(texto)
                    nova_linha(chave, tipo, formato, sujo, r, conjunto, parte)
                else:
                    nova_linha(chave, tipo, formato, texto, 'nenhum', conjunto, parte)
        elif parte == 'comum':
            nova_linha(chave, tipo, formato, texto, 'nenhum', 'test', 'comum')
            sujo, r = ruido(texto)              # a mesma chave e o mesmo formato, com ruído
            nova_linha(chave, tipo, formato, sujo, r, 'test', 'ruido')
        else:                                   # anos_dentro ou anos_fora
            nova_linha(chave, tipo, formato, texto, 'nenhum', 'test', parte)
    if parte == 'comum':                        # o formato reservado: só para as chaves do teste comum
        texto = FORMATOS[FORMATO_RESERVADO](*chave)
        nova_linha(chave, tipo, FORMATO_RESERVADO, texto, 'nenhum', 'test', 'formato')

# -----------------------------------------------------------------------------
# 4. conferências: se alguma falhar, o script para
def existe(a, m, d):
    """Segunda opinião sobre a validade, por outro caminho (datetime em vez de calendar)."""
    try:
        date(a, m, d)
        return True
    except ValueError:
        return False

reservados = ANOS_DENTRO | ANOS_FORA
conjuntos_da_chave = {}
for l in linhas:
    if l['conjunto'] in ('train', 'val'):
        assert l['ano'] not in reservados, f"ano reservado fora do teste: {l}"
        assert l['formato'] != FORMATO_RESERVADO, f"formato reservado fora do teste: {l}"
    assert l['target'][-1] == l['valida'], f"marca do alvo diferente de valida: {l}"
    assert (l['valida'] == 'T') == existe(l['ano'], l['mes'], l['dia']), f"rótulo errado: {l}"
    conjuntos_da_chave.setdefault((l['ano'], l['mes'], l['dia']), set()).add(l['conjunto'])
assert all(len(s) == 1 for s in conjuntos_da_chave.values()), "uma chave caiu em dois conjuntos"

# -----------------------------------------------------------------------------
# 5. gravar e resumir
caminho = os.path.join(pasta, 'dados.csv')
with open(caminho, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=linhas[0].keys())
    writer.writeheader()
    writer.writerows(linhas)
print(f"{len(linhas):,} linhas gravadas em {caminho}")

TIPOS = ['valida', 'fronteira', 'dia_perto', 'dia_longe', 'mes_perto', 'mes_longe']
grupos = [('train', '-'), ('val', '-'), ('test', 'comum'), ('test', 'ruido'),
          ('test', 'formato'), ('test', 'anos_dentro'), ('test', 'anos_fora')]
contagem = Counter((l['conjunto'], l['parte'], l['tipo']) for l in linhas)
print(f"\n{'grupo':<18}{'linhas':>9}" + ''.join(f"{t:>11}" for t in TIPOS))
for c, p in grupos:
    total = sum(contagem[(c, p, t)] for t in TIPOS)
    nome = c if p == '-' else f"{c}/{p}"
    print(f"{nome:<18}{total:>9,}" + ''.join(f"{contagem[(c, p, t)] / total:>11.1%}" for t in TIPOS))

maior = max(len(l['input']) + 1 + len(l['target']) + 1 for l in linhas)   # entrada + '=' + alvo + '\n'
print(f"\nmaior sequência: {maior} caracteres")


