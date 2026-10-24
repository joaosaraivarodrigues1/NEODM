import csv
from datetime import date, timedelta

MESES = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]
MESES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun",
               "jul", "ago", "set", "out", "nov", "dez"]

def todas_as_datas(inicio, fim):
    """Gera todos os dias entre inicio e fim, inclusive."""
    d = inicio
    while d <= fim:
        yield d
        d += timedelta(days=1)

FORMATOS = {
    "extenso":        lambda d: f"{d.day} de {MESES[d.month-1]} de {d.year}",
    "extenso_zero":   lambda d: f"{d.day:02d} de {MESES[d.month-1]} de {d.year}",
    "extenso_caps":   lambda d: f"{d.day} de {MESES[d.month-1].capitalize()} de {d.year}",
    "extenso_upper":  lambda d: f"{d.day} DE {MESES[d.month-1].upper()} DE {d.year}",
    "abreviado":      lambda d: f"{d.day} {MESES_ABREV[d.month-1]} {d.year}",
    "abreviado_pt":   lambda d: f"{d.day} de {MESES_ABREV[d.month-1]}. de {d.year}",
    "barra":          lambda d: f"{d.day:02d}/{d.month:02d}/{d.year}",
    "barra_sem_zero": lambda d: f"{d.day}/{d.month}/{d.year}",
    "hifen":          lambda d: f"{d.day:02d}-{d.month:02d}-{d.year}",
    "ponto":          lambda d: f"{d.day:02d}.{d.month:02d}.{d.year}",
}

def gerar_csv(caminho, inicio, fim):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "target", "formato", "ano", "mes", "dia"])
        for d in todas_as_datas(inicio, fim):
            for nome, render in FORMATOS.items():
                writer.writerow([render(d), d.isoformat(), nome, d.year, d.month, d.day])

if __name__ == "__main__":
    import os
    caminho = os.path.join(os.path.dirname(__file__), "dados.csv")
    gerar_csv(caminho, date(1900, 1, 1), date(2100, 12, 31))

