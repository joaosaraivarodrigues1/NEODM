# Ablação: embedding de entrada e lm_head com pesos separados.
# O weight tying economiza parâmetros com vocabulários grandes (50 mil no GPT-2). Aqui são 57 tokens.
weight_tying = False
