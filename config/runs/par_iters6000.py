# Base mais longa: 6.000 passos em vez de 2.000. Na base, a loss de validação ainda caía no passo 2.000.
# O lr_decay_iters acompanha o max_iters: sem isso, o lr fica parado no min_lr depois do passo 2.000.
max_iters = 6000
lr_decay_iters = 6000
