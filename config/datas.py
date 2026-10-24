# Treino do NEODataModel na tarefa de datas (data em português -> data ISO).
# Modelo pequeno: a tarefa é simples e a máquina tem uma MX550 com 2 GB.

out_dir = 'out-datas'
dataset = 'datas'

# avaliação
eval_interval = 250       # avalia a cada 250 passos
eval_iters = 20           # lotes usados para estimar a loss em cada avaliação
log_interval = 50
always_save_checkpoint = False  # só salva quando a loss de validação melhora

# dados
gradient_accumulation_steps = 1
batch_size = 256
block_size = 34           # maior sequência (35) menos 1: x e y são a sequência deslocada

# modelo: cerca de 0,1 M de parâmetros
n_layer = 2
n_head = 4
n_embd = 64
dropout = 0.0

# otimização
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000     # igual a max_iters
min_lr = 1e-4             # learning_rate / 10
warmup_iters = 100
beta2 = 0.99
