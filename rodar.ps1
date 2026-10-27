# Roda mais uma execução de um experimento (família) e avalia.
#
# Uso:  .\rodar.ps1 <familia>        por exemplo:  .\rodar.ps1 abl_relu
#
# - A config do experimento é config/runs/<familia>.py, só com o que muda em relação a config/datas.py.
# - Cada chamada cria uma run nova, runs/<familia>_s<N>, com o próximo N livre (1, 2, 3...).
# - A seed é 1336 + N: a run _s1 de qualquer família usa 1337, a _s2 usa 1338, e assim por diante.
# - Para ter mais runs da mesma família, é só chamar de novo. O notebook runs.ipynb agrega todas.
param([Parameter(Mandatory = $true)][string]$familia)

$config = "config/runs/$familia.py"
if (-not (Test-Path $config)) {
    Write-Host "não existe ${config}: crie o arquivo com o que muda neste experimento"
    exit 1
}

# próximo N livre
$n = 1
while (Test-Path "runs/${familia}_s$n") { $n++ }
$seed = 1336 + $n
$out = "runs/${familia}_s$n"

Write-Host "== $out (seed $seed)"
python train.py config/datas.py $config --seed=$seed --out_dir=$out
if ($LASTEXITCODE -ne 0) {
    Write-Host "o treino falhou: apague a pasta $out antes de rodar de novo"
    exit $LASTEXITCODE
}
python evaluate.py --out_dir=$out
