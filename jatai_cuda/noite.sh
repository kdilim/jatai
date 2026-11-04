#!/bin/bash
# Treino longo da noite: espera ate HORA, roda INT8 e depois float32 (mesmo modelo, dados e lotes), e escreve o
# resumo. Saida em logs/noite_*.  Uso: setsid nohup bash noite.sh > logs/noite_lancador.log 2>&1 &
HORA="00:30"
L=24; FF=2048; LOTE=8; PASSOS=62500; EVAL=500        # 62.500 x 1024 = 64 M tokens por modo (~0,65 epoca)
PY=${PY:-python3}                                    # interpretador com PyTorch (ex.: PY=.venv/bin/python3)
cd "$(dirname "$0")"
alvo=$(date -d "$HORA" +%s); agora=$(date +%s)
[ "$alvo" -le "$agora" ] && alvo=$(date -d "tomorrow $HORA" +%s)
echo "$(date '+%F %T') esperando ate $(date -d @$alvo '+%F %T')"
sleep $((alvo - agora))
echo "$(date '+%F %T') GPU antes de comecar:"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
for modo in int8 float; do
    echo "$(date '+%F %T') inicio $modo"
    taskset -c 0-1 nice -n 19 $PY noite.py $modo $L $FF $LOTE $PASSOS $EVAL > logs/noite_${modo}_saida.txt 2>&1
    echo "$(date '+%F %T') fim $modo (codigo $?)"
done
$PY noite_resumo.py $L $FF
echo "$(date '+%F %T') pronto"
