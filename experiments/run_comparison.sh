#!/bin/bash

# Script para rodar experimentos comparativos
# Uso: ./experiments/run_comparison.sh

set -e

# Configurações
OUTPUT_DIR="experiments/output"
LOG_FILE="training_log.csv"

# Criar pasta de saída
mkdir -p $OUTPUT_DIR

echo "=== Iniciando Experimentos ==="

# --- Configuração 1: Rede Grande (Baseline) ---
echo ">> Rodando Configuração 1: Rede Grande [784, 2048, 1024, 10]"
rm -f $LOG_FILE
cargo run --bin mlp-gpu --release -- --arch "784,2048,1024,10"
mv $LOG_FILE "$OUTPUT_DIR/run1.csv"
echo ">> Configuração 1 finalizada."

# --- Configuração 2: Rede Pequena ---
echo ">> Rodando Configuração 2: Rede Pequena [784, 128, 64, 10]"
rm -f $LOG_FILE
cargo run --bin mlp-gpu --release -- --arch "784,128,64,10"
mv $LOG_FILE "$OUTPUT_DIR/run2.csv"
echo ">> Configuração 2 finalizada."

# --- Gerar Gráficos ---
echo ">> Gerando gráficos de comparação..."
cargo run --bin plot -- "$OUTPUT_DIR/run1.csv" "Rede Grande (2048/1024)" "$OUTPUT_DIR/run2.csv" "Rede Pequena (128/64)"

# Mover imagem para pasta de output se não estiver lá
if [ -f "comparison_plot.png" ]; then
    mv comparison_plot.png "$OUTPUT_DIR/"
fi

echo "=== Experimentos Concluídos ==="
echo "Resultados salvos em: $OUTPUT_DIR/"
echo "Gráfico: $OUTPUT_DIR/comparison_plot.png"
