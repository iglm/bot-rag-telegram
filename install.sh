#!/bin/bash
# install.sh — Script de instalación del bot RAG

set -e

echo "🤖 Instalando Bot RAG Telegram..."

# 1. Venv
python3 -m venv venv
source venv/bin/activate

# 2. Dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 3. Systemd service (opcional)
if [ "$1" == "--systemd" ]; then
    echo "⚙️ Configurando systemd..."
    sudo cp bot-docs-indexados.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable bot-docs-indexados
    echo "✅ Servicio configurado. Edita /etc/systemd/system/bot-docs-indexados.service"
    echo "   Luego: sudo systemctl start bot-docs-indexados"
fi

echo ""
echo "✅ Instalación completa!"
echo ""
echo "Próximos pasos:"
echo "  1. export BOT_TOKEN_DOCS=tu_token"
echo "  2. export OPENROUTER_API_KEY=tu_key"
echo "  3. python3 bot_documentos_indexados.py"
