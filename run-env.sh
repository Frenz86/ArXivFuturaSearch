#!/bin/bash

# Pattern per riconoscere i segreti (case-insensitive match sul nome variabile)
SECRET_PATTERNS="KEY|PASSWORD|SECRET|TOKEN|CREDENTIAL"

# Check siamo in una repo git
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
  echo "❌ Non sei dentro una repo git."
  exit 1
fi

# Check op installato
if ! command -v op &> /dev/null; then
  echo "❌ 1Password CLI non installato. Installalo da https://developer.1password.com/docs/cli"
  exit 1
fi

# Check op autenticato
if ! op account list &> /dev/null; then
  echo "❌ Non sei autenticato a 1Password. Esegui: op signin"
  exit 1
fi

REPO_URL=$(git remote get-url origin)
REPO_NAME=$(basename "$REPO_URL" .git)

# Check .env esiste
if [ ! -f .env ]; then
  echo "❌ .env non trovato nella repo"
  exit 1
fi

# Separa segreti da config
SECRET_NAMES=()
SECRET_VALUES=()
SECRETS_COUNT=0

while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
  VAR_NAME=$(echo "$line" | cut -d'=' -f1)
  VAR_VALUE=$(echo "$line" | cut -d'=' -f2-)

  if echo "$VAR_NAME" | grep -qiE "$SECRET_PATTERNS"; then
    SECRET_NAMES+=("$VAR_NAME")
    SECRET_VALUES+=("$VAR_VALUE")
    ((SECRETS_COUNT++))
  fi
done < .env

if [ "$SECRETS_COUNT" -eq 0 ]; then
  echo "⚠️  Nessun segreto trovato nel .env (pattern: $SECRET_PATTERNS)"
  echo "   Il .env.tpl conterrà solo valori in chiaro."
else
  # Crea il vault (ignora errore se esiste già)
  op vault create "$REPO_NAME" 2>/dev/null

  # Crea un item separato per ogni segreto
  for i in "${!SECRET_NAMES[@]}"; do
    NAME="${SECRET_NAMES[$i]}"
    VALUE="${SECRET_VALUES[$i]}"

    op item delete "$NAME" --vault "$REPO_NAME" 2>/dev/null
    op item create \
      --vault "$REPO_NAME" \
      --category "Password" \
      --title "$NAME" \
      "password=$VALUE"

    echo "  🔑 $NAME"
  done

  echo "🔐 $SECRETS_COUNT segreti salvati in 1Password (vault: $REPO_NAME)"
fi

# Genera .env.tpl: segreti → op://, config → valore in chiaro
> .env.tpl
while IFS= read -r line; do
  # Preserva commenti e righe vuote
  if [[ "$line" =~ ^#.*$ || -z "$line" ]]; then
    echo "$line" >> .env.tpl
    continue
  fi

  VAR_NAME=$(echo "$line" | cut -d'=' -f1)
  VAR_VALUE=$(echo "$line" | cut -d'=' -f2-)

  if echo "$VAR_NAME" | grep -qiE "$SECRET_PATTERNS"; then
    echo "$VAR_NAME=op://$REPO_NAME/$VAR_NAME/password" >> .env.tpl
  else
    echo "$VAR_NAME=$VAR_VALUE" >> .env.tpl
  fi
done < .env

# Aggiunge .env al .gitignore
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
  echo ".env" >> .gitignore
fi

echo "✅ Done! $SECRETS_COUNT segreti salvati in 1Password, .env.tpl generato."
echo "   Per usarlo: op run --env-file=.env.tpl -- uv run uvicorn app.main:app --port 8000 --reload"
