#!/bin/bash

# =============================================================================
# run-env.sh — Setup segreti in 1Password + genera .env.tpl
#
# Questo script RICHIEDE login interattivo (op signin) perche' deve SCRIVERE
# nel vault. Il Service Account (read-only) serve solo per l'uso quotidiano
# con "op run".
# =============================================================================

# Pattern per riconoscere i segreti (case-insensitive match sul nome variabile)
SECRET_PATTERNS="KEY|PASSWORD|SECRET|TOKEN|CREDENTIAL|USER"

# Determina il nome del campo 1Password in base al tipo di segreto
field_name_for() {
  local var="$1"
  case "$var" in
    *API_KEY*|*APIKEY*)  echo "api-key" ;;
    *SECRET_KEY*|*SECRET*)  echo "secret" ;;
    *PASSWORD*|*PASSWD*)  echo "password" ;;
    *TOKEN*)  echo "token" ;;
    *CREDENTIAL*)  echo "credential" ;;
    *USER*|*USERNAME*)  echo "username" ;;
    *)  echo "password" ;;
  esac
}

# --- Prerequisiti ---
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
  echo "Non sei dentro una repo git."
  exit 1
fi

if ! command -v op &> /dev/null; then
  echo "1Password CLI non installato. Installalo da https://developer.1password.com/docs/cli"
  exit 1
fi

if [ -n "$OP_SERVICE_ACCOUNT_TOKEN" ]; then
  echo "Questo script richiede login interattivo (non Service Account)."
  echo "Rimuovi il token e usa op signin:"
  echo "  unset OP_SERVICE_ACCOUNT_TOKEN"
  echo "  op signin"
  echo "  bash run-env.sh"
  exit 1
fi

if ! op account list &> /dev/null; then
  echo "Non sei autenticato a 1Password. Esegui: op signin"
  exit 1
fi
echo "Autenticato con sessione interattiva"

REPO_URL=$(git remote get-url origin)
REPO_NAME=$(basename "$REPO_URL" .git)

if [ ! -f .env ]; then
  echo ".env non trovato nella repo"
  exit 1
fi

# --- Crea o trova il vault ---
VAULT_ID=$(op vault get "$REPO_NAME" --format=json 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$VAULT_ID" ]; then
  echo "Creazione vault '$REPO_NAME'..."
  VAULT_JSON=$(op vault create "$REPO_NAME" --format=json 2>&1)
  if [ $? -ne 0 ]; then
    echo "Impossibile creare il vault:"
    echo "  $VAULT_JSON"
    exit 1
  fi
  VAULT_ID=$(echo "$VAULT_JSON" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

  if [ -z "$VAULT_ID" ]; then
    echo "Impossibile estrarre ID del vault:"
    echo "  $VAULT_JSON"
    exit 1
  fi
fi
echo "Vault: $REPO_NAME ($VAULT_ID)"

# --- Salva segreti nel vault ---
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
  echo "Nessun segreto trovato nel .env (pattern: $SECRET_PATTERNS)"
  echo "Il .env.tpl conterra' solo valori in chiaro."
else
  for i in "${!SECRET_NAMES[@]}"; do
    NAME="${SECRET_NAMES[$i]}"
    VALUE="${SECRET_VALUES[$i]}"

    # Elimina item esistente (ignora errore se non esiste)
    op item delete "$NAME" --vault "$VAULT_ID" 2>/dev/null || true

    FIELD=$(field_name_for "$NAME")
    op item create \
      --vault "$VAULT_ID" \
      --category "Password" \
      --title "$NAME" \
      "$FIELD=$VALUE"

    echo "  -> $NAME"
  done

  echo "$SECRETS_COUNT segreti salvati in 1Password (vault: $REPO_NAME)"
fi

# --- Genera .env.tpl ---
> .env.tpl
while IFS= read -r line; do
  if [[ "$line" =~ ^#.*$ || -z "$line" ]]; then
    echo "$line" >> .env.tpl
    continue
  fi

  VAR_NAME=$(echo "$line" | cut -d'=' -f1)
  VAR_VALUE=$(echo "$line" | cut -d'=' -f2-)

  if echo "$VAR_NAME" | grep -qiE "$SECRET_PATTERNS"; then
    FIELD=$(field_name_for "$VAR_NAME")
    echo "$VAR_NAME=op://$REPO_NAME/$VAR_NAME/$FIELD" >> .env.tpl
  else
    # Strip inline comments so they don't become part of the variable value
    CLEAN_VALUE=$(echo "$VAR_VALUE" | sed 's/[[:space:]]*#.*$//')
    echo "$VAR_NAME=$CLEAN_VALUE" >> .env.tpl
  fi
done < .env

# Aggiunge .env al .gitignore
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
  echo ".env" >> .gitignore
fi

echo ""
echo "Done! $SECRETS_COUNT segreti salvati in 1Password, .env.tpl generato."

# --- Offri creazione Service Account ---
SA_NAME="Futura-Dev"

# Controlla se il SA esiste già:
# - Windows: legge OP_SERVICE_ACCOUNT_TOKEN a livello User dal registro
# - Linux/macOS: cerca la riga in ~/.profile
EXISTING_TOKEN=""
if command -v powershell.exe >/dev/null 2>&1; then
  EXISTING_TOKEN=$(powershell.exe -NoProfile -Command \
    "[Environment]::GetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN', 'User')" 2>/dev/null | tr -d '\r')
else
  grep -q "OP_SERVICE_ACCOUNT_TOKEN" "$HOME/.profile" 2>/dev/null && EXISTING_TOKEN="found"
fi

if [ -n "$EXISTING_TOKEN" ]; then
  echo ""
  echo "Service Account '$SA_NAME' trovato."
  echo "Aggiungi il vault '$REPO_NAME' dal portale 1Password:"
  echo "  1password.com -> Service Accounts -> $SA_NAME -> Vaults -> Add vault -> $REPO_NAME"
  echo ""
  if command -v powershell.exe >/dev/null 2>&1; then
    echo "Per resettare il token:"
    echo "  [Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN', \$null, 'User')"
  else
    echo "Per resettare il token: rimuovi la riga OP_SERVICE_ACCOUNT_TOKEN da ~/.profile."
  fi
else
echo ""
echo "Vuoi creare il Service Account '$SA_NAME' per evitare op signin in futuro? (y/N)"
read -r CREATE_SA

if [[ "$CREATE_SA" =~ ^[Yy]$ ]]; then
  echo "Creazione Service Account '$SA_NAME'..."
  SA_TOKEN_RAW=$(op service-account create "$SA_NAME" \
    --vault "${VAULT_ID}:read_items" \
    2>&1)
  SA_EXIT=$?

  if [ $SA_EXIT -eq 0 ]; then
    echo ""
    echo "Service Account creato!"
    echo ""
    echo "SALVA QUESTO TOKEN ORA - non sara' piu' visibile."

    # Extract token string
    SA_TOKEN=$(printf "%s" "$SA_TOKEN_RAW" | grep -oE '(ops_|sa_)[A-Za-z0-9._-]+' | head -1)

    if [ -n "$SA_TOKEN" ]; then
      # Persist token for Windows (PowerShell) if available
      if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "[Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN', '$SA_TOKEN', 'User')" >/dev/null 2>&1 \
          && echo "OP_SERVICE_ACCOUNT_TOKEN impostata a livello User (Windows). Apri una nuova shell." \
          || echo "Impossibile impostare la variabile utente su Windows via PowerShell."
      else
        # Persist for POSIX shells
        PROFILE_FILE="$HOME/.profile"
        if [ -w "$PROFILE_FILE" ] || [ ! -e "$PROFILE_FILE" ]; then
          printf "\n# 1Password Service Account token\nexport OP_SERVICE_ACCOUNT_TOKEN='%s'\n" "$SA_TOKEN" >> "$PROFILE_FILE"
          echo "OP_SERVICE_ACCOUNT_TOKEN aggiunta a $PROFILE_FILE. Apri una nuova shell."
        else
          echo "Non ho i permessi per scrivere $PROFILE_FILE."
        fi
      fi

      echo ""
      echo "Token: $SA_TOKEN"
    else
      echo "Non sono riuscito ad estrarre il token. Output completo:"
      echo ""
      echo "$SA_TOKEN_RAW"
      echo ""
      echo "Copia il token e impostalo manualmente:"
      echo "  PowerShell: [Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN','PASTE_TOKEN','User')"
    fi
  else
    echo "Errore nella creazione del Service Account:"
    echo "  $SA_TOKEN_RAW"
  fi
fi
fi  # end: SA non esistente

echo ""
echo "Per avviare l'app:"
echo "  op run --env-file=.env.tpl -- uv run uvicorn app.main:app --port 8000 --reload"
