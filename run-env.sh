#!/bin/bash

# =============================================================================
# run-env-v2.sh — Setup segreti in 1Password + genera .env.tpl
#
# Modalità SA          : export OP_SERVICE_ACCOUNT_TOKEN='ops_...' prima di eseguire
# Modalità interattiva : op signin, poi bash run-env-v2.sh
#
# Il vault non viene ricreato se esiste già (riesecuzione sicura).
# =============================================================================

SECRET_PATTERNS="KEY|PASSWORD|SECRET|TOKEN|CREDENTIAL|USER"

SA_NAME="Futura-Dev"
ADMIN_EMAIL="administration@futuraaigroup.com"

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
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "Non sei dentro una repo git."
  exit 1
fi

if ! command -v op &>/dev/null; then
  echo "1Password CLI non installato. Installalo da https://developer.1password.com/docs/cli"
  exit 1
fi

# --- Rilevamento modalità: SA o interattiva ---
# BOOTSTRAPPED_TOKEN=true significa: token letto dal vault con sessione interattiva ancora attiva.
# In quel caso il token NON viene esportato subito, così possiamo usare la sessione
# interattiva per condividere il vault con l'account del dev prima di passare a SA mode.
BOOTSTRAPPED_TOKEN=false
DEV_EMAIL="dev.futuraai@gmail.com"
SA_TOKEN_FROM_VAULT=""

if [ -n "$OP_SERVICE_ACCOUNT_TOKEN" ]; then
  MODE="sa"
  echo "Modalità: Service Account"
else
  # Token non impostato: tenta di recuperarlo dal vault FuturaDev tramite login interattivo
  if ! op account list &>/dev/null; then
    echo "OP_SERVICE_ACCOUNT_TOKEN non impostato e nessun login attivo."
    echo "Esegui: op signin"
    exit 1
  fi

  echo "OP_SERVICE_ACCOUNT_TOKEN non impostato. Recupero dal vault FuturaDev..."
  SA_TOKEN_FROM_VAULT=$(op read "op://FuturaDev/OP-SA-Token/credential" 2>/dev/null)

  if [ -n "$SA_TOKEN_FROM_VAULT" ]; then
    MODE="sa"
    BOOTSTRAPPED_TOKEN=true

    echo "Token SA recuperato. Modalità: Service Account (account: $DEV_EMAIL)"
    echo ""
    echo "Per non ripetere op signin in futuro, aggiungi al tuo profilo:"
    echo "  export OP_SERVICE_ACCOUNT_TOKEN=\$(op read \"op://FuturaDev/OP-SA-Token/credential\")"
    # Il token NON viene esportato ancora: serve la sessione interattiva per la condivisione vault
  else
    MODE="interactive"
    echo "Modalità: account interattivo (token SA non trovato nel vault FuturaDev)"
  fi
fi

REPO_URL=$(git remote get-url origin)
REPO_NAME=$(basename "$REPO_URL" .git)

if [ ! -f .env ]; then
  echo ".env non trovato nella repo"
  exit 1
fi

# --- Crea o trova il vault ---
# In bootstrap: usa il token SA inline (non esportato) così il SA risulta proprietario del vault.
if [ "$BOOTSTRAPPED_TOKEN" = true ]; then
  VAULT_JSON_GET=$(OP_SERVICE_ACCOUNT_TOKEN="$SA_TOKEN_FROM_VAULT" op vault get "$REPO_NAME" --format=json 2>/dev/null)
else
  VAULT_JSON_GET=$(op vault get "$REPO_NAME" --format=json 2>/dev/null)
fi

VAULT_ID=""
if [ -n "$VAULT_JSON_GET" ]; then
  if command -v jq &>/dev/null; then
    VAULT_ID=$(echo "$VAULT_JSON_GET" | jq -r '.id' 2>/dev/null)
  elif command -v python3 &>/dev/null; then
    VAULT_ID=$(python3 -c "import json,sys; print(json.load(sys.stdin)['id'])" 2>/dev/null <<< "$VAULT_JSON_GET")
  else
    VAULT_ID=$(echo "$VAULT_JSON_GET" | tr -d '\n' | grep -oE '"id": *"[^"]*"' | head -1 | sed 's/.*"id": *"//;s/"//')
  fi
fi

if [ -z "$VAULT_ID" ]; then
  echo "Creazione vault '$REPO_NAME'..."

  if [ "$BOOTSTRAPPED_TOKEN" = true ]; then
    VAULT_JSON=$(OP_SERVICE_ACCOUNT_TOKEN="$SA_TOKEN_FROM_VAULT" op vault create "$REPO_NAME" --format=json 2>&1)
  else
    VAULT_JSON=$(op vault create "$REPO_NAME" --format=json 2>&1)
  fi

  if [ $? -ne 0 ]; then
    echo "Impossibile creare il vault:"
    echo "  $VAULT_JSON"
    exit 1
  fi

  if command -v jq &>/dev/null; then
    VAULT_ID=$(echo "$VAULT_JSON" | jq -r '.id' 2>/dev/null)
  elif command -v python3 &>/dev/null; then
    VAULT_ID=$(python3 -c "import json,sys; print(json.load(sys.stdin)['id'])" 2>/dev/null <<< "$VAULT_JSON")
  else
    VAULT_ID=$(echo "$VAULT_JSON" | tr -d '\n' | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
  fi

  if [ -z "$VAULT_ID" ]; then
    echo "Impossibile estrarre ID del vault:"
    echo "  $VAULT_JSON"
    exit 1
  fi
  VAULT_JUST_CREATED=true
  echo "Vault '$REPO_NAME' creato ($VAULT_ID)"

  # In bootstrap: condividi con il dev e l'admin usando la sessione interattiva ancora attiva
  # (PRIMA di esportare il token SA, che sovrascrive la sessione interattiva)
  if [ "$BOOTSTRAPPED_TOKEN" = true ]; then
    if [ -n "$DEV_EMAIL" ]; then
      SHARE_DEV=$(OP_SERVICE_ACCOUNT_TOKEN="$SA_TOKEN_FROM_VAULT" op vault user grant --vault "$VAULT_ID" --user "$DEV_EMAIL" --permissions="allow_viewing,allow_editing,allow_managing" 2>&1)
      if [ $? -eq 0 ]; then
        echo "Vault condiviso con il tuo account ($DEV_EMAIL)."
      else
        echo "ATTENZIONE: impossibile condividere con $DEV_EMAIL: $SHARE_DEV"
        echo "  Chiedi all'admin di eseguire:"
        echo "  op vault user grant --vault $VAULT_ID --user $DEV_EMAIL --permissions allow_viewing,allow_editing,allow_managing"
      fi
    fi
    SHARE_ADMIN=$(OP_SERVICE_ACCOUNT_TOKEN="$SA_TOKEN_FROM_VAULT" op vault user grant --vault "$VAULT_ID" --user "$ADMIN_EMAIL" --permissions="allow_viewing,allow_editing,allow_managing" 2>&1)
    if [ $? -eq 0 ]; then
      echo "Vault condiviso con l'admin ($ADMIN_EMAIL)."
    else
      echo "ATTENZIONE: impossibile condividere con $ADMIN_EMAIL: $SHARE_ADMIN"
      echo "  Chiedi all'admin di eseguire:"
      echo "  op vault user grant --vault $VAULT_ID --user $ADMIN_EMAIL --permissions allow_viewing,allow_editing,allow_managing"
    fi

    # Ora il token SA può essere esportato per le operazioni sugli items
    export OP_SERVICE_ACCOUNT_TOKEN="$SA_TOKEN_FROM_VAULT"
  fi
else
  VAULT_JUST_CREATED=false
  echo "Vault '$REPO_NAME' già esistente ($VAULT_ID) — skip creazione"

  # Vault esistente: esporta il token SA (sessione interattiva non più necessaria)
  if [ "$BOOTSTRAPPED_TOKEN" = true ]; then
    export OP_SERVICE_ACCOUNT_TOKEN="$SA_TOKEN_FROM_VAULT"
  fi
fi

# --- Condivisione vault con admin (solo in modalità interattiva pura) ---
if [ "$MODE" = "interactive" ]; then
  if op vault user list "$VAULT_ID" --format=json 2>/dev/null | grep -q "$ADMIN_EMAIL"; then
    echo "Admin ($ADMIN_EMAIL) ha già accesso al vault."
  else
    echo ""
    echo "Vuoi condividere il vault '$REPO_NAME' con l'admin ($ADMIN_EMAIL)? (y/N)"
    read -r SHARE_CHOICE
    if [[ "$SHARE_CHOICE" =~ ^[Yy]$ ]]; then
      SHARE_OUTPUT=$(op vault user grant --vault "$VAULT_ID" --user "$ADMIN_EMAIL" --permissions="allow_viewing,allow_editing,allow_managing" 2>&1)
      if [ $? -eq 0 ]; then
        echo "Vault condiviso con l'admin."
      else
        echo "ATTENZIONE: Impossibile condividere il vault:"
        echo "  $SHARE_OUTPUT"
        echo "Condividilo manualmente:"
        echo "  op vault user grant --vault $VAULT_ID --user $ADMIN_EMAIL --permissions allow_viewing,allow_editing,allow_managing"
      fi
    fi
  fi
fi

# --- Avviso SA (solo in modalità interattiva, solo se vault appena creato) ---
# La CLI 1Password non supporta l'assegnazione di un SA a un vault esistente.
# L'unico modo automatico è che il SA crei il vault lui stesso (--can-create-vaults).
if [ "$MODE" = "interactive" ] && [ "$VAULT_JUST_CREATED" = "true" ]; then
  echo ""
  echo "ATTENZIONE: assegna manualmente il vault al Service Account '$SA_NAME':"
  echo "  1password.com -> Service Accounts -> $SA_NAME -> Vaults -> Add vault -> $REPO_NAME"
  echo ""
  echo "Oppure riesegui con il token SA (il vault non verrà ricreato):"
  echo "  export OP_SERVICE_ACCOUNT_TOKEN=\$(op read \"op://FuturaDev/OP-SA-Token/credential\")"
  echo "  bash $(basename "$0")"
fi

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
  echo ""
  echo "Nessun segreto trovato nel .env (pattern: $SECRET_PATTERNS)"
  echo "Il .env.tpl conterra' solo valori in chiaro."
else
  echo ""
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
    echo "$VAR_NAME=$VAR_VALUE" >> .env.tpl
  fi
done < .env

# Aggiunge .env al .gitignore
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
  echo ".env" >> .gitignore
fi

echo ""
echo "Done! $SECRETS_COUNT segreti salvati in 1Password, .env.tpl generato."
echo ""
echo "Per avviare l'app:"
echo "  op run --env-file=.env.tpl -- uv run uvicorn app.main:app --port 8000 --reload"
