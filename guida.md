# Gestione Secrets con 1Password

## Installazione 1Password CLI

### Windows

Installa dal Microsoft Store:

```
ms-windows-store://pdp/?ProductId=9NBLGGH4NNS1
```

Oppure via **winget**:

```powershell
winget install AgileBits.1Password.CLI
```

### macOS

```bash
brew install 1password-cli
```

### Linux

```bash
curl -sS https://downloads.1password.com/linux/keys/1password.asc | \
  sudo gpg --dearmor --output /usr/share/keyrings/1password-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/1password-archive-keyring.gpg] https://downloads.1password.com/linux/debian/$(dpkg --print-architecture) stable main" | \
  sudo tee /etc/apt/sources.list.d/1password.list

sudo apt update && sudo apt install 1password-cli
```

Verifica l'installazione:

```bash
op --version
```

---

## File creati

| File | Scopo |
|------|-------|
| `.env.tpl` | Template con riferimenti `op://Team/ArXivFuturaSearch/*` — committabile, zero segreti |
| `run.sh` | `op run` + `docker compose up` — produzione |
| `run-dev.sh` | `op run` + `uvicorn --reload` — sviluppo locale |

---

## Setup per il team (da fare una volta)

1. **Installare 1Password CLI** (vedi sopra)

2. **Login**:
   ```bash
   op signin
   ```

3. **Creare l'item** nel vault **Team** → Nuovo item "ArXivFuturaSearch" con questi campi:
   - `OPENROUTER_API_KEY`
   - `POSTGRES_PASSWORD`
   - `REDIS_PASSWORD`
   - `SECRET_KEY`

---

## Utilizzo

```bash
# Sviluppo locale
./run-dev.sh

# Docker Compose (detached)
./run.sh -d

# Test con segreti iniettati
./run-dev.sh pytest

# Qualsiasi comando con segreti
./run-dev.sh python script.py
```
