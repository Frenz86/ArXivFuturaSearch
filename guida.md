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
