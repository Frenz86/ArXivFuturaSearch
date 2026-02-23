<#
.SYNOPSIS
    Setup segreti in 1Password + genera .env.tpl

.DESCRIPTION
    Modalita SA          : imposta OP_SERVICE_ACCOUNT_TOKEN prima di eseguire
    Modalita interattiva : op signin, poi .\run-env-v2.ps1

    Il vault non viene ricreato se esiste gia (riesecuzione sicura).
#>

$ErrorActionPreference = "Stop"

$SECRET_PATTERNS     = "KEY|PASSWORD|SECRET|TOKEN|CREDENTIAL|USER"
$SA_NAME             = "Futura-Dev"
$ADMIN_EMAIL         = "administration@futuraaigroup.com"
$DEV_EMAIL           = "dev.futuraai@gmail.com"
$BOOTSTRAPPED_TOKEN  = $false
$saTokenFromVault    = $null

function Get-FieldName {
    param([string]$var)
    switch -Wildcard ($var) {
        "*API_KEY*"    { return "api-key" }
        "*APIKEY*"     { return "api-key" }
        "*SECRET_KEY*" { return "secret" }
        "*SECRET*"     { return "secret" }
        "*PASSWORD*"   { return "password" }
        "*PASSWD*"     { return "password" }
        "*TOKEN*"      { return "token" }
        "*CREDENTIAL*" { return "credential" }
        "*USER*"       { return "username" }
        "*USERNAME*"   { return "username" }
        default        { return "password" }
    }
}

# --- Prerequisiti ---
$null = git rev-parse --is-inside-work-tree 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Non sei dentro una repo git." -ForegroundColor Red; exit 1
}

if (-not (Get-Command op -ErrorAction SilentlyContinue)) {
    Write-Host "1Password CLI non installato. Installalo da https://developer.1password.com/docs/cli" -ForegroundColor Red; exit 1
}

# --- Rilevamento modalita: SA o interattiva ---
if ($env:OP_SERVICE_ACCOUNT_TOKEN) {
    $MODE = "sa"
    Write-Host "Modalita: Service Account" -ForegroundColor Cyan
} else {
    $null = op account list 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "OP_SERVICE_ACCOUNT_TOKEN non impostato e nessun login attivo." -ForegroundColor Red
        Write-Host "Esegui: op signin"
        exit 1
    }

    Write-Host "OP_SERVICE_ACCOUNT_TOKEN non impostato. Recupero dal vault FuturaDev..." -ForegroundColor Cyan
    $ErrorActionPreference = "Continue"
    $saTokenFromVault = op read "op://FuturaDev/OP-SA-Token/credential" 2>$null
    $ErrorActionPreference = "Stop"

    if ($saTokenFromVault) {
        $MODE = "sa"
        $BOOTSTRAPPED_TOKEN = $true
        Write-Host "Token SA recuperato. Modalita: Service Account (account: $DEV_EMAIL)" -ForegroundColor Green
        Write-Host ""
        Write-Host "Per non ripetere op signin in futuro, aggiungi al tuo profilo:" -ForegroundColor Cyan
        Write-Host "  [Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN', (op read 'op://FuturaDev/OP-SA-Token/credential'), 'User')"
    } else {
        $MODE = "interactive"
        Write-Host "Modalita: account interattivo (token SA non trovato nel vault FuturaDev)" -ForegroundColor Yellow
    }
}

$REPO_URL  = git remote get-url origin
$REPO_NAME = [System.IO.Path]::GetFileNameWithoutExtension($REPO_URL)

if (-not (Test-Path .env)) {
    Write-Host ".env non trovato nella repo" -ForegroundColor Red; exit 1
}

# --- Crea o trova il vault ---
$VAULT_ID = $null
$ErrorActionPreference = "Continue"
if ($BOOTSTRAPPED_TOKEN) {
    $env:OP_SERVICE_ACCOUNT_TOKEN = $saTokenFromVault
    $vaultJson = op vault get $REPO_NAME --format=json 2>$null
    $env:OP_SERVICE_ACCOUNT_TOKEN = $null
} else {
    $vaultJson = op vault get $REPO_NAME --format=json 2>$null
}
$ErrorActionPreference = "Stop"

if ($vaultJson) {
    try { $VAULT_ID = ($vaultJson | ConvertFrom-Json).id } catch { $VAULT_ID = $null }
}

# Fallback: op vault get potrebbe fallire se l'account non ha accesso diretto;
# cerca il vault per nome in op vault list
if (-not $VAULT_ID) {
    $ErrorActionPreference = "Continue"
    $allVaultsJson = op vault list --format=json 2>$null
    $ErrorActionPreference = "Stop"
    if ($allVaultsJson) {
        try {
            $found = ($allVaultsJson | ConvertFrom-Json) | Where-Object { $_.name -eq $REPO_NAME } | Select-Object -First 1
            if ($found) { $VAULT_ID = $found.id }
        } catch {}
    }
}

if (-not $VAULT_ID) {
    Write-Host "Creazione vault '$REPO_NAME'..." -ForegroundColor Cyan

    $ErrorActionPreference = "Continue"
    if ($BOOTSTRAPPED_TOKEN) {
        $env:OP_SERVICE_ACCOUNT_TOKEN = $saTokenFromVault
        $createRaw = op vault create $REPO_NAME --format=json 2>&1
        $createExit = $LASTEXITCODE
        $env:OP_SERVICE_ACCOUNT_TOKEN = $null
    } else {
        $createRaw = op vault create $REPO_NAME --format=json 2>&1
        $createExit = $LASTEXITCODE
    }
    $ErrorActionPreference = "Stop"

    if ($createExit -ne 0) {
        Write-Host "Impossibile creare il vault. Errore op:" -ForegroundColor Red
        Write-Host "$createRaw" -ForegroundColor Red
        exit 1
    }
    $VAULT_ID = ($createRaw | ConvertFrom-Json).id
    if (-not $VAULT_ID) {
        Write-Host "Impossibile estrarre ID del vault." -ForegroundColor Red; exit 1
    }
    $VAULT_JUST_CREATED = $true
    Write-Host "Vault '$REPO_NAME' creato ($VAULT_ID)" -ForegroundColor Green

    if ($BOOTSTRAPPED_TOKEN) {
        $ErrorActionPreference = "Continue"

        $env:OP_SERVICE_ACCOUNT_TOKEN = $saTokenFromVault
        $shareDevOut = op vault user grant --vault $VAULT_ID --user $DEV_EMAIL --permissions "allow_viewing,allow_editing,allow_managing" 2>&1
        $shareDevExit = $LASTEXITCODE
        $env:OP_SERVICE_ACCOUNT_TOKEN = $null
        if ($shareDevExit -eq 0) {
            Write-Host "Vault condiviso con il tuo account ($DEV_EMAIL)." -ForegroundColor Green
        } else {
            Write-Host "ATTENZIONE: impossibile condividere con ${DEV_EMAIL}: $shareDevOut" -ForegroundColor Yellow
            Write-Host "  Chiedi all'admin di eseguire:" -ForegroundColor Yellow
            Write-Host "  op vault user grant --vault $VAULT_ID --user $DEV_EMAIL --permissions allow_viewing,allow_editing,allow_managing"
        }

        $env:OP_SERVICE_ACCOUNT_TOKEN = $saTokenFromVault
        $shareAdminOut = op vault user grant --vault $VAULT_ID --user $ADMIN_EMAIL --permissions "allow_viewing,allow_editing,allow_managing" 2>&1
        $shareAdminExit = $LASTEXITCODE
        $env:OP_SERVICE_ACCOUNT_TOKEN = $null
        if ($shareAdminExit -eq 0) {
            Write-Host "Vault condiviso con l'admin ($ADMIN_EMAIL)." -ForegroundColor Green
        } else {
            Write-Host "ATTENZIONE: impossibile condividere con ${ADMIN_EMAIL}: $shareAdminOut" -ForegroundColor Yellow
            Write-Host "  Chiedi all'admin di eseguire:" -ForegroundColor Yellow
            Write-Host "  op vault user grant --vault $VAULT_ID --user $ADMIN_EMAIL --permissions allow_viewing,allow_editing,allow_managing"
        }

        $ErrorActionPreference = "Stop"
        $env:OP_SERVICE_ACCOUNT_TOKEN = $saTokenFromVault
    }
} else {
    $VAULT_JUST_CREATED = $false
    Write-Host "Vault '$REPO_NAME' gia esistente ($VAULT_ID) - skip creazione" -ForegroundColor Cyan

    if ($BOOTSTRAPPED_TOKEN) {
        $env:OP_SERVICE_ACCOUNT_TOKEN = $saTokenFromVault
    }
}

# --- Condivisione vault con admin (solo in modalita interattiva pura) ---
if ($MODE -eq "interactive") {
    $ErrorActionPreference = "Continue"
    $vaultUsersRaw = op vault user list $VAULT_ID --format=json 2>$null | Out-String
    $ErrorActionPreference = "Stop"

    if ($vaultUsersRaw -match [regex]::Escape($ADMIN_EMAIL)) {
        Write-Host "Admin ($ADMIN_EMAIL) ha gia accesso al vault." -ForegroundColor Cyan
    } else {
        Write-Host ""
        $shareChoice = Read-Host "Vuoi condividere il vault '$REPO_NAME' con l'admin ($ADMIN_EMAIL)? (y/N)"
        if ($shareChoice -match '^[Yy]$') {
            $ErrorActionPreference = "Continue"
            $shareOutput = op vault user grant --vault $VAULT_ID --user $ADMIN_EMAIL --permissions "allow_viewing,allow_editing,allow_managing" 2>&1
            $ErrorActionPreference = "Stop"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Vault condiviso con l'admin." -ForegroundColor Green
            } else {
                Write-Host "ATTENZIONE: Impossibile condividere il vault:" -ForegroundColor Yellow
                Write-Host "  $shareOutput"
                Write-Host "Condividilo manualmente:"
                Write-Host "  op vault user grant --vault $VAULT_ID --user $ADMIN_EMAIL --permissions allow_viewing,allow_editing,allow_managing"
            }
        }
    }
}

# --- Avviso SA (solo in modalita interattiva, solo se vault appena creato) ---
if ($MODE -eq "interactive" -and $VAULT_JUST_CREATED) {
    Write-Host ""
    Write-Host "ATTENZIONE: assegna manualmente il vault al Service Account '$SA_NAME':" -ForegroundColor Yellow
    Write-Host "  1password.com -> Service Accounts -> $SA_NAME -> Vaults -> Add vault -> $REPO_NAME"
    Write-Host ""
    Write-Host "Oppure riesegui con il token SA (il vault non verra ricreato):" -ForegroundColor Cyan
    Write-Host '  $env:OP_SERVICE_ACCOUNT_TOKEN = (op read "op://FuturaDev/OP-SA-Token/credential")'
    Write-Host "  .\run-env-v2.ps1"
}

# --- Salva segreti nel vault ---
$secretNames  = @()
$secretValues = @()

foreach ($line in Get-Content .env) {
    if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
    $eqIdx = $line.IndexOf('=')
    if ($eqIdx -lt 0) { continue }
    $varName  = $line.Substring(0, $eqIdx)
    $varValue = $line.Substring($eqIdx + 1)

    if ($varName -match $SECRET_PATTERNS) {
        $secretNames  += $varName
        $secretValues += $varValue
    }
}

$SECRETS_COUNT = $secretNames.Count

if ($SECRETS_COUNT -eq 0) {
    Write-Host ""
    Write-Host "Nessun segreto trovato nel .env (pattern: $SECRET_PATTERNS)" -ForegroundColor Yellow
    Write-Host "Il .env.tpl conterra' solo valori in chiaro."
} else {
    Write-Host ""
    for ($i = 0; $i -lt $secretNames.Count; $i++) {
        $sName  = $secretNames[$i]
        $sValue = $secretValues[$i]

        $ErrorActionPreference = "Continue"
        op item delete $sName --vault $VAULT_ID 2>$null
        $ErrorActionPreference = "Stop"

        $sField    = Get-FieldName $sName
        $sFieldArg = "$sField=$sValue"
        op item create --vault $VAULT_ID --category "Password" --title $sName $sFieldArg

        Write-Host "  -> $sName" -ForegroundColor Green
    }
    Write-Host "$SECRETS_COUNT segreti salvati in 1Password (vault: $REPO_NAME)" -ForegroundColor Green
}

# --- Genera .env.tpl ---
$tplLines = @()
foreach ($line in Get-Content .env) {
    if ($line -match '^\s*#' -or $line -match '^\s*$') {
        $tplLines += $line; continue
    }
    $eqIdx = $line.IndexOf('=')
    if ($eqIdx -lt 0) { $tplLines += $line; continue }

    $varName  = $line.Substring(0, $eqIdx)
    $varValue = $line.Substring($eqIdx + 1)

    if ($varName -match $SECRET_PATTERNS) {
        $tField    = Get-FieldName $varName
        $tplLines += "$varName=op://$REPO_NAME/$varName/$tField"
    } else {
        $tplLines += $line
    }
}
$tplLines | Set-Content -Path .env.tpl -Encoding UTF8

# Aggiunge .env al .gitignore
if (Test-Path .gitignore) {
    $gi = Get-Content .gitignore
    if ($gi -notcontains ".env") { Add-Content .gitignore ".env" }
} else {
    ".env" | Set-Content .gitignore -Encoding UTF8
}

Write-Host ""
Write-Host "Done! $SECRETS_COUNT segreti salvati in 1Password, .env.tpl generato." -ForegroundColor Green
Write-Host ""
Write-Host "Per avviare l'app:"
Write-Host "  op run --env-file=.env.tpl -- uv run uvicorn app.main:app --port 8000 --reload"