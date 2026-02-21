<#
.SYNOPSIS
    Setup segreti in 1Password + genera .env.tpl
.DESCRIPTION
    Questo script RICHIEDE login interattivo (op signin) perche' deve SCRIVERE
    nel vault. Il Service Account (read-only) serve solo per l'uso quotidiano
    con "op run".
#>

$ErrorActionPreference = "Stop"

# Pattern per riconoscere i segreti
$SECRET_PATTERNS = "KEY|PASSWORD|SECRET|TOKEN|CREDENTIAL|USER"

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
    Write-Host "Non sei dentro una repo git." -ForegroundColor Red
    exit 1
}

if (-not (Get-Command op -ErrorAction SilentlyContinue)) {
    Write-Host "1Password CLI non installato. Installalo da https://developer.1password.com/docs/cli" -ForegroundColor Red
    exit 1
}

if ($env:OP_SERVICE_ACCOUNT_TOKEN) {
    Write-Host "Questo script richiede login interattivo (non Service Account)." -ForegroundColor Yellow
    Write-Host "Rimuovi il token e usa op signin:"
    Write-Host '  $env:OP_SERVICE_ACCOUNT_TOKEN = $null'
    Write-Host "  op signin"
    Write-Host "  .\run-env.ps1"
    exit 1
}

$null = op account list 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Non sei autenticato a 1Password. Esegui: op signin" -ForegroundColor Red
    exit 1
}
Write-Host "Autenticato con sessione interattiva" -ForegroundColor Green

$REPO_URL = git remote get-url origin
$REPO_NAME = [System.IO.Path]::GetFileNameWithoutExtension($REPO_URL)

if (-not (Test-Path .env)) {
    Write-Host ".env non trovato nella repo" -ForegroundColor Red
    exit 1
}

# --- Crea o trova il vault ---
$VAULT_ID = $null
try {
    $vaultJson = op vault get $REPO_NAME --format=json 2>$null
    if ($vaultJson) {
        $vaultObj = $vaultJson | ConvertFrom-Json
        $VAULT_ID = $vaultObj.id
    }
}
catch {
    $VAULT_ID = $null
}

if (-not $VAULT_ID) {
    Write-Host "Creazione vault '$REPO_NAME'..." -ForegroundColor Cyan
    $createRaw = op vault create $REPO_NAME --format=json
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Impossibile creare il vault." -ForegroundColor Red
        exit 1
    }
    $createObj = $createRaw | ConvertFrom-Json
    $VAULT_ID = $createObj.id
    if (-not $VAULT_ID) {
        Write-Host "Impossibile estrarre ID del vault." -ForegroundColor Red
        exit 1
    }
}
Write-Host "Vault: $REPO_NAME ($VAULT_ID)" -ForegroundColor Cyan

# --- Salva segreti nel vault ---
$secretNames = @()
$secretValues = @()

foreach ($line in Get-Content .env) {
    if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
    $eqIdx = $line.IndexOf('=')
    if ($eqIdx -lt 0) { continue }
    $varName = $line.Substring(0, $eqIdx)
    $varValue = $line.Substring($eqIdx + 1)

    if ($varName -match $SECRET_PATTERNS) {
        $secretNames += $varName
        $secretValues += $varValue
    }
}

$SECRETS_COUNT = $secretNames.Count

if ($SECRETS_COUNT -eq 0) {
    Write-Host "Nessun segreto trovato nel .env (pattern: $SECRET_PATTERNS)" -ForegroundColor Yellow
    Write-Host "Il .env.tpl conterra' solo valori in chiaro."
}
else {
    for ($i = 0; $i -lt $secretNames.Count; $i++) {
        $sName = $secretNames[$i]
        $sValue = $secretValues[$i]

        # Elimina item esistente (ignora errore se non esiste)
        $ErrorActionPreference = "Continue"
        op item delete $sName --vault $VAULT_ID 2>$null
        $ErrorActionPreference = "Stop"

        $sField = Get-FieldName $sName
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
        $tplLines += $line
        continue
    }
    $eqIdx = $line.IndexOf('=')
    if ($eqIdx -lt 0) {
        $tplLines += $line
        continue
    }
    $varName = $line.Substring(0, $eqIdx)
    $varValue = $line.Substring($eqIdx + 1)

    if ($varName -match $SECRET_PATTERNS) {
        $tField = Get-FieldName $varName
        $tplLines += "$varName=op://$REPO_NAME/$varName/$tField"
    }
    else {
        # Strip inline comments so they don't become part of the variable value
        $cleanValue = ($varValue -split '\s+#')[0]
        $tplLines += "$varName=$cleanValue"
    }
}
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllLines((Join-Path (Get-Location).Path '.env.tpl'), $tplLines, $utf8NoBom)

# Aggiunge .env al .gitignore
if (Test-Path .gitignore) {
    $gi = Get-Content .gitignore
    if ($gi -notcontains ".env") {
        Add-Content .gitignore ".env"
    }
}
else {
    ".env" | Set-Content .gitignore -Encoding UTF8
}

Write-Host ""
Write-Host "Done! $SECRETS_COUNT segreti salvati in 1Password, .env.tpl generato." -ForegroundColor Green

# --- Offri creazione Service Account ---
$SA_NAME = "Futura-Dev"

# Controlla se il SA esiste già: la variabile User-level persiste anche se
# nella sessione corrente è stata azzerata con $env:OP_SERVICE_ACCOUNT_TOKEN = $null
$existingSA = [bool][Environment]::GetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN', 'User')

if ($existingSA) {
    Write-Host ""
    Write-Host "Service Account '$SA_NAME' trovato." -ForegroundColor Cyan
    Write-Host "Aggiungi il vault '$REPO_NAME' dal portale 1Password:" -ForegroundColor Yellow
    Write-Host "  1password.com -> Service Accounts -> $SA_NAME -> Vaults -> Add vault -> $REPO_NAME" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "Per resettare il token:" -ForegroundColor DarkGray
    Write-Host "  [Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN', `$null, 'User')" -ForegroundColor DarkGray
}
else {

Write-Host ""
$createSA = Read-Host "Vuoi creare il Service Account '$SA_NAME' per evitare op signin in futuro? (y/N)"

if ($createSA -match '^[Yy]$') {
    Write-Host "Creazione Service Account '$SA_NAME'..." -ForegroundColor Cyan
    $saVaultArg = $VAULT_ID + ":read_items"
    $saOutput = op service-account create $SA_NAME --vault $saVaultArg 2>&1
    $saExitCode = $LASTEXITCODE
    $saString = $saOutput | Out-String

    if ($saExitCode -eq 0) {
        Write-Host ""
        Write-Host "Service Account creato!" -ForegroundColor Green
        Write-Host ""
        Write-Host "SALVA QUESTO TOKEN ORA - non sara' piu' visibile." -ForegroundColor Yellow

        if ($saString -match '(ops_[A-Za-z0-9._\-]+|sa_[A-Za-z0-9._\-]+)') {
            $SA_TOKEN = $Matches[1]

            try {
                [Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN', $SA_TOKEN, 'User')
                Write-Host "OP_SERVICE_ACCOUNT_TOKEN impostata a livello User (Windows). Apri una nuova shell." -ForegroundColor Green
            }
            catch {
                Write-Host "Impossibile impostare la variabile utente." -ForegroundColor Yellow
            }

            Write-Host ""
            Write-Host "Token: $SA_TOKEN"
        }
        else {
            Write-Host "Non sono riuscito ad estrarre il token. Output completo:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host $saString
            Write-Host ""
            Write-Host "Copia il token e impostalo manualmente:"
            Write-Host "[Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN','PASTE_TOKEN','User')"
        }
    }
    else {
        Write-Host "Errore nella creazione del Service Account:" -ForegroundColor Red
        Write-Host $saString
    }
}
} # end: SA non esistente

Write-Host ""
Write-Host 'Per avviare l''app:'
Write-Host "  op run --env-file=.env.tpl -- uv run uvicorn app.main:app --port 8000 --reload"
