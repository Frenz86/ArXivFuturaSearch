<#
.SYNOPSIS
    Crea il Service Account 'Futura-Dev' in 1Password senza scadenza.
    Da eseguire una volta sola con login interattivo (op signin).
#>

$SA_NAME = "Futura-Dev"

# --- Prerequisiti ---
if (-not (Get-Command op -ErrorAction SilentlyContinue)) {
    Write-Host "1Password CLI non installato." -ForegroundColor Red; exit 1
}
if ($env:OP_SERVICE_ACCOUNT_TOKEN) {
    Write-Host "Rimuovi OP_SERVICE_ACCOUNT_TOKEN prima di procedere:" -ForegroundColor Yellow
    Write-Host '  $env:OP_SERVICE_ACCOUNT_TOKEN = $null'
    Write-Host "  op signin"; exit 1
}
$null = op account list 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Non autenticato. Esegui: op signin" -ForegroundColor Red; exit 1
}

# --- Mostra vault disponibili (escludi Personal/Private, non supportati dai SA) ---
Write-Host ""
$allVaultsRaw = op vault list --format=json 2>$null | ConvertFrom-Json
# I vault personali (di default si chiamano "Private") non sono accessibili dai SA
$vaults = $allVaultsRaw | Where-Object { $_.name -notin @('Private', 'Personal') }

Write-Host "Vault disponibili (esclusi vault personali):" -ForegroundColor Cyan
$vaults | ForEach-Object { Write-Host "  - $($_.name)  [$($_.id)]" }

Write-Host ""
Write-Host "Vuoi dare accesso a TUTTI i vault elencati? (y/N)" -ForegroundColor Yellow
$allVaults = Read-Host

$vaultArgs = @()
if ($allVaults -match '^[Yy]$') {
    foreach ($v in $vaults) {
        $vaultArgs += '--vault'; $vaultArgs += "$($v.id):read_items"
    }
} else {
    $names = (Read-Host "Nomi vault separati da virgola") -split ',' | ForEach-Object { $_.Trim() }
    foreach ($name in $names) {
        $vaultArgs += '--vault'; $vaultArgs += "${name}:read_items"
    }
}

# --- Crea SA (nessun --expires-in = nessuna scadenza) ---
Write-Host ""
Write-Host "Creazione Service Account '$SA_NAME' senza scadenza..." -ForegroundColor Cyan

$saArgs = @('service-account', 'create', $SA_NAME) + $vaultArgs
$saOutput = & op @saArgs 2>&1
$saString = $saOutput | Out-String

if ($LASTEXITCODE -ne 0) {
    Write-Host "Errore:" -ForegroundColor Red
    Write-Host $saString; exit 1
}

Write-Host "Service Account '$SA_NAME' creato!" -ForegroundColor Green
Write-Host ""
Write-Host "SALVA QUESTO TOKEN ORA - non sara' piu' visibile." -ForegroundColor Yellow

if ($saString -match '(ops_[A-Za-z0-9._\-]+|sa_[A-Za-z0-9._\-]+)') {
    $SA_TOKEN = $Matches[1]
    try {
        [Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN', $SA_TOKEN, 'User')
        Write-Host "OP_SERVICE_ACCOUNT_TOKEN salvata (User-level). Apri una nuova shell." -ForegroundColor Green
    } catch {
        Write-Host "Impossibile salvare automaticamente." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Token: $SA_TOKEN"
} else {
    Write-Host "Non sono riuscito ad estrarre il token. Output completo:" -ForegroundColor Yellow
    Write-Host $saString
    Write-Host ""
    Write-Host "Copia il token manualmente e imposta:"
    Write-Host "  [Environment]::SetEnvironmentVariable('OP_SERVICE_ACCOUNT_TOKEN','TOKEN','User')"
}
