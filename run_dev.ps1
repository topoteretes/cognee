# MemoryOS Dev Launcher

$host.ui.RawUI.WindowTitle = "MemoryOS Launcher"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "        MemoryOS Developer Console        " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Ensure backend virtual environment exists
if (-not (Test-Path "backend\.venv")) {
    Write-Host "[Error] Virtual environment not found in backend\.venv!" -ForegroundColor Red
    Write-Host "Please wait for installation to finish." -ForegroundColor Yellow
    Exit
}

# Start Backend API
Write-Host "[+] Launching FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
$BackendProcess = Start-Process -FilePath "backend\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload" -PassThru -NoNewWindow

# Start Frontend
Write-Host "[+] Launching Next.js Frontend on http://localhost:3000..." -ForegroundColor Green
$FrontendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c cd apps\web && npm run dev" -PassThru -NoNewWindow

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host " MemoryOS is running!                      " -ForegroundColor Green
Write-Host " - Frontend: http://localhost:3000        " -ForegroundColor Green
Write-Host " - Backend API: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Press Ctrl+C in this window to stop all services." -ForegroundColor Yellow
Write-Host ""

# Wait loop and clean termination
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host ""
    Write-Host "[-] Shutting down MemoryOS services..." -ForegroundColor Yellow
    
    if ($BackendProcess) {
        Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($FrontendProcess) {
        # Kill the node / dev server child processes
        $Children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $FrontendProcess.Id }
        foreach ($Child in $Children) {
            Stop-Process -Id $Child.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $FrontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "[+] MemoryOS services stopped successfully." -ForegroundColor Green
}
