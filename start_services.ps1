# ==============================================================================
# SkyGuard AI — Service Launcher
# Starts FastAPI REST server and Streamlit Dashboard for Cloudflare Tunnel
# ==============================================================================

param (
    [ValidateSet("all", "dashboard", "api")]
    [string]$Service = "all"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  SKYGUARD AI — METEOROLOGICAL SERVICE LAUNCHER" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if ($Service -eq "all" -or $Service -eq "api") {
    Write-Host "[+] Launching FastAPI REST API on port 8000..." -ForegroundColor Green
    Start-Process python -ArgumentList "-m uvicorn api.server:app --host 0.0.0.0 --port 8000"
}

if ($Service -eq "all" -or $Service -eq "dashboard") {
    Write-Host "[+] Launching Streamlit Surveillance Dashboard on port 8501..." -ForegroundColor Green
    Start-Process python -ArgumentList "-m streamlit run dashboard/app.py"
}

Write-Host "`n[+] Services running in background:" -ForegroundColor Cyan
Write-Host "    • Streamlit Dashboard : http://localhost:8501" -ForegroundColor Yellow
Write-Host "    • FastAPI REST API   : http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "`n[i] Connected to Cloudflare Tunnel 'my-laptop-tunnel'!" -ForegroundColor Gray
