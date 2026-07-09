# SRS 一键启动（开两个新窗口：后端 + 前端）
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\SRS_融合'; pip install -r '$root\requirements.txt' -q; python srs_api_v13.py --port 8765"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\tmp_SRS'; npm install; npm run dev"
Write-Host "Backend -> http://localhost:8765/api/health"
Write-Host "Frontend -> http://localhost:5173"
