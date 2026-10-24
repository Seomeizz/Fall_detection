$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceApp = $PSScriptRoot
$runtimeApp = Join-Path $repoRoot "mobile_app_runtime"
$manifestPath = Join-Path $runtimeApp "android\app\src\main\AndroidManifest.xml"

if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
    Write-Host "Khong tim thay lenh flutter. Hay cai Flutter SDK va them flutter\bin vao PATH." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $manifestPath)) {
    if (Test-Path $runtimeApp) {
        Write-Host "Thu muc mobile_app_runtime da ton tai nhung chua co Android scaffold." -ForegroundColor Red
        Write-Host "Hay doi ten/xoa thu muc nay neu no la thu muc tao loi, sau do chay lai script." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Tao Flutter Android project lan dau..." -ForegroundColor Cyan
    flutter create --platforms=android --project-name fall_detection_monitor $runtimeApp
    if ($LASTEXITCODE -ne 0) { throw "flutter create that bai." }
}

Write-Host "Copy source giao dien Flutter vao project chay..." -ForegroundColor Cyan
Copy-Item (Join-Path $sourceApp "lib\main.dart") (Join-Path $runtimeApp "lib\main.dart") -Force
Copy-Item (Join-Path $sourceApp "pubspec.yaml") (Join-Path $runtimeApp "pubspec.yaml") -Force

# Permit HTTP to a trusted laptop on the local Wi-Fi during development.
# This is not suitable for exposing the backend to the public Internet.
if (Test-Path $manifestPath) {
    $manifest = Get-Content $manifestPath -Raw
    if ($manifest -notmatch 'android.permission.INTERNET') {
        $applicationIndex = $manifest.IndexOf("<application")
        if ($applicationIndex -lt 0) { throw "Khong tim thay the <application> trong AndroidManifest.xml." }
        $permission = "    <uses-permission android:name=`"android.permission.INTERNET`" />`r`n`r`n"
        $manifest = $manifest.Insert($applicationIndex, $permission)
    }
    if ($manifest -notmatch 'android:usesCleartextTraffic') {
        $manifest = [regex]::Replace($manifest, '<application(?=\s)', '<application android:usesCleartextTraffic="true"', [System.Text.RegularExpressions.RegexOptions]::None)
    }
    Set-Content -Path $manifestPath -Value $manifest -Encoding UTF8
}

Push-Location $runtimeApp
try {
    Write-Host "Tai dependencies Flutter..." -ForegroundColor Cyan
    flutter pub get
    if ($LASTEXITCODE -ne 0) { throw "flutter pub get that bai." }
    Write-Host "Danh sach thiet bi Flutter nhin thay:" -ForegroundColor Cyan
    flutter devices
    Write-Host "Mo app tren emulator/dien thoai. Backend chua can chay; giao dien se hien thi trang thai offline." -ForegroundColor Green
    flutter run
    if ($LASTEXITCODE -ne 0) { throw "flutter run that bai. Kiem tra flutter devices va flutter doctor -v." }
}
finally {
    Pop-Location
}
