@echo off
setlocal
cd /d "%~dp0"

:menu
echo.
echo  Ebook AI service (Docker)
echo  -------------------------
echo   1. Start
echo   2. Stop
echo   3. Exit
echo.
set "option="
set /p "option=Choose 1, 2 or 3: "

if "%option%"=="1" goto start
if "%option%"=="2" goto stop
if "%option%"=="3" goto end
if not defined option goto end
echo Unknown option.
goto menu

:start
call :check_docker || goto menu
if not exist ".env" (
    echo The .env file is missing.
    goto menu
)
echo Starting... the first time takes several minutes.
docker compose up -d --build
if errorlevel 1 (
    echo Start failed. See the messages above.
) else (
    echo Started: http://localhost:8000/health
)
goto menu

:stop
call :check_docker || goto menu
docker compose down
if errorlevel 1 (
    echo Stop failed. See the messages above.
) else (
    echo Stopped.
)
goto menu

:check_docker
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Start Docker Desktop and try again.
    exit /b 1
)
exit /b 0

:end
endlocal
