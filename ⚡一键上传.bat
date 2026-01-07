@echo off
chcp 65001 >nul
echo ============================================================
echo WaveRNet 一键上传到 GitHub
echo ============================================================
echo.
echo 请确认：
echo 1. 你已经在 GitHub 创建了 WaveRNet 仓库
echo 2. 你已经更新了 README.md 和 LICENSE 中的个人信息
echo.
set /p confirm="确认继续？(Y/N): "
if /i not "%confirm%"=="Y" (
    echo 已取消
    pause
    exit
)

echo.
echo [1/5] 初始化 Git 仓库...
git init
if errorlevel 1 (
    echo 错误：Git 初始化失败！请确保已安装 Git
    pause
    exit
)

echo.
echo [2/5] 添加所有文件...
git add .

echo.
echo [3/5] 提交代码...
git commit -m "Initial commit: WaveRNet implementation"

echo.
echo [4/5] 设置主分支...
git branch -M main

echo.
echo [5/5] 连接到 GitHub 并推送...
echo.
echo 请输入你的 GitHub 用户名：
set /p username="用户名: "
echo.
echo 正在连接到 https://github.com/%username%/WaveRNet.git
git remote add origin https://github.com/%username%/WaveRNet.git
git push -u origin main

echo.
echo ============================================================
echo 上传完成！
echo 访问你的仓库：https://github.com/%username%/WaveRNet
echo ============================================================
pause
