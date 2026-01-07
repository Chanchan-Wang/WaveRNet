@echo off
chcp 65001 >nul
echo ============================================================
echo 🚨 紧急修复：正确上传 WaveRNet 到 GitHub
echo ============================================================
echo.
echo 【第1步】配置 Git 用户信息
echo.
git config --global user.name "Chanchan Wang"
git config --global user.email "wusheng070@gmail.com"
echo ✅ Git 用户信息配置完成
echo.
echo ============================================================
echo 【第2步】进入正确的目录
echo.
cd /d "%~dp0"
echo 当前目录：%CD%
echo.
echo ============================================================
echo 【第3步】初始化 Git 仓库
echo.
git init
echo.
echo ============================================================
echo 【第4步】添加文件到 Git
echo.
git add .
echo.
echo ============================================================
echo 【第5步】提交代码
echo.
git commit -m "Initial commit: WaveRNet implementation"
echo.
echo ============================================================
echo 【第6步】设置主分支
echo.
git branch -M main
echo.
echo ============================================================
echo 【第7步】连接到 GitHub
echo.
git remote add origin https://github.com/Chanchan-Wang/WaveRNet.git
echo.
echo ============================================================
echo 【第8步】推送到 GitHub
echo.
echo 注意：推送时会要求登录
echo - 用户名：Chanchan-Wang
echo - 密码：使用 Personal Access Token（不是 GitHub 密码）
echo.
git push -u origin main
echo.
echo ============================================================
echo 🎉 上传完成！
echo.
echo 访问你的仓库：https://github.com/Chanchan-Wang/WaveRNet
echo ============================================================
pause
