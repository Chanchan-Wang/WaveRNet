@echo off
echo ============================================================
echo WaveRNet GitHub 上传命令
echo ============================================================
echo.
echo 请按照以下步骤操作：
echo.
echo 1. 在 GitHub 上创建新仓库
echo    - 访问 https://github.com/new
echo    - 仓库名称：WaveRNet
echo    - 描述：Wavelet-Guided Frequency Learning for Domain-Generalized Retinal Vessel Segmentation
echo    - 选择 Public
echo    - 不要初始化 README（我们已经有了）
echo.
echo 2. 初始化本地仓库并上传
echo    执行以下命令：
echo.
echo    cd WaveRNet_GitHub
echo    git init
echo    git add .
echo    git commit -m "Initial commit: WaveRNet implementation"
echo    git branch -M main
echo    git remote add origin https://github.com/你的用户名/WaveRNet.git
echo    git push -u origin main
echo.
echo 3. 上传完成后
echo    - 在 GitHub 仓库页面添加 Topics 标签：
echo      deep-learning, medical-imaging, retinal-vessel-segmentation, 
echo      domain-generalization, wavelet-transform, pytorch
echo    - 检查 README 显示是否正常
echo    - 上传预训练模型到云盘并更新链接
echo.
echo ============================================================
echo 提示：记得先更新 README.md 中的个人信息！
echo ============================================================
pause
