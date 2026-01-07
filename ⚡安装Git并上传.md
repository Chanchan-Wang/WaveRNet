# 🚨 需要先安装 Git

你的电脑上还没有安装 Git，所以无法执行 git 命令。

## 📥 第一步：安装 Git

### 方法1：直接下载安装（推荐）

1. 访问 Git 官网：https://git-scm.com/download/win
2. 下载会自动开始，如果没有，点击 "Click here to download manually"
3. 下载完成后，双击安装包
4. 安装过程中，**全部使用默认选项**，一直点"Next"即可
5. 安装完成后，**重启命令行窗口**（关闭当前 cmd，重新打开）

### 方法2：使用 GitHub Desktop（更简单，推荐新手）

如果你觉得命令行太复杂，可以使用 GitHub Desktop：

1. 访问：https://desktop.github.com/
2. 下载并安装 GitHub Desktop
3. 打开软件，登录你的 GitHub 账号
4. 点击 "File" → "Add local repository"
5. 选择 `F:\SAM_base\WaveRNet_GitHub` 文件夹
6. 如果提示"This directory does not appear to be a Git repository"
   - 点击 "create a repository"
   - 取消勾选 "Initialize this repository with a README"
   - 点击 "Create Repository"
7. 点击 "Publish repository"
8. 确认名称是 `WaveRNet`
9. 取消勾选 "Keep this code private"（让仓库公开）
10. 点击 "Publish Repository"

完成！

---

## 🚀 第二步：安装 Git 后上传代码

### 如果你安装了 Git（方法1）

安装完 Git 后，**重新打开命令行**，然后执行：

```cmd
cd F:\SAM_base\WaveRNet_GitHub
git init
git add .
git commit -m "Initial commit: WaveRNet implementation"
git branch -M main
git remote add origin https://github.com/Chanchan-Wang/WaveRNet.git
git push -u origin main
```

**注意**：推送时如果要求密码，需要使用 Personal Access Token（不是 GitHub 密码）

### 创建 Personal Access Token

1. 登录 GitHub 网站
2. 点击右上角头像 → Settings
3. 左侧菜单最下面 → Developer settings
4. 点击 Personal access tokens → Tokens (classic)
5. 点击 Generate new token → Generate new token (classic)
6. 填写：
   - Note: `WaveRNet Upload`
   - Expiration: `90 days`
   - 勾选 **repo**（所有 repo 相关权限）
7. 点击 Generate token
8. **立即复制这个 token！**（离开页面后就看不到了）
9. 当命令行要求输入密码时，粘贴这个 token

---

## ✅ 验证上传成功

上传完成后，访问：https://github.com/Chanchan-Wang/WaveRNet

你应该能看到：
- README.md 显示在页面上
- 所有文件夹（models, datasets, utils, configs等）
- 你的代码已经成功上传！

---

## 🎯 推荐方案

**对于新手，我强烈推荐使用 GitHub Desktop（方法2）**，因为：
- 不需要记命令
- 图形界面更直观
- 不需要配置 Token
- 一键完成所有操作

如果你想学习 Git 命令，可以选择方法1。

---

## 📞 遇到问题？

如果安装或上传过程中遇到任何问题，告诉我具体的错误信息，我会帮你解决！
