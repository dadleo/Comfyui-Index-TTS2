# 📦 ComfyUI Registry 发布指南 / Publishing Guide

本文档说明如何将 IndexTTS2 插件发布到 ComfyUI Registry。

This document explains how to publish the IndexTTS2 plugin to ComfyUI Registry.

---

## 📋 目录 / Table of Contents

1. [前置要求](#前置要求--prerequisites)
2. [发布步骤](#发布步骤--publishing-steps)
3. [配置说明](#配置说明--configuration-details)
4. [验证发布](#验证发布--verify-publication)
5. [常见问题](#常见问题--faq)

---

## 🔧 前置要求 / Prerequisites

### 1. GitHub 仓库要求

✅ **必需文件**：
- [x] `pyproject.toml` - 项目元数据和 ComfyUI Registry 配置
- [x] `README.md` - 项目文档
- [x] `LICENSE` - 开源许可证
- [x] `requirements.txt` - Python 依赖
- [x] `__init__.py` - ComfyUI 节点注册

✅ **推荐文件**：
- [x] `INSTALL_GUIDE.md` - 安装指南
- [x] `UPDATE_INSTRUCTIONS.md` - 更新说明
- [x] `workflow-examples/` - 工作流示例

### 2. ComfyUI Registry 账号

1. 访问 [ComfyUI Registry](https://registry.comfy.org/)
2. 使用 GitHub 账号登录
3. 创建 Publisher 账号
4. 获取 Publisher ID

---

## 🚀 发布步骤 / Publishing Steps

### 步骤 1: 更新 pyproject.toml

确保 `pyproject.toml` 中的 Publisher ID 正确：

```toml
[tool.comfy]
PublisherId = "xuchenxu168"  # 替换为您的 Publisher ID
```

### 步骤 2: 准备图标和横幅（可选但推荐）

#### 图标 (Icon)
- **尺寸**: 400x400 像素（正方形）
- **格式**: SVG, PNG, JPG, GIF
- **位置**: `docs/icon.png`
- **URL**: `https://raw.githubusercontent.com/xuchenxu168/Comfyui-Index-TTS2/main/docs/icon.png`

#### 横幅 (Banner)
- **比例**: 21:9
- **推荐尺寸**: 1260x540 像素
- **格式**: SVG, PNG, JPG, GIF
- **位置**: `docs/banner.png`
- **URL**: `https://raw.githubusercontent.com/xuchenxu168/Comfyui-Index-TTS2/main/docs/banner.png`

如果暂时没有图标和横幅，可以先注释掉这两行：

```toml
# Icon = "https://raw.githubusercontent.com/xuchenxu168/Comfyui-Index-TTS2/main/docs/icon.png"
# Banner = "https://raw.githubusercontent.com/xuchenxu168/Comfyui-Index-TTS2/main/docs/banner.png"
```

### 步骤 3: 更新版本号

在 `pyproject.toml` 中更新版本号：

```toml
[project]
version = "1.2.0"  # 使用语义化版本号
```

**版本号规则**：
- **主版本号 (Major)**: 不兼容的 API 变更
- **次版本号 (Minor)**: 向后兼容的功能新增
- **修订号 (Patch)**: 向后兼容的问题修正

### 步骤 4: 提交更改

```bash
git add pyproject.toml LICENSE PUBLISHING_GUIDE.md
git commit -m "feat: add pyproject.toml for ComfyUI Registry publishing

- Add comprehensive project metadata
- Configure ComfyUI Registry settings
- Add Apache 2.0 license
- Add publishing guide documentation"
git push origin main
```

### 步骤 5: 创建 Git Tag（推荐）

```bash
# 创建带注释的标签
git tag -a v1.2.0 -m "Release v1.2.0: ComfyUI Registry support

- Add pyproject.toml configuration
- Add Apache 2.0 license
- Improve documentation
- Fix emotion text analysis
- Fix parameter compatibility issues"

# 推送标签到远程仓库
git push origin v1.2.0
```

### 步骤 6: 在 ComfyUI Registry 注册

1. 访问 [ComfyUI Registry](https://registry.comfy.org/)
2. 登录您的账号
3. 点击 "Publish Node" 或 "Add Node"
4. 输入 GitHub 仓库 URL: `https://github.com/xuchenxu168/Comfyui-Index-TTS2`
5. Registry 会自动读取 `pyproject.toml` 中的配置
6. 检查信息无误后提交

### 步骤 7: 等待审核

- ComfyUI Registry 会自动验证您的配置
- 通常几分钟内完成
- 如有问题会显示错误信息

---

## ⚙️ 配置说明 / Configuration Details

### pyproject.toml 关键配置

#### 1. 项目基本信息

```toml
[project]
name = "comfyui-indextts2"  # 唯一标识符，小写，使用连字符
version = "1.2.0"  # 语义化版本号
description = "..."  # 简短描述
```

#### 2. ComfyUI 特定配置

```toml
[tool.comfy]
PublisherId = "xuchenxu168"  # 您的 Publisher ID
DisplayName = "IndexTTS2 - AI-Enhanced Text-to-Speech"  # 显示名称
Icon = "..."  # 图标 URL（可选）
Banner = "..."  # 横幅 URL（可选）
requires-comfyui = ">=0.1.0"  # ComfyUI 版本要求
```

#### 3. 依赖管理

```toml
[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}  # 从 requirements.txt 读取
```

#### 4. 平台兼容性

```toml
classifiers = [
    "Operating System :: OS Independent",  # 跨平台
    "Environment :: GPU :: NVIDIA CUDA",  # NVIDIA GPU 支持
    "Environment :: GPU :: AMD ROCm",  # AMD GPU 支持
]
```

---

## ✅ 验证发布 / Verify Publication

### 1. 在 ComfyUI Manager 中搜索

发布成功后，用户可以在 ComfyUI Manager 中搜索 "IndexTTS2" 找到您的插件。

### 2. 检查 Registry 页面

访问您的插件页面：
```
https://registry.comfy.org/publishers/xuchenxu168/nodes/comfyui-indextts2
```

### 3. 测试安装

在新的 ComfyUI 环境中测试安装：

```bash
# 通过 ComfyUI Manager 安装
# 或手动安装
cd ComfyUI/custom_nodes
git clone https://github.com/xuchenxu168/Comfyui-Index-TTS2.git
cd Comfyui-Index-TTS2
pip install -r requirements.txt
```

---

## ❓ 常见问题 / FAQ

### Q1: 如何获取 Publisher ID？

**A**: 
1. 访问 https://registry.comfy.org/
2. 使用 GitHub 登录
3. 进入 Settings 或 Profile
4. 创建或查看您的 Publisher ID

### Q2: 图标和横幅是必需的吗？

**A**: 不是必需的，但强烈推荐。它们能让您的插件在 Registry 中更醒目。

### Q3: 如何更新已发布的插件？

**A**: 
1. 更新代码和 `pyproject.toml` 中的版本号
2. 提交并推送到 GitHub
3. 创建新的 Git tag
4. Registry 会自动检测更新

### Q4: 支持哪些许可证？

**A**: 支持所有 OSI 批准的开源许可证，如：
- Apache-2.0
- MIT
- GPL-3.0
- BSD-3-Clause

### Q5: 如何处理依赖冲突？

**A**: 
- 在 `requirements.txt` 中使用版本范围而非固定版本
- 避免与 ComfyUI 核心依赖冲突
- 使用 `optional-dependencies` 分离可选功能

### Q6: 发布失败怎么办？

**A**: 检查以下常见问题：
- [ ] `pyproject.toml` 格式是否正确
- [ ] Publisher ID 是否正确
- [ ] 仓库是否公开
- [ ] 是否包含必需文件（`__init__.py`, `README.md`）
- [ ] 版本号格式是否符合语义化版本规范

---

## 📚 参考资源 / References

- [ComfyUI Registry 官方文档](https://docs.comfy.org/zh-CN/registry/publishing)
- [Python Packaging 指南](https://packaging.python.org/)
- [语义化版本规范](https://semver.org/lang/zh-CN/)
- [TOML 格式规范](https://toml.io/)

---

## 🎉 发布后的工作

### 1. 宣传推广

- 在 ComfyUI 社区分享
- 在 GitHub Discussions 发布公告
- 更新项目 README 添加安装徽章

### 2. 维护更新

- 定期更新依赖版本
- 修复用户报告的问题
- 添加新功能
- 保持文档更新

### 3. 用户支持

- 及时回复 Issues
- 提供清晰的文档
- 创建示例工作流
- 建立用户社区

---

**祝您发布顺利！🚀**

如有问题，欢迎在 [GitHub Issues](https://github.com/xuchenxu168/Comfyui-Index-TTS2/issues) 提问。

