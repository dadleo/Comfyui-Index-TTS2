# 更新说明 / Update Instructions

## 🚨 如果你遇到以下错误 / If you encounter this error:

```
TypeError: IndexTTS2.__init__() got an unexpected keyword argument 'is_fp16'
```

**这说明你的代码版本过旧，需要更新！**  
**This means your code is outdated and needs to be updated!**

---

## 📥 如何更新 / How to Update

### 方法 1: Git Pull（推荐 / Recommended）

如果你是通过 Git 克隆的仓库：

```bash
cd ComfyUI/custom_nodes/Comfyui-Index-TTS2
git pull origin main
```

### 方法 2: 手动下载

1. 访问 GitHub 仓库
2. 下载最新的代码
3. 替换你的 `Comfyui-Index-TTS2` 文件夹

### 方法 3: ComfyUI Manager

1. 打开 ComfyUI Manager
2. 找到 "Comfyui-Index-TTS2"
3. 点击 "Update" 按钮

---

## ✅ 最新修复包含 / Latest Fixes Include

### 修复 1: 参数名错误 (Commit: 4d299da)
- **问题**: 所有节点使用了错误的参数名 `is_fp16`
- **修复**: 改为正确的参数名 `use_fp16`
- **影响文件**:
  - `nodes/basic_tts_node.py`
  - `nodes/advanced_control_node.py`
  - `nodes/duration_control_node.py`
  - `nodes/emotion_control_node.py`
  - `nodes/model_manager_node.py`

### 修复 2: 情感文本描述功能 (Commit: c730727)
- **问题**: 情感文本描述不起作用
- **修复**: 使用关键词匹配分析情感文本
- **影响文件**:
  - `indextts/infer_v2.py`

### 修复 3: 加速功能集成
- **新增**: GPT2 加速引擎
- **新增**: Torch Compile 优化
- **影响文件**:
  - `indextts/accel/` (新目录)
  - `indextts/gpt/model_v2.py`
  - `nodes/multi_talk_node.py`
  - `nodes/emotion_voice_multi_talk_node.py`

---

## 🔍 如何验证更新成功 / How to Verify Update

### 检查 1: 查看提交历史

```bash
cd ComfyUI/custom_nodes/Comfyui-Index-TTS2
git log --oneline -5
```

你应该看到：
```
4d299da fix: correct parameter name from is_fp16 to use_fp16 in all nodes
1d036fd docs: add emotion text description fix documentation
c730727 fix: use keyword-based emotion analysis when Qwen model unavailable
...
```

### 检查 2: 查看文件内容

打开 `nodes/emotion_control_node.py`，找到第 408-413 行：

```python
model = IndexTTS2(
    cfg_path=config_path,
    model_dir=model_dir,
    use_fp16=use_fp16,  # ✅ 应该是 use_fp16，不是 is_fp16
    use_cuda_kernel=use_cuda_kernel
)
```

如果你看到的是 `is_fp16`，说明更新失败，需要重新更新。

### 检查 3: 重启 ComfyUI

更新代码后，**必须重启 ComfyUI** 才能生效！

---

## 🐛 常见问题 / FAQ

### Q1: 我更新后还是报错怎么办？

**A**: 确保你已经：
1. ✅ 完全关闭 ComfyUI
2. ✅ 更新代码到最新版本
3. ✅ 重新启动 ComfyUI
4. ✅ 清除浏览器缓存（Ctrl+F5）

### Q2: Git pull 失败怎么办？

**A**: 如果有本地修改冲突：

```bash
# 保存你的本地修改
git stash

# 拉取最新代码
git pull origin main

# 恢复你的本地修改（可选）
git stash pop
```

或者直接重置到最新版本（⚠️ 会丢失本地修改）：

```bash
git fetch origin
git reset --hard origin/main
```

### Q3: 我没有使用 Git，怎么更新？

**A**: 
1. 备份你的 `Comfyui-Index-TTS2` 文件夹
2. 删除旧的 `Comfyui-Index-TTS2` 文件夹
3. 重新下载最新版本
4. 解压到 `ComfyUI/custom_nodes/` 目录

### Q4: 更新后模型需要重新下载吗？

**A**: 不需要！模型文件不受影响，只是代码更新。

---

## 📝 更新日志 / Changelog

### 2025-11-05 (最新 / Latest)

#### 修复 (Fixes)
- ✅ 修复所有节点的 `is_fp16` 参数错误
- ✅ 修复情感文本描述不起作用的问题
- ✅ 修复 BigVGAN 设备类型错误
- ✅ 修复 GPT2 加速初始化错误

#### 新增 (New Features)
- ✨ 集成官方加速功能
- ✨ 添加 GPT2 加速引擎参数
- ✨ 添加 Torch Compile 优化参数
- ✨ 增强的关键词情感分析

#### 文档 (Documentation)
- 📄 添加 `EMOTION_TEXT_FIX.md` - 情感文本修复说明
- 📄 添加 `UPDATE_INSTRUCTIONS.md` - 更新说明（本文件）

---

## 💡 需要帮助？ / Need Help?

如果更新后仍然有问题，请：

1. **检查错误日志**：复制完整的错误信息
2. **提供环境信息**：
   - ComfyUI 版本
   - Python 版本
   - PyTorch 版本
   - 操作系统
3. **提交 Issue**：在 GitHub 仓库提交详细的问题报告

---

## 🎯 快速修复指南 / Quick Fix Guide

如果你只是想快速修复 `is_fp16` 错误：

### 手动修复（不推荐，但可以应急）

打开以下 5 个文件，将所有 `is_fp16=use_fp16` 改为 `use_fp16=use_fp16`：

1. `nodes/basic_tts_node.py` (第 320 行)
2. `nodes/advanced_control_node.py` (第 474 行)
3. `nodes/duration_control_node.py` (第 300 行)
4. `nodes/emotion_control_node.py` (第 411 行)
5. `nodes/model_manager_node.py` (第 409 行)

**但是强烈建议使用 Git Pull 更新整个仓库，以获得所有修复和改进！**

---

## 📞 联系方式 / Contact

- GitHub Issues: [提交问题](https://github.com/your-repo/issues)
- 讨论区: [参与讨论](https://github.com/your-repo/discussions)

---

**最后更新 / Last Updated**: 2025-11-05  
**版本 / Version**: v1.0 (Commit: 4d299da)

