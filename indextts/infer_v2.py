import os
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import json
import re
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

# 使用简化兼容层导入 transformers 组件
from indextts.compat.simple_imports import AutoTokenizer, SeamlessM4TFeatureExtractor
from modelscope import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import safetensors
from transformers import SeamlessM4TFeatureExtractor
import random
import torch.nn.functional as F

# 导入高级音频系统
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from advanced_audio_systems import SpeakerEmbeddingCache, VoiceConsistencyController, AdaptiveQualityMonitor
    ADVANCED_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"[IndexTTS2] 高级音频系统导入失败: {e}")
    ADVANCED_SYSTEMS_AVAILABLE = False

# AI增强系统导入
try:
    from ai_enhanced_systems import IntelligentParameterLearner, AdaptiveAudioEnhancer, IntelligentQualityPredictor, AdaptiveCacheStrategy
    AI_ENHANCED_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"[IndexTTS2] AI增强系统导入失败: {e}")
    AI_ENHANCED_SYSTEMS_AVAILABLE = False

# 高质量重采样器
class AdvancedResampler:
    """高质量音频重采样器"""

    def __init__(self):
        self.resamplers = {}
        self.cache_size = 10

    def _create_high_quality_resampler(self, orig_sr: int, target_sr: int):
        """创建高质量重采样器"""
        return torchaudio.transforms.Resample(
            orig_sr, target_sr,
            resampling_method="sinc_interp_kaiser",  # 使用新的方法名
            lowpass_filter_width=64,
            rolloff=0.99,
            beta=14.769656459379492  # Kaiser窗参数，平衡通带纹波和阻带衰减
        )

    def resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """高质量重采样"""
        if orig_sr == target_sr:
            return audio

        key = (orig_sr, target_sr)
        if key not in self.resamplers:
            if len(self.resamplers) >= self.cache_size:
                # 清理最旧的重采样器
                oldest_key = next(iter(self.resamplers))
                del self.resamplers[oldest_key]

            self.resamplers[key] = self._create_high_quality_resampler(orig_sr, target_sr)

        return self.resamplers[key](audio)

class IndexTTS2:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_accel=False, use_torch_compile=False
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            use_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
            use_accel (bool): whether to use acceleration engine for GPT2 or not.
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
        """
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile

        # ========== 系统性属性初始化 - 避免所有AttributeError ==========
        # 进度引用显示（可选）
        self.gr_progress = None

        # 缓存参考音频
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

        # 模型相关属性
        self.semantic_model = None
        self.semantic_codec = None
        self.gpt = None
        self.s2mel = None
        self.bigvgan = None
        self.campplus_model = None
        self.extract_features = None
        self.normalizer = None
        self.tokenizer = None

        # 统计和矩阵属性
        self.semantic_mean = None
        self.semantic_std = None
        self.emo_matrix = None
        self.spk_matrix = None
        self.emo_num = None

        # mel_fn函数 - 使用正确的导入路径
        try:
            from indextts.s2mel.modules.audio import mel_spectrogram
            mel_fn_args = {
                "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
                "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
                "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
                "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
                "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
                "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
                "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
                "center": False
            }
            self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
        except ImportError as e:
            print(f"[WARNING] mel_spectrogram导入失败: {e}")
            print("[WARNING] 将在后续初始化中重试")
            self.mel_fn = None

        # 模型版本
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

        # 高质量重采样器
        self.advanced_resampler = AdvancedResampler()

        # 高级音频系统（第二阶段改进）
        if ADVANCED_SYSTEMS_AVAILABLE:
            self.speaker_embedding_cache = SpeakerEmbeddingCache(
                cache_size=200,
                similarity_threshold=0.95,
                enable_multi_sample_fusion=True
            )
            self.voice_consistency_controller = VoiceConsistencyController(
                consistency_threshold=0.8,
                adaptation_rate=0.1
            )
            self.quality_monitor = AdaptiveQualityMonitor()
            print("[IndexTTS2] ✓ 高级音频系统初始化完成")
        else:
            self.speaker_embedding_cache = None
            self.voice_consistency_controller = None
            self.quality_monitor = None

        # AI增强系统（第三阶段改进）
        if AI_ENHANCED_SYSTEMS_AVAILABLE:
            self.parameter_learner = IntelligentParameterLearner()
            self.audio_enhancer = AdaptiveAudioEnhancer(self.parameter_learner)
            self.quality_predictor = IntelligentQualityPredictor(self.parameter_learner)
            self.adaptive_cache_strategy = AdaptiveCacheStrategy(self.parameter_learner)

            # 将自适应缓存策略集成到现有缓存系统
            if self.speaker_embedding_cache:
                self.speaker_embedding_cache.adaptive_cache_strategy = self.adaptive_cache_strategy

            print("[IndexTTS2] ✓ AI增强系统初始化完成")
        else:
            self.parameter_learner = None
            self.audio_enhancer = None
            self.quality_predictor = None
            self.adaptive_cache_strategy = None
            print("[IndexTTS2] ⚠️ 高级音频系统不可用，使用基础功能")

        print("[IndexTTS2] ✓ 属性初始化完成")

        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))

        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)

        if use_deepspeed:
            try:
                import deepspeed
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")

        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

                print(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                print(f"{e!r}")
                self.use_cuda_kernel = False

        # 检查本地是否已有w2v-bert模型文件
        from indextts.utils.model_cache_manager import get_indextts2_cache_dir
        cache_dir = get_indextts2_cache_dir()

        # 初始化local_w2v_path变量
        local_w2v_path = None

        # 检查所有可能的HuggingFace缓存格式
        hf_cache_paths = [
            # 标准external_models缓存
            cache_dir / "w2v_bert" / "models--facebook--w2v-bert-2.0",
            cache_dir / "w2v_bert" / "facebook_w2v-bert-2.0",
            # HuggingFace Hub缓存
            cache_dir / "huggingface" / "hub" / "models--facebook--w2v-bert-2.0",
            cache_dir / "huggingface" / "transformers" / "models--facebook--w2v-bert-2.0",
            # 其他可能的格式
            cache_dir / "models--facebook--w2v-bert-2.0",
            cache_dir.parent / "w2v_bert" / "models--facebook--w2v-bert-2.0",
        ]

        # 查找HuggingFace缓存中的snapshots目录
        for hf_path in hf_cache_paths:
            if hf_path.exists():
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            config_file = snapshot / "config.json"
                            model_file = snapshot / "model.safetensors"
                            preprocessor_file = snapshot / "preprocessor_config.json"
                            if config_file.exists() and model_file.exists():
                                local_w2v_path = snapshot
                                print(f"[IndexTTS2] 发现本地w2v-bert模型 (HuggingFace缓存): {local_w2v_path}")
                                break
                    if local_w2v_path:
                        break

        # 如果HuggingFace缓存中没有找到，检查直接路径
        if not local_w2v_path:
            direct_paths = [
                cache_dir / "w2v_bert",  # 标准缓存路径
                cache_dir,  # 直接在external_models目录
                cache_dir.parent / "w2v_bert",  # 上一级目录的w2v_bert文件夹
            ]
            for path in direct_paths:
                config_file = path / "config.json"
                model_file = path / "model.safetensors"
                if config_file.exists() and model_file.exists():
                    local_w2v_path = path
                    print(f"[IndexTTS2] 发现本地w2v-bert模型 (直接路径): {local_w2v_path}")
                    break

        # 加载SeamlessM4TFeatureExtractor
        if local_w2v_path:
            print(f"[IndexTTS2] 使用本地w2v-bert模型: {local_w2v_path}")
            self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
                str(local_w2v_path),
                local_files_only=True
            )
        else:
            print(f"[IndexTTS2] 本地未找到w2v-bert模型，尝试从远程下载...")
            from indextts.utils.model_cache_manager import get_hf_download_kwargs
            w2v_kwargs = get_hf_download_kwargs("facebook/w2v-bert-2.0")
            self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0", **w2v_kwargs
            )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)

        # 检查本地是否已有MaskGCT语义编解码器
        local_maskgct_path = None

        # 检查所有可能的HuggingFace缓存格式
        maskgct_cache_paths = [
            # 标准external_models缓存
            cache_dir / "maskgct" / "models--amphion--MaskGCT",
            cache_dir / "maskgct" / "amphion_MaskGCT",
            # HuggingFace Hub缓存
            cache_dir / "huggingface" / "hub" / "models--amphion--MaskGCT",
            cache_dir / "huggingface" / "transformers" / "models--amphion--MaskGCT",
            # 其他可能的格式
            cache_dir / "models--amphion--MaskGCT",
            cache_dir.parent / "maskgct" / "models--amphion--MaskGCT",
        ]

        # 查找HuggingFace缓存中的snapshots目录
        for hf_path in maskgct_cache_paths:
            if hf_path.exists():
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            semantic_codec_file = snapshot / "semantic_codec" / "model.safetensors"
                            if semantic_codec_file.exists():
                                local_maskgct_path = semantic_codec_file
                                print(f"[IndexTTS2] 发现本地MaskGCT语义编解码器 (HuggingFace缓存): {local_maskgct_path}")
                                break
                    if local_maskgct_path:
                        break

        # 加载MaskGCT语义编解码器
        if local_maskgct_path:
            print(f"[IndexTTS2] 使用本地MaskGCT语义编解码器: {local_maskgct_path}")
            semantic_code_ckpt = str(local_maskgct_path)
        else:
            print(f"[IndexTTS2] 本地未找到MaskGCT语义编解码器，尝试从远程下载...")
            from indextts.utils.model_cache_manager import get_hf_download_kwargs
            maskgct_kwargs = get_hf_download_kwargs("amphion/MaskGCT")
            semantic_code_ckpt = hf_hub_download(
                "amphion/MaskGCT",
                filename="semantic_codec/model.safetensors",
                **maskgct_kwargs
            )
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        if self.use_torch_compile:
            print(">> Enabling torch.compile optimization for S2Mel")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")
        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)

        # 检查本地是否已有CAMPPlus模型
        local_campplus_path = None

        # 检查所有可能的HuggingFace缓存格式
        campplus_cache_paths = [
            # 标准external_models缓存
            cache_dir / "campplus" / "models--funasr--campplus",
            cache_dir / "campplus" / "funasr_campplus",
            # HuggingFace Hub缓存
            cache_dir / "huggingface" / "hub" / "models--funasr--campplus",
            cache_dir / "huggingface" / "transformers" / "models--funasr--campplus",
            # 其他可能的格式
            cache_dir / "models--funasr--campplus",
            cache_dir.parent / "campplus" / "models--funasr--campplus",
        ]

        # 查找HuggingFace缓存中的snapshots目录
        for hf_path in campplus_cache_paths:
            if hf_path.exists():
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            campplus_file = snapshot / "campplus_cn_common.bin"
                            if campplus_file.exists():
                                local_campplus_path = campplus_file
                                print(f"[IndexTTS2] 发现本地CAMPPlus模型 (HuggingFace缓存): {local_campplus_path}")
                                break
                    if local_campplus_path:
                        break

        # 加载CAMPPlus模型
        if local_campplus_path:
            print(f"[IndexTTS2] 使用本地CAMPPlus模型: {local_campplus_path}")
            campplus_ckpt_path = str(local_campplus_path)
        else:
            print(f"[IndexTTS2] 本地未找到CAMPPlus模型，尝试从远程下载...")
            from indextts.utils.model_cache_manager import get_hf_download_kwargs
            campplus_kwargs = get_hf_download_kwargs("funasr/campplus")
            campplus_ckpt_path = hf_hub_download(
                "funasr/campplus",
                filename="campplus_cn_common.bin",
                **campplus_kwargs
            )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_ckpt_path)

        bigvgan_name = self.cfg.vocoder.name
        # 下载BigVGAN到ComfyUI模型目录
        from indextts.utils.model_cache_manager import get_bigvgan_download_kwargs
        bigvgan_kwargs = get_bigvgan_download_kwargs(bigvgan_name)
        
        # 检查本地是否已有BigVGAN模型文件
        local_bigvgan_path = None

        # 检查多个可能的本地路径
        local_bigvgan_paths = [
            cache_dir / "bigvgan",  # 标准缓存路径
            cache_dir,  # 直接在external_models目录
            cache_dir.parent / "bigvgan",  # 上一级目录的bigvgan文件夹
        ]
        
        # 检查所有可能的HuggingFace缓存格式
        hf_cache_paths = [
            # 标准external_models缓存
            cache_dir / "bigvgan" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
            cache_dir / "bigvgan" / "nvidia_bigvgan_v2_22khz_80band_256x",
            # HuggingFace Hub缓存
            cache_dir / "huggingface" / "hub" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
            cache_dir / "huggingface" / "transformers" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
            # 其他可能的格式
            cache_dir / "models--nvidia--bigvgan_v2_22khz_80band_256x",
            cache_dir.parent / "bigvgan" / "models--nvidia--bigvgan_v2_22khz_80band_256x",
        ]
        
        # 查找HuggingFace缓存中的snapshots目录
        for hf_path in hf_cache_paths:
            if hf_path.exists():
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            config_file = snapshot / "config.json"
                            model_file = snapshot / "bigvgan_generator.pt"
                            if config_file.exists() and model_file.exists():
                                local_bigvgan_path = snapshot
                                print(f"[IndexTTS2] 发现本地BigVGAN模型 (HuggingFace缓存): {local_bigvgan_path}")
                                break
                    if local_bigvgan_path:
                        break
        
        # 如果HuggingFace缓存中没有找到，检查直接路径
        if not local_bigvgan_path:
            for path in local_bigvgan_paths:
                config_file = path / "config.json"
                model_file = path / "bigvgan_generator.pt"
                if config_file.exists() and model_file.exists():
                    local_bigvgan_path = path
                    print(f"[IndexTTS2] 发现本地BigVGAN模型 (直接路径): {local_bigvgan_path}")
                    break
        
        # 添加超时和错误处理的BigVGAN加载
        print(f"[IndexTTS2] 开始加载BigVGAN模型: {bigvgan_name}")
        print(f"[IndexTTS2] 缓存目录: {bigvgan_kwargs['cache_dir']}")
        
        # 检查系统是否支持signal.SIGALRM (Windows不支持)
        import threading
        import platform
        import signal
        
        # 检查是否在主线程中运行
        import threading
        is_main_thread = threading.current_thread() is threading.main_thread()
        
        if platform.system() == "Windows" or not hasattr(signal, 'SIGALRM') or not is_main_thread:
            # Windows系统、没有SIGALRM或不在主线程中，使用threading超时机制
            print("[IndexTTS2] 使用threading超时机制 (跨平台兼容)")
            
            def load_bigvgan_with_timeout():
                try:
                    # 优先使用本地路径
                    if local_bigvgan_path:
                        print(f"[IndexTTS2] 使用本地BigVGAN模型: {local_bigvgan_path}")
                        self.bigvgan = bigvgan.BigVGAN.from_pretrained(
                            str(local_bigvgan_path),  # 使用本地路径
                            use_cuda_kernel=False,
                            cache_dir=bigvgan_kwargs["cache_dir"]
                        )
                    else:
                        print(f"[IndexTTS2] 从HuggingFace下载BigVGAN模型: {bigvgan_name}")
                        self.bigvgan = bigvgan.BigVGAN.from_pretrained(
                            bigvgan_name,  # 使用远程ID
                            use_cuda_kernel=False,
                            cache_dir=bigvgan_kwargs["cache_dir"]
                        )

                    print("[IndexTTS2] 开始后处理BigVGAN模型...")

                    # 检查GPU内存
                    if isinstance(self.device, str) and self.device.startswith('cuda'):
                        torch.cuda.empty_cache()  # 清理GPU缓存
                        print(f"[IndexTTS2] GPU内存清理完成")
                    elif hasattr(self.device, 'type') and self.device.type == 'cuda':
                        torch.cuda.empty_cache()  # 清理GPU缓存
                        print(f"[IndexTTS2] GPU内存清理完成")

                    # 移动模型到设备
                    print(f"[IndexTTS2] 将BigVGAN模型移动到设备: {self.device}")
                    self.bigvgan = self.bigvgan.to(self.device)
                    print("[IndexTTS2] ✓ 模型移动完成")

                    # 移除权重归一化
                    print("[IndexTTS2] 移除权重归一化...")
                    self.bigvgan.remove_weight_norm()
                    print("[IndexTTS2] ✓ 权重归一化移除完成")

                    # 设置为评估模式
                    print("[IndexTTS2] 设置模型为评估模式...")
                    self.bigvgan.eval()
                    print("[IndexTTS2] ✓ 评估模式设置完成")

                    return True
                except Exception as e:
                    print(f"[ERROR] BigVGAN模型加载失败: {e}")
                    return False
            
            # 使用线程和超时
            result = [False]
            exception = [None]
            
            def target():
                try:
                    result[0] = load_bigvgan_with_timeout()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=300)  # 5分钟超时
            
            if thread.is_alive():
                print("[ERROR] BigVGAN模型加载超时，可能是网络问题")
                print("[ERROR] 请检查网络连接或尝试使用代理")
                raise TimeoutError("BigVGAN模型加载超时")
            
            if exception[0]:
                raise exception[0]
            
            if not result[0]:
                raise RuntimeError("BigVGAN模型加载失败")

            print(">> bigvgan weights restored from:", local_bigvgan_path if local_bigvgan_path else bigvgan_name)
            
        else:
            # Unix/Linux系统且在主线程中，使用signal超时机制
            print("[IndexTTS2] 使用signal超时机制 (Unix/Linux主线程)")
            
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("BigVGAN模型加载超时")
            
            # 设置超时（5分钟）
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5分钟超时
            
            try:
                # 优先使用本地路径
                if local_bigvgan_path:
                    print(f"[IndexTTS2] 使用本地BigVGAN模型: {local_bigvgan_path}")
                    self.bigvgan = bigvgan.BigVGAN.from_pretrained(
                        str(local_bigvgan_path),  # 使用本地路径
                        use_cuda_kernel=False,
                        cache_dir=bigvgan_kwargs["cache_dir"]
                    )
                else:
                    print(f"[IndexTTS2] 从HuggingFace下载BigVGAN模型: {bigvgan_name}")
                    self.bigvgan = bigvgan.BigVGAN.from_pretrained(
                        bigvgan_name,  # 使用远程ID
                        use_cuda_kernel=False,
                        cache_dir=bigvgan_kwargs["cache_dir"]
                    )

                print("[IndexTTS2] 开始后处理BigVGAN模型...")

                # 检查GPU内存
                if isinstance(self.device, str) and self.device.startswith('cuda'):
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    print(f"[IndexTTS2] GPU内存清理完成")
                elif hasattr(self.device, 'type') and self.device.type == 'cuda':
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    print(f"[IndexTTS2] GPU内存清理完成")

                # 移动模型到设备
                print(f"[IndexTTS2] 将BigVGAN模型移动到设备: {self.device}")
                self.bigvgan = self.bigvgan.to(self.device)
                print("[IndexTTS2] ✓ 模型移动完成")

                # 移除权重归一化
                print("[IndexTTS2] 移除权重归一化...")
                self.bigvgan.remove_weight_norm()
                print("[IndexTTS2] ✓ 权重归一化移除完成")

                # 设置为评估模式
                print("[IndexTTS2] 设置模型为评估模式...")
                self.bigvgan.eval()
                print("[IndexTTS2] ✓ 评估模式设置完成")

                signal.alarm(0)  # 取消超时

                print(">> bigvgan weights restored from:", local_bigvgan_path if local_bigvgan_path else bigvgan_name)
                
            except TimeoutError:
                signal.alarm(0)
                print("[ERROR] BigVGAN模型加载超时，可能是网络问题")
                print("[ERROR] 请检查网络连接或尝试使用代理")
                raise
            except Exception as e:
                signal.alarm(0)
                print(f"[ERROR] BigVGAN模型加载失败: {e}")
                print(f"[ERROR] 模型名称: {bigvgan_name}")
                print(f"[ERROR] 缓存目录: {bigvgan_kwargs['cache_dir']}")
                if local_bigvgan_path:
                    print(f"[ERROR] 本地路径: {local_bigvgan_path}")
                raise

        # 检查BPE模型文件路径
        bpe_filename = self.cfg.dataset["bpe_model"]

        # 构建可能的BPE文件路径，使用多种方法确保兼容性
        possible_bpe_paths = []

        # 方法1: 直接使用os.path.join（保持原有兼容性）
        possible_bpe_paths.append(os.path.join(self.model_dir, bpe_filename))
        possible_bpe_paths.append(os.path.join(self.model_dir, "bpe_model.model"))

        # 方法2: 如果model_dir指向checkpoints，尝试上一级目录
        parent_dir = os.path.dirname(self.model_dir)
        possible_bpe_paths.append(os.path.join(parent_dir, bpe_filename))

        # 方法3: 相对于当前脚本的路径
        script_dir = os.path.dirname(__file__)
        possible_bpe_paths.append(os.path.join(script_dir, "..", "bpe_model.model"))

        # 方法4: 在当前工作目录查找
        possible_bpe_paths.append(bpe_filename)
        possible_bpe_paths.append("bpe_model.model")

        self.bpe_path = None
        for path in possible_bpe_paths:
            if os.path.exists(path):
                self.bpe_path = path
                print(f"[IndexTTS2] 发现BPE模型文件: {self.bpe_path}")
                break

        if not self.bpe_path:
            raise FileNotFoundError(f"BPE模型文件未找到: {bpe_filename}")

        print("[IndexTTS2] 开始创建TextNormalizer...")
        import platform
        current_os = platform.system()
        print(f"[IndexTTS2] 检测到操作系统: {current_os}")

        if current_os == "Windows":
            print("[IndexTTS2] Windows系统，尝试使用完整版TextNormalizer...")
            try:
                # 尝试使用完整版TextNormalizer
                from indextts.utils.front import TextNormalizer
                self.normalizer = TextNormalizer()
                self.normalizer.load()
                print("[IndexTTS2] ✓ 使用完整版TextNormalizer")
                print(">> TextNormalizer loaded")
            except Exception as e:
                print(f"[IndexTTS2] 完整版TextNormalizer加载失败: {e}")
                print("[IndexTTS2] 回退到简化版TextNormalizer...")
                try:
                    self.normalizer = self._create_fallback_normalizer()
                    print("[IndexTTS2] ✓ 使用简化版TextNormalizer（回退方案）")
                    print(">> TextNormalizer loaded")
                except Exception as fallback_e:
                    print(f"[ERROR] 简化版TextNormalizer也失败: {fallback_e}")
                    raise RuntimeError(f"TextNormalizer初始化失败: {fallback_e}")
        else:
            print(f"[IndexTTS2] 非Windows系统（{current_os}），使用简化版TextNormalizer...")
            try:
                # --- FIX: Attempt Smart Normalizer First on ALL Platforms ---
                from indextts.utils.front import TextNormalizer
                self.normalizer = TextNormalizer()
                self.normalizer.load()
                print("[IndexTTS2] ✓ Using Standard TextNormalizer")
            except Exception as e:
                print(f"[IndexTTS2] Standard TextNormalizer failed ({e}), using Fallback.")
                self.normalizer = self._create_fallback_normalizer()
            print(">> TextNormalizer loaded")

        # 创建TextTokenizer
        try:
            print(f"[IndexTTS2] 开始创建TextTokenizer，BPE路径: {self.bpe_path}")
            self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
            print("[IndexTTS2] ✓ TextTokenizer创建完成")
            print(">> bpe model loaded from:", self.bpe_path)
        except Exception as e:
            print(f"[ERROR] 创建TextTokenizer失败: {e}")
            raise RuntimeError(f"TextTokenizer初始化失败: {e}")

        # 加载情感和说话人矩阵
        try:
            print(f"[IndexTTS2] 开始加载情感矩阵: {self.cfg.emo_matrix}")
            emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
            self.emo_matrix = emo_matrix.to(self.device)
            self.emo_num = list(self.cfg.emo_num)
            print("[IndexTTS2] ✓ 情感矩阵加载完成")

            print(f"[IndexTTS2] 开始加载说话人矩阵: {self.cfg.spk_matrix}")
            spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
            self.spk_matrix = spk_matrix.to(self.device)
            print("[IndexTTS2] ✓ 说话人矩阵加载完成")

            # 执行矩阵分割
            self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
            self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)
            print(f"[IndexTTS2] ✓ 矩阵分割完成")

        except Exception as e:
            print(f"[ERROR] 矩阵加载失败: {e}")
            raise RuntimeError(f"矩阵加载失败: {e}")

        # 后备mel_fn初始化（如果前面失败了）
        if self.mel_fn is None:
            try:
                from indextts.s2mel.modules.audio import mel_spectrogram
                mel_fn_args = {
                    "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
                    "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
                    "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
                    "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
                    "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
                    "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
                    "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
                    "center": False
                }
                self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
                print("[IndexTTS2] ✓ mel_fn后备初始化成功")
            except Exception as e:
                print(f"[ERROR] mel_fn后备初始化也失败: {e}")
                raise RuntimeError(f"无法初始化mel_fn函数: {e}")

        print("[IndexTTS2] ✓ IndexTTS2初始化完成")

    def _create_fallback_normalizer(self):
        """创建一个增强的TextNormalizer作为回退方案，包含数字转换功能"""
        class EnhancedFallbackTextNormalizer:
            def __init__(self):
                self.zh_normalizer = self._create_enhanced_normalizer()
                self.en_normalizer = self._create_simple_normalizer()

            def _create_enhanced_normalizer(self):
                class EnhancedNormalizer:
                    def __init__(self):
                        # 中文数字映射
                        self.digit_map = {
                            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
                        }
                        self.unit_map = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿']

                    def number_to_chinese(self, num_str):
                        """将数字字符串转换为中文"""
                        try:
                            num = int(num_str)
                            if num == 0:
                                return '零'

                            # 扩展的数字转换（支持0-99999999）
                            if num < 10:
                                return self.digit_map[str(num)]
                            elif num < 100:
                                tens = num // 10
                                ones = num % 10
                                if tens == 1:
                                    result = '十'
                                else:
                                    result = self.digit_map[str(tens)] + '十'
                                if ones > 0:
                                    result += self.digit_map[str(ones)]
                                return result
                            elif num < 1000:
                                hundreds = num // 100
                                remainder = num % 100
                                result = self.digit_map[str(hundreds)] + '百'
                                if remainder > 0:
                                    if remainder < 10:
                                        result += '零' + self.digit_map[str(remainder)]
                                    else:
                                        result += self.number_to_chinese(str(remainder))
                                return result
                            elif num < 10000:
                                thousands = num // 1000
                                remainder = num % 1000
                                result = self.digit_map[str(thousands)] + '千'
                                if remainder > 0:
                                    if remainder < 100:
                                        result += '零' + self.number_to_chinese(str(remainder))
                                    else:
                                        result += self.number_to_chinese(str(remainder))
                                return result
                            elif num < 100000:
                                # 万级别处理
                                wan = num // 10000
                                remainder = num % 10000
                                if wan == 1:
                                    result = '一万'
                                else:
                                    result = self.number_to_chinese(str(wan)) + '万'
                                if remainder > 0:
                                    if remainder < 1000:
                                        result += '零' + self.number_to_chinese(str(remainder))
                                    else:
                                        result += self.number_to_chinese(str(remainder))
                                return result
                            elif num < 1000000:
                                # 十万级别
                                wan = num // 10000
                                remainder = num % 10000
                                result = self.number_to_chinese(str(wan)) + '万'
                                if remainder > 0:
                                    if remainder < 1000:
                                        result += '零' + self.number_to_chinese(str(remainder))
                                    else:
                                        result += self.number_to_chinese(str(remainder))
                                return result
                            else:
                                # 对于更大的数字，简化处理
                                if num < 100000000:  # 一亿以下
                                    wan = num // 10000
                                    remainder = num % 10000
                                    result = self.number_to_chinese(str(wan)) + '万'
                                    if remainder > 0:
                                        if remainder < 1000:
                                            result += '零' + self.number_to_chinese(str(remainder))
                                        else:
                                            result += self.number_to_chinese(str(remainder))
                                    return result
                                else:
                                    # 超大数字，逐位转换
                                    return ''.join(self.digit_map.get(d, d) for d in num_str)
                        except:
                            # 如果转换失败，逐位转换
                            return ''.join(self.digit_map.get(d, d) for d in num_str)

                    def normalize(self, text):
                        import re

                        # 基本的文本清理
                        text = re.sub(r'["""]', '"', text)
                        text = re.sub(r"[''']", "'", text)
                        text = re.sub(r'[…]', '...', text)
                        text = re.sub(r'[—–]', '-', text)

                        # 数字转换（匹配连续的数字）
                        def replace_numbers(match):
                            number = match.group()
                            return self.number_to_chinese(number)

                        # 转换连续的数字
                        text = re.sub(r'\d+', replace_numbers, text)

                        # 标准化空白字符
                        text = re.sub(r'\s+', ' ', text)
                        text = text.strip()
                        return text

                return EnhancedNormalizer()

            def _create_simple_normalizer(self):
                class SimpleNormalizer:
                    def normalize(self, text):
                        import re
                        # 英文数字保持不变，只做基本清理
                        text = re.sub(r'["""]', '"', text)
                        text = re.sub(r"[''']", "'", text)
                        text = re.sub(r'[…]', '...', text)
                        text = re.sub(r'[—–]', '-', text)
                        text = re.sub(r'\s+', ' ', text)
                        text = text.strip()
                        return text
                return SimpleNormalizer()

            def normalize(self, text, lang="zh"):
                if lang == "zh":
                    return self.zh_normalizer.normalize(text)
                else:
                    return self.en_normalizer.normalize(text)

            def load(self):
                """兼容原始TextNormalizer接口的load方法"""
                # 简化版本不需要加载外部资源，直接返回
                pass

        return EnhancedFallbackTextNormalizer()

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def _load_and_cut_audio(self,audio_path,max_audio_length_seconds,verbose=False,sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path,sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr
    
    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector

    # 原始推理模式
    def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_sentence=120, speed=1.0, **generation_kwargs): # <--- Added speed
        
        # Check if speed is passed in kwargs (common in ComfyUI wrappers)
        if 'speed' in generation_kwargs:
            speed = generation_kwargs.pop('speed')

        if 'stream_return' in generation_kwargs:
            stream_return = generation_kwargs.pop('stream_return')
            return self.infer_generator(
                spk_audio_prompt, text, output_path,
                emo_audio_prompt, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_sentence, stream_return,
                speed=speed, # <--- Pass speed
                **generation_kwargs
            )
        else:
            try:
                return list(self.infer_generator(
                    spk_audio_prompt, text, output_path,
                    emo_audio_prompt, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_sentence,
                    speed=speed, # <--- Pass speed
                    **generation_kwargs
                ))[0]
            except IndexError:
                return None

    def infer_generator(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_sentence=120, stream_return=False, quick_streaming_tokens=0, 
              speed=1.0, # <--- Added speed argument
              **generation_kwargs):
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()
        
        # FIX: Remove extra kwargs (Corrected)
        generation_kwargs.pop('max_text_tokens_per_segment', None)
        generation_kwargs.pop('max_text_tokens_per_sentence', None)

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            
            # --- FIX: Qwen Inference with Lazy Check ---
            if self.qwen_emo:
                emo_dict, content = self.qwen_emo.inference(emo_text)
                print(f"[IndexTTS2] Qwen Raw Output: {content}")
                
                # Lazy Check
                if emo_dict.get("neutral", 0) > 0.6:
                    print(f"[IndexTTS2] ⚠️  Model output Neutral > 0.6. Checking Keywords...")
                    keyword_scores, found = self.qwen_emo._fallback_emotion_analysis(text)
                    if found and keyword_scores.get("neutral", 1.0) < 0.5:
                         print(f"[IndexTTS2] 🚀 Keywords Overrode Model: {keyword_scores}")
                         emo_dict = keyword_scores
                
                print(f"[IndexTTS2] Final Applied Emotion: {emo_dict}")
                emo_vector = list(emo_dict.values())
            else:
                emo_dict, _ = self.qwen_emo._fallback_emotion_analysis(text)
                emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # --- FIX: Apply Demo's Normalization Logic ---
            emo_vector = self.normalize_emo_vec(emo_vector, apply_bias=True)
            print(f"[IndexTTS2] Normalized Emotion Vector: {emo_vector}")
            
            # Original scaling logic
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio,sr = self._load_and_cut_audio(spk_audio_prompt,15,verbose)
            audio_22k = self.advanced_resampler.resample(audio, sr, 22050)
            audio_16k = self.advanced_resampler.resample(audio, sr, 16000)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                     num_mel_bins=80,
                                                     dither=0,
                                                     sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
            style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

            prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                     ylens=ref_target_lengths,
                                                                     n_quantizers=3,
                                                                     f0=None)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            # --- FIX: Matrix Construction ---
            try:
                selected_emotions = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
                selected_emotions = torch.cat(selected_emotions, 0)
                emovec_mat = weight_vector.unsqueeze(1) * selected_emotions
                emovec_mat = torch.sum(emovec_mat, 0).unsqueeze(0)
            except Exception as e:
                print(f"[IndexTTS2] Matrix Error: {e}. Using default.")
                emovec_mat = torch.zeros((1, self.emo_matrix[0].shape[1]), device=self.device)
        else:
            emovec_mat = torch.zeros((1, self.emo_matrix[0].shape[1]), device=self.device)

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt,15,verbose,sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"]
            emo_attention_mask = emo_inputs["attention_mask"]
            emo_input_features = emo_input_features.to(self.device)
            emo_attention_mask = emo_attention_mask.to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        # --- FIX: Use split_sentences ---
        if hasattr(self.tokenizer, 'split_segments'):
             segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_sentence, quick_streaming_tokens = quick_streaming_tokens)
        else:
             segments = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
             
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):")
            print( "     Tokens which can't be encoded: ", [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id])
            print(f"     Consider updating the BPE model or modifying the text to avoid unknown tokens.")
                  
        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*segments, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None # for stream_return
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(0.2 + 0.7 * seg_idx / segments_count,
                                  f"speech synthesis {seg_idx + 1}/{segments_count}...")

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as segment tokens", text_token_syms == sent)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha
                    )

                    if emo_vector is not None:
                        # --- FIX: Matrix Mixing Logic ---
                        weight_sum = torch.clamp(torch.sum(weight_vector), 0.0, 1.0)
                        if weight_sum == 0: weight_sum = 0.8
                        # Incorporate emo_alpha influence
                        effective_weight = weight_sum * min(1.0, emo_alpha)
                        emovec = (1.0 - effective_weight) * emovec + effective_weight * emovec_mat

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs
                    )

                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                #                 if verbose:
                #                     print(codes, type(codes))
                #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                #                     print(f"code len: {code_lens}")

                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=self.device),
                        codes,
                        torch.tensor([codes.shape[-1]], device=self.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    latent = self.s2mel.models['gpt_layer'](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2) + latent
                    
                    # === SPEED FIX ===
                    # 1.72 is base duration. 
                    # If speed is 1.5, we divide duration by 1.5 to make it shorter (faster).
                    safe_speed = max(0.1, float(speed)) # Prevent division by zero
                    target_lengths = (code_lens * (1.72 / safe_speed)).long()
                    # =================

                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    vc_target = self.s2mel.models['cfm'].inference(
                        cat_condition, torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                        ref_mel, style, None, 25, inference_cfg_rate=0.7
                    )
                    vc_target = vc_target[:, :, ref_mel.size(-1):]

                    m_start_time = time.perf_counter()
                    # --- FIX: BIGVGAN CPU FORCE ---
                    wav = self.bigvgan.to("cpu")(vc_target.float().to("cpu")).squeeze().unsqueeze(0)
                    print(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    if silence == None:
                        silence = self.insert_interval_silence([torch.zeros(1, 100)], sampling_rate=sampling_rate, interval_silence=interval_silence)[0]
                    yield silence
        end_time = time.perf_counter()

        self._set_gr_progress(0.9, "saving audio...")
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        
        if self.quality_monitor is not None:
             try:
                 # 对最终音频进行质量评估
                 quality_assessment = self.quality_monitor.assess_quality(wav.float(), sampling_rate)

                 print(f"[IndexTTS2] 🎵 音频质量评估:")
                 print(f"  - 综合质量分数: {quality_assessment['overall_quality']:.3f}")
                 print(f"  - SNR: {quality_assessment['metrics']['snr']:.1f} dB")
                 print(f"  - THD: {quality_assessment['metrics']['thd']:.3f}")
                 print(f"  - 动态范围: {quality_assessment['metrics']['dynamic_range']:.1f} dB")
                 print(f"  - 峰值电平: {quality_assessment['metrics']['peak_level']:.1f} dB")

                 if quality_assessment['violations'] > 0:
                     print(f"  ⚠️  检测到 {quality_assessment['violations']} 项质量问题")
                     print(f"  ℹ️ 自动改进功能已禁用，使用原始音频")
                 else:
                     print(f"  ✅ 音频质量良好")

             except Exception as e:
                 print(f"[IndexTTS2] ⚠️ 质量监控失败: {e}")
        
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (sampling_rate, wav_data)

def find_most_similar_cosine(query_vector, matrix):
    try:
        query_vector = query_vector.float()
        matrix = matrix.float()

        # 检查输入的有效性
        if matrix.shape[0] == 0:
            print("[IndexTTS2] Warning: empty matrix in find_most_similar_cosine, returning 0")
            return 0

        if torch.isnan(query_vector).any() or torch.isinf(query_vector).any():
            print("[IndexTTS2] Warning: invalid query_vector in find_most_similar_cosine, returning 0")
            return 0

        similarities = F.cosine_similarity(query_vector, matrix, dim=1)

        # 检查相似度计算结果
        if torch.isnan(similarities).any() or torch.isinf(similarities).any():
            print("[IndexTTS2] Warning: invalid similarities in find_most_similar_cosine, returning 0")
            return 0

        most_similar_index = torch.argmax(similarities)

        # 确保索引在有效范围内
        index_value = most_similar_index.item()
        if index_value >= matrix.shape[0]:
            print(f"[IndexTTS2] Warning: computed index {index_value} >= matrix size {matrix.shape[0]}, using 0")
            return 0

        return index_value

    except Exception as e:
        print(f"[IndexTTS2] Error in find_most_similar_cosine: {e}")
        return 0

class QwenEmotion:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None; self.tokenizer = None; self.is_available = False
        self._initialize_default_attributes()
        
        try:
            if os.path.exists(model_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=True)
                # --- FIX: FORCE FLOAT32 FOR MAC CPU ---
                self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.float32, device_map="auto", local_files_only=True, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
            self.is_available = True
            print(f"[IndexTTS2] ✅ Qwen emotion model loaded (Float32)!")
        except Exception as e:
            print(f"[IndexTTS2] ⚠️  Qwen load failed: {e}")

    def _initialize_default_attributes(self):
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy", "愤怒": "angry", "悲伤": "sad", "恐惧": "afraid",
            "反感": "disgusted", "低落": "melancholic", "惊讶": "surprised", "自然": "calm",
            "anger": "angry", # Fix typo
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {"低落", "melancholy", "melancholic", "depression", "depressed", "gloomy", "overwhelmed", "worried", "tired", "exhausted"}
        self.convert_dict = self.cn_key_to_en
        self.backup_dict = {"happy": 0, "angry": 0, "sad": 0, "fear": 0, "hate": 0, "low": 0, "surprise": 0, "neutral": 1.0}
        self.min_score = 0.0; self.max_score = 1.2

    def clamp_score(self, value):
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content):
        import re
        # --- FIX: Robust Regex Parsing ---
        pattern = r'["\']?([a-zA-Z\u4e00-\u9fa5]+)["\']?\s*[:=]\s*([0-9.]+)'
        parsed_data = {}
        for match in re.finditer(pattern, str(content)):
            try: parsed_data[match.group(1)] = float(match.group(2))
            except: continue
        
        emotion_dict = {}
        en_keys = set(self.cn_key_to_en.values())
        legacy_map = {"neutral": "calm", "fear": "afraid", "hate": "disgust", "low": "melancholic", "surprise": "surprised"}

        for raw_key, val in parsed_data.items():
            target_key = None
            if raw_key in self.cn_key_to_en: target_key = self.cn_key_to_en[raw_key]
            elif raw_key in en_keys: target_key = raw_key
            elif raw_key in legacy_map: target_key = legacy_map[raw_key]
            
            if target_key: emotion_dict[target_key] = self.clamp_score(val)

        # Ensure all keys present and map back to Comfy expected keys
        final_comfy_dict = {
            "happy": emotion_dict.get("happy", 0.0),
            "angry": emotion_dict.get("angry", 0.0),
            "sad": emotion_dict.get("sad", 0.0),
            "fear": emotion_dict.get("afraid", 0.0), # Map back internal 'afraid' to 'fear'
            "hate": emotion_dict.get("disgust", 0.0), # Map back internal 'disgust' to 'hate'
            "low": emotion_dict.get("melancholic", 0.0), # Map back internal 'melancholic' to 'low'
            "surprise": emotion_dict.get("surprised", 0.0),
            "neutral": emotion_dict.get("calm", 0.0)
        }
        
        # If empty, default to neutral
        if all(v <= 0 for v in final_comfy_dict.values()):
             final_comfy_dict["neutral"] = 1.0
             
        return final_comfy_dict

    def inference(self, text_input):
        if not self.is_available or self.model is None:
            return self._fallback_emotion_analysis(text_input)[0], "Fallback"

        try:
            # --- FIX: Prompt Strategy ---
            messages = [{"role": "system", "content": f"{self.prompt}"}, {"role": "user", "content": f"{text_input}"}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            
            # --- FIX: CPU PATCH + DO_SAMPLE=TRUE ---
            model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
            generated_ids = self.model.to("cpu").generate(
                **model_inputs, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id, do_sample=True
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            try: index = len(output_ids) - output_ids[::-1].index(151668)
            except: index = 0
            
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
            print(f"[IndexTTS2] Qwen Raw Output: {content}")
            
            content_dict = self.convert(content)
            
            # --- FIX: Lazy Check ---
            if content_dict.get("neutral", 0) > 0.6:
                print(f"[IndexTTS2] ⚠️  Model output Neutral > 0.6. Checking Keywords...")
                keyword_scores, found = self._fallback_emotion_analysis(text_input)
                if found and keyword_scores.get("neutral", 1.0) < 0.5:
                     print(f"[IndexTTS2] 🚀 Keywords Overrode Model: {keyword_scores}")
                     content_dict = keyword_scores

            # Melancholy Fix
            text_input_lower = text_input.lower()
            if any(word in text_input_lower for word in self.melancholic_words):
                print("[IndexTTS2] Applying Melancholy Fix")
                sad_score = content_dict.get("sad", 0.0)
                low_score = content_dict.get("low", 0.0)
                content_dict["low"] = max(sad_score, low_score, 0.8) 
                content_dict["sad"] = 0.0

            return content_dict, content

        except Exception as e:
            print(f"[IndexTTS2] ⚠️ Inference failed: {e}")
            return self._fallback_emotion_analysis(text_input)[0], str(e)

    def _fallback_emotion_analysis(self, text_input):
        text_lower = text_input.lower()
        emotion_keywords = {
            "happy": ["happy", "excited", "glad", "joy", "great", "love", "高兴", "开心", "太棒"],
            "angry": ["angry", "mad", "hate", "stupid", "damn", "wrong", "fault", "blame", "生气", "愤怒", "滚", "不对", "检讨", "讨厌"],
            "sad": ["sad", "crying", "sorry", "miss", "难过", "悲伤", "哭", "遗憾"],
            "fear": ["scared", "fear", "afraid", "help", "worried", "worry", "害怕", "恐惧", "担心", "救命"],
            "hate": ["hate", "disgust", "sick", "repulsive", "反感", "恶心", "恨"],
            "low": ["tired", "exhausted", "overwhelmed", "depressed", "gloomy", "sigh", "累", "低落", "郁闷", "无助", "唉"],
            "surprise": ["wow", "shock", "surprise", "really", "what", "惊讶", "震惊", "真的"],
            "neutral": ["okay", "fine", "normal", "hello", "yes", "no", "还好"]
        }
        
        scores = {k: 0.0 for k in emotion_keywords.keys()}
        scores["neutral"] = 1.0
        found = False
        for emo, words in emotion_keywords.items():
            if emo == "neutral": continue
            for w in words:
                if w in text_lower:
                    scores[emo] = 0.9
                    scores["neutral"] = 0.0
                    found = True
                    break
        
        if found: scores["neutral"] = 0.0
        return scores, found

if __name__ == "__main__":
    pass
