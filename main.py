import os
import re
import sys
import time
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from urllib.parse import urljoin, quote

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import AstrBotConfig, logger


# ─────────────────────────────────────────────
#  Plugin metadata
# ─────────────────────────────────────────────
@register(
    "astrbot_plugin_hardwareinfo",
    "SakuraMikku",
    "硬件信息查询（CPU/GPU搜索+天梯图+参数图片）",
    "0.0.8",
    "https://github.com/wuxinTLH/astrbot_plugin_hardwareinfo",
)
class HardwareInfoPlugin(Star):

    # ══════════════════════════════════════════
    #  常量与类级别映射
    # ══════════════════════════════════════════

    # 参数中文翻译映射（大幅扩充）
    PARAM_CN_MAP: Dict[str, str] = {
        # ── 通用 ──────────────────────────────
        "Name": "名称",
        "Codename": "代号",
        "Architecture": "架构",
        "Manufacturer": "制造商",
        "Foundry": "代工厂",
        "Released": "发布日期",
        "Launch Date": "上市日期",
        "Launch Price": "首发价格",
        "Market Segment": "市场定位",
        "Part#": "型号编号",
        "OEM/Tray": "散片/OEM",
        # ── CPU 基本 ─────────────────────────
        "Socket": "接口类型",
        "Platform": "平台",
        "Core Count": "核心数",
        "Thread Count": "线程数",
        "Performance Cores": "性能核心数",
        "Efficiency Cores": "能效核心数",
        "Base Clock": "基础频率",
        "Boost Clock": "加速频率",
        "All-Core Boost": "全核加速频率",
        "TDP": "功耗(TDP)",
        "TDP Up": "最大功耗",
        "TDP Down": "最小功耗",
        "cTDP Up": "可配置最大功耗",
        "cTDP Down": "可配置最小功耗",
        "Base Power": "基础功耗",
        "Max Power": "最大功耗",
        "Process Size": "制程工艺",
        "Transistors": "晶体管数量",
        "Die Size": "核心面积",
        "Chips": "芯片数量",
        "Die Count": "晶粒数量",
        "Integrated Graphics": "核心显卡",
        "GPU Frequency": "核显频率",
        "Max GPU Frequency": "核显最大频率",
        # ── CPU 缓存 ─────────────────────────
        "L1 Cache": "L1 缓存",
        "L2 Cache": "L2 缓存",
        "L3 Cache": "L3 缓存",
        "L4 Cache": "L4 缓存",
        "L1 Instruction Cache": "L1 指令缓存",
        "L1 Data Cache": "L1 数据缓存",
        "Cache per Core": "每核缓存",
        # ── CPU 内存 ─────────────────────────
        "Memory Support": "内存支持",
        "Memory Type": "内存类型",
        "Memory Channels": "内存通道数",
        "Max Memory": "最大内存容量",
        "Max Memory Bandwidth": "最大内存带宽",
        "ECC Memory": "ECC 内存支持",
        "Memory Speed": "内存频率",
        # ── CPU 接口/扩展 ────────────────────
        "PCIe Version": "PCIe 版本",
        "PCIe Lanes": "PCIe 通道数",
        "Instruction Set": "指令集",
        "Instruction Set Extensions": "指令集扩展",
        "Virtualization": "虚拟化支持",
        "Thermal Solution": "散热方案",
        "Operating Temperature Max": "最高工作温度",
        "Operating Temperature Min": "最低工作温度",
        # ── GPU 基本 ─────────────────────────
        "GPU Name": "GPU 名称",
        "GPU Variant": "GPU 变种",
        "GPU Die": "GPU 核心",
        "Bus Interface": "总线接口",
        "Slot Width": "插槽宽度",
        "Length": "显卡长度",
        "Height": "显卡高度",
        "Width": "显卡厚度",
        "Cooler": "散热器类型",
        "LED Lighting": "LED 灯效",
        "Manufacturing Process": "制造工艺",
        # ── GPU 显存 ─────────────────────────
        "Memory Size": "显存容量",
        "Memory Bus": "显存位宽",
        "Memory Clock": "显存频率",
        "Memory Bandwidth": "显存带宽",
        "Memory Speed": "显存速率",
        "Effective Memory Clock": "等效显存频率",
        # ── GPU 渲染单元 ─────────────────────
        "Shading Units": "流处理器",
        "TMUs": "纹理单元 (TMU)",
        "ROPs": "光栅单元 (ROP)",
        "SM Count": "SM 单元数",
        "Compute Units": "计算单元 (CU)",
        "Execution Units": "执行单元 (EU)",
        "Shader Processors": "着色处理器",
        "Texture Units": "纹理处理单元",
        "Render Output Units": "渲染输出单元",
        # ── GPU AI/RT ────────────────────────
        "Tensor Cores": "张量核心",
        "RT Cores": "光线追踪核心",
        "Ray Accelerators": "光追加速器",
        "AI Accelerators": "AI 加速单元",
        # ── GPU 性能 ─────────────────────────
        "Pixel Fillrate": "像素填充率",
        "Texture Fillrate": "纹理填充率",
        "FP16 (half) performance": "FP16 半精度性能",
        "FP32 (float) performance": "FP32 单精度性能",
        "FP64 (double) performance": "FP64 双精度性能",
        "FP16 Performance": "FP16 性能",
        "FP32 Performance": "FP32 性能",
        "FP64 Performance": "FP64 性能",
        "INT8 Performance": "INT8 整数性能",
        "Tensor Performance": "张量运算性能",
        # ── GPU 电源 ─────────────────────────
        "Board Power": "板卡功耗 (TGP)",
        "TGP": "总图形功耗",
        "TBP": "总板卡功耗",
        "Suggested PSU": "建议电源",
        "Power Connectors": "供电接口",
        "Minimum PSU Recommendation": "最低电源推荐",
        # ── GPU 频率 ─────────────────────────
        "Base Clock": "基础频率",
        "Boost Clock": "加速频率",
        "Game Clock": "游戏频率",
        # ── GPU 显示输出 ─────────────────────
        "Outputs": "视频输出接口",
        "HDMI": "HDMI 接口",
        "DisplayPort": "DP 接口",
        "Max Resolution": "最大分辨率",
        "Multi Monitor": "多显示器支持",
        "HDCP": "HDCP 支持",
        # ── GPU 图形 API ─────────────────────
        "DirectX": "DirectX 版本",
        "OpenGL": "OpenGL 版本",
        "OpenCL": "OpenCL 版本",
        "Vulkan": "Vulkan 版本",
        "CUDA": "CUDA 计算能力",
        "Shader Model": "着色器模型版本",
        "Metal": "Metal 版本",
        "ROCm": "ROCm 版本",
        # ── GPU 编解码 ───────────────────────
        "Encode/Decode": "编解码支持",
        "H.264 Encode": "H.264 硬件编码",
        "H.264 Decode": "H.264 硬件解码",
        "H.265/HEVC Encode": "H.265 硬件编码",
        "H.265/HEVC Decode": "H.265 硬件解码",
        "AV1 Encode": "AV1 硬件编码",
        "AV1 Decode": "AV1 硬件解码",
        "VP9 Decode": "VP9 硬件解码",
        "VP8 Decode": "VP8 硬件解码",
        "Video Decode": "视频硬件解码",
        "Video Encode": "视频硬件编码",
        "NVENC": "NVIDIA 硬件编码器 (NVENC)",
        "NVDEC": "NVIDIA 硬件解码器 (NVDEC)",
        # ── GPU 特性 ─────────────────────────
        "NVLink": "NVLink 互联",
        "SLI": "SLI 多卡互联",
        "CrossFire": "CrossFire 多卡互联",
        "Infinity Fabric Link": "Infinity Fabric 互联",
        "Resizable BAR": "可调 BAR",
        "DirectML": "DirectML 支持",
        "ROCm AI": "ROCm AI 支持",
        "FidelityFX": "AMD FidelityFX",
        "FSR": "AMD FidelityFX 超级分辨率",
        "DLSS": "NVIDIA DLSS",
        "XeSS": "Intel XeSS 超采样",
    }

    # 详情页分区标题中英映射
    SECTION_CN_MAP: Dict[str, str] = {
        "Clock Speeds": "频率参数",
        "Clocks": "频率参数",
        "Memory": "显存规格",
        "Memory Specifications": "内存规格",
        "Render Config": "渲染配置",
        "Theoretical Performance": "理论性能",
        "Board Design": "显卡设计",
        "Graphics Features": "图形特性",
        "Features": "功能特性",
        "Physical": "物理参数",
        "Processor": "处理器信息",
        "Performance": "性能参数",
        "Architecture": "架构参数",
        "Core Config": "核心配置",
        "Cache": "缓存配置",
        "Expansion": "扩展接口",
        "Power Management": "电源管理",
        "Thermals": "散热参数",
        "General Specifications": "基本规格",
        "Overview": "基本概况",
        "Compute": "计算特性",
        "Video Features": "视频特性",
        "Video Output": "视频输出",
        "Compatibility": "兼容性",
        "Multimedia": "多媒体特性",
        "Display Support": "显示器支持",
        "API Support": "API 支持",
        "Encoding": "视频编码",
        "Decoding": "视频解码",
        "AI & Ray Tracing": "AI 与光线追踪",
        "NVIDIA Technologies": "NVIDIA 专有技术",
        "AMD Technologies": "AMD 专有技术",
        "Intel Technologies": "Intel 专有技术",
        "Multi-GPU Support": "多 GPU 支持",
    }

    # 跳过的无效分区
    SKIP_SECTIONS = frozenset({"Notes", "GB202 GPU Notes", "Disclaimer", "Errata"})

    # TechPowerUp 入口
    TPU_BASE = {
        "cpu": "https://www.techpowerup.com/cpu-specs/",
        "gpu": "https://www.techpowerup.com/gpu-specs/",
    }

    # 天梯图资源
    HARDWARE_RANKING = {
        "cpu": {
            "url": "https://pica.zhimg.com/v2-18344d446f16199d4208dd9149528834_1440w.webp?consumer=ZHI_MENG",
            "filename": "cpu_ranking.webp",
        },
        "gpu": {
            "url": "https://pic1.zhimg.com/v2-ca6724487c3bcd007598b20eb8693bc4_r.jpg",
            "filename": "gpu_ranking.jpg",
        },
    }

    # 反爬关键词
    VERIFY_KEYWORDS = frozenset({
        "automated bot check",
        "your browser must support javascript",
        "机器人验证",
        "cloudflare",
        "please enable javascript",
        "just a moment",
        "checking your browser",
        "ddos-guard",
        "ray id",
    })

    # 请求头模板
    BASE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-CH-UA": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    }

    # ══════════════════════════════════════════
    #  初始化
    # ══════════════════════════════════════════

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)

        # 基础配置
        self.cooldown_period: int = int(config.get("cooldown_period", 30))
        self.cache_ttl: int = int(config.get("cache_ttl", 120))
        self.temp_id_expire: int = int(config.get("temp_id_expire", 600))
        self.insecure_skip_verify: bool = bool(config.get("insecure_skip_verify", False))
        self.max_retries: int = min(int(config.get("max_retries", 3)), 5)
        self.request_timeout: int = int(config.get("request_timeout", 20))
        # 参数图片磁盘缓存有效期（秒），默认 7 天
        self.image_cache_ttl: int = int(config.get("image_cache_ttl", 7 * 24 * 3600))

        # Cookie
        self.custom_cookies: Dict[str, str] = dict(config.get("custom_cookies", {}) or {})
        env_cookie_json = os.environ.get("ASTR_PLUGIN_HW_COOKIES")
        if env_cookie_json:
            try:
                env_cookies = json.loads(env_cookie_json)
                if isinstance(env_cookies, dict):
                    self.custom_cookies.update(env_cookies)
                    logger.info("[HW] 已从环境变量加载自定义 Cookie")
                else:
                    logger.warning("[HW] 环境变量 ASTR_PLUGIN_HW_COOKIES 期望 JSON 对象")
            except Exception:
                logger.exception("[HW] 解析环境变量 ASTR_PLUGIN_HW_COOKIES 异常，已忽略")

        # 目录初始化
        try:
            data_dir = Path(StarTools.get_data_dir("astrbot_plugin_hardwareinfo"))
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            data_dir = Path(__file__).parent.resolve() / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

        self.base_dir = Path(__file__).parent.resolve()
        self.cache_dir = data_dir / "cache"
        self.param_image_dir = data_dir / "param_images"
        self.font_dir = self.base_dir / "fonts"

        for p in (self.cache_dir, self.param_image_dir, self.font_dir):
            try:
                p.mkdir(parents=True, exist_ok=True)
                if sys.platform.startswith("linux"):
                    os.chmod(str(p), 0o755)
            except Exception:
                logger.debug(f"[HW] 目录操作跳过：{p}")

        # 字体
        self.mandatory_font = str(self.font_dir / "simhei.ttf")
        self.system_fonts = [
            "simhei.ttf", "wqy-microhei.ttc", "NotoSansCJK-Regular.ttc",
            "NotoSansSC-Regular.otf", "PingFang.ttc", "Microsoft YaHei.ttf",
        ]

        # 内存缓存（身份隔离）
        # search_cache[identity][hw_type] = {"results": [...], "expire": float}
        self.search_cache: Dict[Tuple[str, str], Dict[str, Dict]] = {}
        # cooldown[identity][hw_type] = last_call_time (float)
        self.last_called_times: Dict[Tuple[str, str], Dict[str, float]] = {}
        # 磁盘参数图片 mtime 缓存，避免重复 stat
        self._img_mtime_cache: Dict[str, float] = {}
        self._cache_lock = asyncio.Lock()

        # 缓存清理任务句柄
        self._cleanup_task: Optional[asyncio.Task] = None

        # HTTP session
        self._http_session: Optional[aiohttp.ClientSession] = None

    # ══════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════

    async def initialize(self):
        logger.info("[HW] 硬件信息查询插件初始化")
        ssl_ctx = False if self.insecure_skip_verify else True
        if self.insecure_skip_verify:
            logger.warning("[HW] insecure_skip_verify=True，已跳过 TLS 验证（存在安全风险）")

        conn = TCPConnector(
            ssl=ssl_ctx,
            limit=8,            # 全局并发连接上限
            limit_per_host=4,   # 单域名并发上限，避免触发限速
            ttl_dns_cache=300,  # DNS 缓存 5 分钟
            use_dns_cache=True,
        )
        timeout = ClientTimeout(total=self.request_timeout, connect=8, sock_read=15)
        self._http_session = aiohttp.ClientSession(
            connector=conn,
            timeout=timeout,
            headers=self.BASE_HEADERS,
        )

        if not Path(self.mandatory_font).exists():
            logger.warning("[HW] 未找到 fonts/simhei.ttf，中文可能乱码")

        # 启动定时内存缓存清理（每 10 分钟）
        self._cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        logger.info("[HW] 缓存清理协程已启动")

    async def terminate(self):
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        try:
            if self._http_session and not self._http_session.closed:
                await self._http_session.close()
        except Exception:
            logger.exception("[HW] 关闭 HTTP session 异常")
        logger.info("[HW] 硬件信息查询插件已卸载")

    # ══════════════════════════════════════════
    #  定时缓存清理
    # ══════════════════════════════════════════

    async def _cache_cleanup_loop(self):
        """每 10 分钟清理过期的内存搜索缓存和磁盘参数图片。"""
        while True:
            try:
                await asyncio.sleep(600)
                await self._evict_expired_memory_cache()
                await asyncio.to_thread(self._evict_expired_disk_images)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[HW] 缓存清理异常")

    async def _evict_expired_memory_cache(self):
        now = time.time()
        async with self._cache_lock:
            dead_identities = []
            for identity, hw_map in self.search_cache.items():
                for hw_type in list(hw_map.keys()):
                    if now > hw_map[hw_type].get("expire", 0):
                        del hw_map[hw_type]
                if not hw_map:
                    dead_identities.append(identity)
            for identity in dead_identities:
                del self.search_cache[identity]
            # 同步清理过旧冷却记录（超过 1 小时未触发）
            stale_limit = now - 3600
            for identity in list(self.last_called_times.keys()):
                entry = self.last_called_times[identity]
                for hw_type in list(entry.keys()):
                    if entry[hw_type] < stale_limit:
                        del entry[hw_type]
                if not entry:
                    del self.last_called_times[identity]
        logger.debug("[HW] 内存缓存清理完成")

    def _evict_expired_disk_images(self):
        """清理超过 image_cache_ttl 的磁盘参数图片。"""
        now = time.time()
        cleaned = 0
        for img_file in self.param_image_dir.glob("*.png"):
            try:
                if now - img_file.stat().st_mtime > self.image_cache_ttl:
                    img_file.unlink(missing_ok=True)
                    cleaned += 1
            except Exception:
                pass
        if cleaned:
            logger.info(f"[HW] 清理过期参数图片 {cleaned} 个")

    # ══════════════════════════════════════════
    #  工具函数
    # ══════════════════════════════════════════

    def _get_identity(self, event: AstrMessageEvent) -> Tuple[str, str]:
        """多平台兼容的用户+群组身份提取。"""
        # user_id
        user_id = (
            getattr(event, "user_id", None)
            or getattr(getattr(event, "sender", None), "user_id", None)
            or getattr(getattr(event, "from_user", None), "id", None)
            or getattr(getattr(event, "author", None), "id", None)
        )
        user_id = str(user_id) if user_id else f"temp_{int(time.time() // self.temp_id_expire)}"

        # group_id
        group_id = (
            getattr(event, "group_id", None)
            or getattr(getattr(event, "session", None), "group_id", None)
            or getattr(getattr(event, "group", None), "id", None)
        )
        if group_id:
            group_id = str(group_id)
        else:
            group_id = f"private_{abs(hash(user_id)) % 100003}"

        return user_id, group_id

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\[At:[^\]]+\]", "", text)
        text = re.sub(r"<at[^>]*>.*?</at>", "", text, flags=re.I | re.S)
        text = text.strip().lstrip("/\\／﹨")
        return re.sub(r"\s+", " ", text).strip()

    def _is_on_cooldown(self, identity: Tuple[str, str], hw_type: str) -> Tuple[bool, int]:
        remaining = self.cooldown_period - (
            time.time() - self.last_called_times.get(identity, {}).get(hw_type, 0.0)
        )
        return (True, max(0, int(round(remaining)))) if remaining > 0 else (False, 0)

    def _is_verification_page(self, html: str) -> bool:
        if not html:
            return False
        lower = html.lower()
        return any(kw in lower for kw in self.VERIFY_KEYWORDS)

    def _translate_param(self, name: str) -> str:
        cleaned = name.strip().rstrip(":")
        return self.PARAM_CN_MAP.get(cleaned, cleaned)

    def _translate_section(self, name: str) -> str:
        return self.SECTION_CN_MAP.get(name.strip(), name.strip())

    @staticmethod
    def _safe_filename(name: str, max_len: int = 100) -> str:
        return re.sub(r'[\\/*?:"<>|]', "_", name)[:max_len]

    def _param_image_path(self, hw_type: str, hw_name: str) -> Path:
        """根据型号名称生成稳定的参数图片路径（含内容哈希，避免文件名冲突）。"""
        key = hashlib.md5(f"{hw_type}::{hw_name}".encode()).hexdigest()[:12]
        safe = self._safe_filename(hw_name)
        return self.param_image_dir / f"{safe}_{key}.png"

    # ══════════════════════════════════════════
    #  HTTP 会话管理
    # ══════════════════════════════════════════

    def _get_session(self) -> Tuple[aiohttp.ClientSession, bool]:
        """返回 (session, is_temp)。session 无效时临时创建。"""
        if self._http_session and not self._http_session.closed:
            return self._http_session, False
        ssl_ctx = False if self.insecure_skip_verify else True
        conn = TCPConnector(ssl=ssl_ctx, limit=4, ttl_dns_cache=300)
        session = aiohttp.ClientSession(
            connector=conn,
            timeout=ClientTimeout(total=self.request_timeout),
            headers=self.BASE_HEADERS,
        )
        return session, True

    # ══════════════════════════════════════════
    #  网络请求
    # ══════════════════════════════════════════

    async def _fetch_html(self, url: str, referer: str) -> Tuple[str, bool]:
        """
        通用 HTML 抓取。
        返回 (html_text, is_verify_page)。
        失败返回 ("", False)。
        """
        extra_headers = {"Referer": referer}
        session, is_temp = self._get_session()
        try:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(
                        url,
                        headers=extra_headers,
                        cookies=self.custom_cookies or None,
                        allow_redirects=True,
                    ) as resp:
                        if resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", 10))
                            logger.warning(f"[HW] 429 限速，等待 {retry_after}s (尝试{attempt+1})")
                            await asyncio.sleep(retry_after)
                            continue
                        if resp.status >= 500:
                            logger.warning(f"[HW] 服务端错误 {resp.status} (尝试{attempt+1})")
                            await asyncio.sleep(2 ** attempt)
                            continue
                        text = await resp.text(errors="replace")
                        if self._is_verification_page(text):
                            logger.info(f"[HW] 检测到验证页面：{url}")
                            return "", True
                        if resp.status == 200:
                            return text, False
                        logger.warning(f"[HW] 状态 {resp.status}：{url} (尝试{attempt+1})")
                except asyncio.CancelledError:
                    raise
                except (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError) as e:
                    logger.warning(f"[HW] 连接异常 (尝试{attempt+1}/{self.max_retries})：{e}")
                except aiohttp.ClientError as e:
                    logger.warning(f"[HW] 客户端异常 (尝试{attempt+1}/{self.max_retries})：{e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 8))
            logger.error(f"[HW] 请求彻底失败（{self.max_retries} 次）：{url}")
            return "", False
        finally:
            if is_temp:
                try:
                    await session.close()
                except Exception:
                    pass

    async def _get_ranking_image(self, hw_type: str) -> str:
        """获取天梯图，优先本地缓存。返回本地文件路径或空字符串。"""
        info = self.HARDWARE_RANKING.get(hw_type)
        if not info:
            return ""
        local_path = self.cache_dir / info["filename"]
        if local_path.exists() and local_path.stat().st_size > 1024:
            return str(local_path)

        logger.info(f"[HW] 下载天梯图：{info['url']}")
        session, is_temp = self._get_session()
        try:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(
                        info["url"],
                        timeout=ClientTimeout(total=45),
                    ) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        content = await resp.read()
                        if len(content) < 512:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_bytes(content)
                        if sys.platform.startswith("linux"):
                            try:
                                os.chmod(str(local_path), 0o644)
                            except Exception:
                                pass
                        logger.info(f"[HW] 天梯图保存：{local_path}")
                        return str(local_path)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"[HW] 下载天梯图异常 (尝试{attempt+1})：{e}")
                await asyncio.sleep(2 ** attempt)
        finally:
            if is_temp:
                try:
                    await session.close()
                except Exception:
                    pass
        return ""

    # ══════════════════════════════════════════
    #  解析逻辑（在线程中执行）
    # ══════════════════════════════════════════

    def _parse_search_results(self, hw_type: str, html: str) -> List[Dict[str, str]]:
        """解析 TechPowerUp 搜索结果列表。"""
        results: List[Dict[str, str]] = []
        if not html:
            return results
        soup = BeautifulSoup(html, "lxml")
        try:
            if hw_type == "cpu":
                table = soup.select_one("table.items-desktop-table")
                if not table:
                    logger.warning("[HW][CPU] 未找到搜索结果表格")
                    return results
                for tr in table.find_all("tr", recursive=False):
                    if tr.find("th", recursive=False):
                        continue
                    first_td = tr.find("td", recursive=False)
                    if not first_td:
                        continue
                    a = first_td.find("a", class_=lambda c: c != "item-image-link")
                    if a and a.get_text(strip=True):
                        results.append({
                            "name": a.get_text(strip=True),
                            "detail_url": urljoin(self.TPU_BASE["cpu"], a.get("href", "")),
                        })
            else:
                # GPU：先尝试 tr-based，再 fallback 到 td-grouping
                table = soup.select_one("div#list table.items-desktop-table") or \
                        soup.select_one("table.items-desktop-table")
                if not table:
                    logger.warning("[HW][GPU] 未找到搜索结果表格")
                    return results
                # 优先按 tr 解析，更鲁棒
                for tr in table.find_all("tr"):
                    if tr.find("th"):
                        continue
                    name_div = tr.find("div", class_="item-name")
                    if not name_div:
                        continue
                    a = name_div.find("a", recursive=False)
                    if a and a.get_text(strip=True):
                        results.append({
                            "name": a.get_text(strip=True),
                            "detail_url": urljoin(self.TPU_BASE["gpu"], a.get("href", "")),
                        })
                # fallback：td-grouping（旧页面结构）
                if not results:
                    all_tds = [
                        td for td in table.find_all("td", recursive=False)
                        if not td.find_parent("thead")
                    ]
                    for td_group in [all_tds[i:i+6] for i in range(0, len(all_tds), 6)]:
                        try:
                            name_div = td_group[0].find("div", class_="item-name")
                            if not name_div:
                                continue
                            a = name_div.find("a", recursive=False)
                            if a and a.get_text(strip=True):
                                results.append({
                                    "name": a.get_text(strip=True),
                                    "detail_url": urljoin(self.TPU_BASE["gpu"], a.get("href", "")),
                                })
                        except Exception:
                            continue
        except Exception:
            logger.exception("[HW] 解析搜索结果异常")
        logger.info(f"[HW][{hw_type.upper()}] 搜索到 {len(results)} 条结果")
        return results

    def _parse_detail_info(self, hw_type: str, html: str) -> Tuple[List[str], str]:
        """
        解析详情页参数。
        返回 (param_lines, hardware_name)。
        """
        if not html:
            return ["详情页获取失败"], "获取失败"
        soup = BeautifulSoup(html, "lxml")

        # 提取硬件名称
        hw_name = "未知硬件"
        for sel in ("h1.pagetitle", "h1.page-title", "div#content h1", "h1"):
            tag = soup.select_one(sel)
            if tag:
                candidate = tag.get_text(strip=True)
                if len(candidate) >= 3 and candidate.lower() not in {"specifications", "details"}:
                    hw_name = candidate
                    break

        param_lines: List[str] = [f"【{hw_name}】", "─" * 20]

        sections = soup.find_all("section", class_="details")
        if not sections:
            param_lines.append("未找到参数区域")
            return param_lines, hw_name

        for section in sections:
            title_tag = section.find(["h1", "h2"])
            if not title_tag:
                continue
            section_en = title_tag.get_text(strip=True)
            if section_en in self.SKIP_SECTIONS:
                continue
            section_cn = self._translate_section(section_en)
            param_lines.append(f"【{section_cn}】")

            rows = self._extract_section_rows(hw_type, section)
            for k, v in rows:
                cn_key = self._translate_param(k)
                param_lines.append(f"{cn_key}：{v}")

        if len(param_lines) <= 2:
            param_lines.append("未提取到详细参数")

        return param_lines, hw_name

    def _extract_section_rows(self, hw_type: str, section) -> List[Tuple[str, str]]:
        """从 section DOM 提取 (英文键, 值) 列表，支持多种页面结构。"""
        rows: List[Tuple[str, str]] = []

        # 优先：dl.clearfix（GPU 常见结构）
        for dl in section.find_all("dl", class_="clearfix"):
            dt = dl.find("dt")
            dd = dl.find("dd")
            if dt and dd:
                k = dt.get_text(strip=True)
                v = dd.get_text(separator=" / ", strip=True).replace("\n", " ")
                if k and v:
                    rows.append((k, v))
        if rows:
            return rows

        # 次选：table > tr > th + td
        table = section.find("table")
        if table:
            for tr in table.find_all("tr"):
                th = tr.find("th")
                td = tr.find("td")
                if th and td:
                    k = th.get_text(strip=True).rstrip(":")
                    v = td.get_text(separator=" ", strip=True).replace("\n", " ")
                    if k and v:
                        rows.append((k, v))
        return rows

    def _get_chinese_font(self, size: int) -> ImageFont.FreeTypeFont:
        """优先加载中文字体，降级到 PIL 默认。"""
        for path in [self.mandatory_font] + self.system_fonts:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        logger.error("[HW] 无可用中文字体，使用 PIL 默认字体（中文可能乱码）")
        return ImageFont.load_default()

    def _generate_param_image(self, hw_type: str, hw_name: str, param_lines: List[str]) -> str:
        """
        用 Pillow 绘制参数图片并保存为 PNG。
        返回图片路径或空字符串（失败时）。
        """
        img_path = self._param_image_path(hw_type, hw_name)

        # 磁盘命中：文件存在且未过期
        if img_path.exists():
            age = time.time() - img_path.stat().st_mtime
            if age < self.image_cache_ttl:
                logger.debug(f"[HW] 复用参数图片缓存：{img_path.name}")
                return str(img_path)

        # ── 绘制 ──
        IMG_W = 860
        PADDING = 44
        LINE_H = 32
        TITLE_H = 44
        DIVIDER_GAP = 16

        title_font = self._get_chinese_font(24)
        section_font = self._get_chinese_font(17)
        param_font = self._get_chinese_font(14)

        # 计算所需高度
        total_h = PADDING + TITLE_H + DIVIDER_GAP
        for line in param_lines[2:]:  # 跳过名称行和分隔行
            total_h += LINE_H + (4 if line.startswith("【") else 0)
        total_h += PADDING

        img = Image.new("RGB", (IMG_W, max(total_h, 200)), color="#f8f9fa")
        draw = ImageDraw.Draw(img)

        # 顶部色条
        draw.rectangle([(0, 0), (IMG_W, 6)], fill="#4a90d9")

        # 标题
        title_text = f"{hw_type.upper()} · {hw_name}"
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        tx = (IMG_W - (bbox[2] - bbox[0])) // 2
        draw.text((tx, PADDING), title_text, font=title_font, fill="#1a1a2e")

        # 分隔线
        y = PADDING + TITLE_H
        draw.line([(PADDING, y), (IMG_W - PADDING, y)], fill="#d0d7de", width=1)
        y += DIVIDER_GAP

        # 参数行
        for line in param_lines[2:]:
            if line.startswith("【") and line.endswith("】"):
                # 分区标题
                y += 4
                draw.rectangle([(PADDING, y), (IMG_W - PADDING, y + LINE_H)], fill="#eef2ff")
                draw.text((PADDING + 12, y + 8), line, font=section_font, fill="#2c3e70")
                y += LINE_H + 8
            elif "：" in line:
                k, v = line.split("：", 1)
                # 交替行背景
                row_color = "#ffffff" if (y // LINE_H) % 2 == 0 else "#f6f8fa"
                draw.rectangle([(PADDING, y), (IMG_W - PADDING, y + LINE_H)], fill=row_color)
                draw.text((PADDING + 12, y + 9), k, font=param_font, fill="#24292f")
                v_bbox = draw.textbbox((0, 0), v, font=param_font)
                v_w = v_bbox[2] - v_bbox[0]
                vx = IMG_W - PADDING - 12 - v_w
                draw.text((vx, y + 9), v, font=param_font, fill="#57606a")
                y += LINE_H
            else:
                draw.text((PADDING + 12, y + 9), line, font=param_font, fill="#8b949e")
                y += LINE_H

        # 底部水印
        watermark = "Data from TechPowerUp"
        wm_font = self._get_chinese_font(11)
        wm_bbox = draw.textbbox((0, 0), watermark, font=wm_font)
        draw.text(
            (IMG_W - PADDING - (wm_bbox[2] - wm_bbox[0]), max(total_h - PADDING + 4, y + 4)),
            watermark, font=wm_font, fill="#c0c8d0",
        )

        try:
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(img_path), format="PNG", optimize=True)
            if sys.platform.startswith("linux"):
                try:
                    os.chmod(str(img_path), 0o644)
                except Exception:
                    pass
            logger.info(f"[HW] 参数图片生成：{img_path.name}")
            return str(img_path)
        except Exception:
            logger.exception("[HW] 保存参数图片失败")
            return ""

    # ══════════════════════════════════════════
    #  详情获取（网络 + 解析 + 图片三合一）
    # ══════════════════════════════════════════

    async def _get_hardware_detail(
        self, hw_type: str, detail_url: str
    ) -> Tuple[List[str], str, bool]:
        """
        拉取并解析硬件详情页，生成参数图片。
        返回 (param_lines, img_path, is_verify)。
        """
        html, is_verify = await self._fetch_html(detail_url, self.TPU_BASE[hw_type])
        if is_verify:
            return ["触发机器人验证，无法获取详情"], "", True
        if not html:
            return ["详情页请求失败"], "", False

        try:
            param_lines, hw_name = await asyncio.to_thread(
                self._parse_detail_info, hw_type, html
            )
        except Exception:
            logger.exception("[HW] 详情解析线程异常")
            return ["解析失败"], "", False

        # 检查磁盘缓存命中（_generate_param_image 内部已处理）
        try:
            img_path = await asyncio.to_thread(
                self._generate_param_image, hw_type, hw_name, param_lines
            )
        except Exception:
            logger.exception("[HW] 图片生成线程异常")
            img_path = ""

        return param_lines, img_path, False

    # ══════════════════════════════════════════
    #  主处理逻辑
    # ══════════════════════════════════════════

    async def _handle_hardware_query(self, event: AstrMessageEvent, hw_type: str):
        """
        统一处理 cpu/gpu 查询指令。
        hw_type 已规范化为小写。
        """
        identity = self._get_identity(event)
        raw = getattr(event, "message_str", None) or getattr(event, "message", "") or ""
        clean = self._clean_text(str(raw))
        logger.info(f"[HW][{identity[0]}@{identity[1]}] 原始指令：{clean}")

        # 正则剥离指令词（含可选前缀 /／，不区分大小写），剩余为参数
        # 例："/GpU RTX 4090" -> param="RTX 4090"；"/CPU" -> param=None
        _m = re.match(
            r"^[/\uff0f]?" + re.escape(hw_type) + r"(?:\s+(.+))?$",
            clean, re.IGNORECASE
        )
        param = _m.group(1).strip() if (_m and _m.group(1)) else None

        # ── 无参数：直接发天梯图 ──
        if param is None:
            img_path = await self._get_ranking_image(hw_type)
            if img_path:
                yield event.image_result(img_path)
            else:
                yield event.plain_result(
                    f"[{hw_type.upper()}] 天梯图获取失败，请稍后重试\n"
                    f"直接访问：https://www.techpowerup.com/{hw_type}-specs/"
                )
            return

        # ── 读取当前缓存 ──
        async with self._cache_lock:
            user_cache = self.search_cache.get(identity, {}).get(hw_type, {})
            cache_valid = bool(user_cache) and time.time() < user_cache.get("expire", 0)

        # ── 数字序号：查详情 ──
        if param.isdigit() and cache_valid:
            idx = int(param) - 1
            results = user_cache.get("results", [])
            if 0 <= idx < len(results):
                selected = results[idx]
                yield event.plain_result(
                    f"[{hw_type.upper()}] 正在获取「{selected['name']}」参数..."
                )
                param_lines, img_path, is_verify = await self._get_hardware_detail(
                    hw_type, selected["detail_url"]
                )
                if is_verify:
                    yield event.plain_result(self._verify_hint(hw_type, selected["detail_url"]))
                    return
                if img_path and Path(img_path).exists():
                    yield event.image_result(img_path)
                else:
                    yield event.plain_result(
                        f"[{hw_type.upper()}] 参数图片生成失败，文字版：\n"
                        + "\n".join(param_lines[:40])
                    )
                return
            # 序号越界时提示
            yield event.plain_result(
                f"[{hw_type.upper()}] 序号 {param} 超出范围（共 {len(results)} 条），"
                f"请重新搜索"
            )
            return

        # ── 关键词搜索：冷却检查 ──
        on_cd, remaining = self._is_on_cooldown(identity, hw_type)
        if on_cd:
            yield event.plain_result(f"[{hw_type.upper()}] 冷却中，请 {remaining} 秒后再试")
            return

        yield event.plain_result(
            f"[{hw_type.upper()}] 搜索中：{param}（结果缓存 {self.cache_ttl}s）"
        )

        search_url = f"{self.TPU_BASE[hw_type]}?q={quote(param)}"
        html, is_verify = await self._fetch_html(search_url, self.TPU_BASE[hw_type])

        # 无论成功与否，先记录冷却时间戳
        async with self._cache_lock:
            self.last_called_times.setdefault(identity, {})[hw_type] = time.time()

        if is_verify:
            yield event.plain_result(self._verify_hint(hw_type, search_url))
            return

        try:
            results = await asyncio.to_thread(
                self._parse_search_results, hw_type, html
            ) if html else []
        except Exception:
            logger.exception("[HW] 搜索结果解析线程异常")
            results = []

        if not results:
            yield event.plain_result(f"[{hw_type.upper()}] 未找到「{param}」相关型号")
            return

        # 写缓存
        async with self._cache_lock:
            self.search_cache.setdefault(identity, {})[hw_type] = {
                "results": results,
                "expire": time.time() + self.cache_ttl,
            }

        lines = [f"[{hw_type.upper()}] 搜索「{param}」共 {len(results)} 条：\n"]
        lines += [f"{i}. {item['name']}" for i, item in enumerate(results, 1)]
        lines.append(f"\n回复「{hw_type} 序号」查看详细参数图片")
        yield event.plain_result("\n".join(lines))

    # ── 验证提示统一生成 ──
    def _verify_hint(self, hw_type: str, url: str) -> str:
        return (
            f"[{hw_type.upper()}] 触发 TechPowerUp 机器人验证！\n"
            "解决方法：\n"
            f"1. 浏览器访问：{url}\n"
            "2. 完成验证码（滑块/图片）\n"
            "3. 将 Cookie 注入环境变量 ASTR_PLUGIN_HW_COOKIES（JSON 格式）\n"
            "4. 重启插件后重试"
        )

    # ══════════════════════════════════════════
    #  命令绑定（正则大小写不敏感匹配）
    #  匹配形如：/cpu  /CPU  /GpU  /cpu rtx4090
    #  前缀 / 可选，指令词不区分大小写
    # ══════════════════════════════════════════

    @filter.regex(r"^[/／]?cpu(\s+.*)?$", re.IGNORECASE)
    async def cpu_info(self, event: AstrMessageEvent):
        """查询 CPU 天梯图 / 搜索型号 / 查看参数"""
        async for r in self._handle_hardware_query(event, "cpu"):
            yield r

    @filter.regex(r"^[/／]?gpu(\s+.*)?$", re.IGNORECASE)
    async def gpu_info(self, event: AstrMessageEvent):
        """查询 GPU 天梯图 / 搜索型号 / 查看参数"""
        async for r in self._handle_hardware_query(event, "gpu"):
            yield r