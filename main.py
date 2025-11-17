import os
import re
import sys
import time
import ssl
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple

from urllib.parse import urljoin, quote

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

# Plugin metadata kept as original author/description
@register(
    "astrbot_plugin_hardwareinfo",
    "SakuraMikku",
    "硬件信息查询（CPU/GPU搜索+天梯图+参数图片）",
    "0.0.5",
    "https://github.com/wuxinTLH/astrbot_plugin_hardwareinfo",
)
class HardwareInfoPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)

        # 基础配置（可通过插件配置覆盖）
        self.cooldown_period: int = int(config.get("cooldown_period", 30))
        self.cache_ttl: int = int(config.get("cache_ttl", 60))
        self.temp_id_expire: int = int(config.get("temp_id_expire", 600))

        # 反爬/网络配置
        # 默认为 False（不开启不安全的跳过证书校验）
        self.insecure_skip_verify: bool = bool(config.get("insecure_skip_verify", False))
        # 请求重试次数
        self.max_retries: int = int(config.get("max_retries", 2))

        # 从配置读取自定义 Cookie（非强制）
        self.custom_cookies: Dict[str, str] = config.get("custom_cookies", {}) or {}

        # 允许从环境变量读取 JSON 格式的 cookie（优先于配置文件）
        # 环境变量名：ASTR_PLUGIN_HW_COOKIES，值为 JSON 字符串，例如: {"cookieName": "value"}
        env_cookie_json = os.environ.get("ASTR_PLUGIN_HW_COOKIES")
        if env_cookie_json:
            try:
                env_cookies = json.loads(env_cookie_json)
                if isinstance(env_cookies, dict):
                    # 不在日志中打印 cookie 内容，只记录加载行为
                    self.custom_cookies.update(env_cookies)
                    logger.info("已从环境变量加载自定义 Cookie（未在日志中显示其值）")
                else:
                    logger.warning("环境变量 ASTR_PLUGIN_HW_COOKIES 格式错误，期待 JSON 对象")
            except Exception:
                logger.exception("解析环境变量 ASTR_PLUGIN_HW_COOKIES 时发生异常，忽略该环境变量")

        # 文件/路径
        self.base_dir = Path(__file__).parent.resolve()
        self.cache_dir = self.base_dir / "cache"
        self.param_image_dir = self.base_dir / "param_images"
        self.font_dir = self.base_dir / "fonts"

        for p in (self.cache_dir, self.param_image_dir, self.font_dir):
            p.mkdir(parents=True, exist_ok=True)
            # 尝试设置安全的默认权限（Linux）
            try:
                if sys.platform.startswith("linux"):
                    os.chmod(str(p), 0o755)
            except Exception:
                logger.debug(f"设置目录权限失败：{p}")

        # 字体配置
        self.mandatory_font = str(self.font_dir / "simhei.ttf")
        self.system_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "simhei.ttf",
            "wqy-microhei.ttc",
            "noto-sans-cjk-sc.ttc",
        ]

        # 天梯图与 TechPowerUp 基础 URL
        self.hardware_ranking = {
            "cpu": {
                "url": "https://pica.zhimg.com/v2-18344d446f16199d4208dd9149528834_1440w.webp?consumer=ZHI_MENG",
                "local_path": str(self.cache_dir / "cpu_ranking.webp"),
            },
            "gpu": {
                "url": "https://pic1.zhimg.com/v2-ca6724487c3bcd007598b20eb8693bc4_r.jpg",
                "local_path": str(self.cache_dir / "gpu_ranking.jpg"),
            },
        }

        self.tpu_base = {"cpu": "https://www.techpowerup.com/cpu-specs/", "gpu": "https://www.techpowerup.com/gpu-specs/"}

        # 参数中文映射（保留原始映射）
        self.param_cn_map: Dict[str, str] = {
            "Name": "名称",
            "Codename": "代号",
            "Architecture": "架构",
            "Manufacturer": "制造商",
            "Released": "发布日期",
            "Launch Price": "首发价格",
            "Market Segment": "市场定位",
            "Socket": "接口类型",
            "Core Count": "核心数",
            "Thread Count": "线程数",
            "Base Clock": "基础频率",
            "Boost Clock": "加速频率",
            "All-Core Boost": "全核加速频率",
            "TDP": "功耗(TDP)",
            "TDP Up": "最大功耗",
            "TDP Down": "最小功耗",
            "Process Size": "制程工艺",
            "Transistors": "晶体管数量",
            "Die Size": "核心面积",
            "L1 Cache": "L1缓存",
            "L2 Cache": "L2缓存",
            "L3 Cache": "L3缓存",
            "Memory Support": "内存支持",
            "Memory Channels": "内存通道数",
            "ECC Memory": "ECC内存支持",
            "PCIe Version": "PCIe版本",
            "PCIe Lanes": "PCIe通道数",
            "Instruction Set": "指令集",
            "Virtualization": "虚拟化支持",
            "Thermal Solution": "散热方案",
            "GPU Name": "GPU名称",
            "GPU Variant": "GPU变种",
            "Launch Date": "发布日期",
            "Bus Interface": "总线接口",
            "Memory Size": "显存容量",
            "Memory Type": "显存类型",
            "Memory Bus": "显存位宽",
            "Memory Clock": "显存频率",
            "Memory Bandwidth": "显存带宽",
            "Shading Units": "流处理器",
            "TMUs": "纹理单元",
            "ROPs": "光栅单元",
            "SM Count": "SM单元数",
            "Tensor Cores": "张量核心",
            "RT Cores": "光线追踪核心",
            "Compute Units": "计算单元",
            "Pixel Fillrate": "像素填充率",
            "Texture Fillrate": "纹理填充率",
            "FP32 Performance": "FP32性能",
            "Board Power": "板卡功耗",
            "Suggested PSU": "建议电源",
            "Power Connectors": "供电接口",
            "Slot Width": "插槽宽度",
            "Length": "显卡长度",
            "Height": "显卡高度",
            "Width": "显卡厚度",
            "Outputs": "视频输出",
            "DirectX": "DirectX版本",
            "OpenGL": "OpenGL版本",
            "OpenCL": "OpenCL版本",
            "Vulkan": "Vulkan版本",
            "CUDA": "CUDA版本",
            "Shader Model": "着色器模型",
            "Max Resolution": "最大分辨率",
            "Multi Monitor": "多显示器支持",
            "HDCP": "HDCP支持",
            "Cooler": "散热器类型",
            "LED Lighting": "LED灯效",
            "Manufacturing Process": "制造工艺",
        }

        # 内存结构（按 user+group 隔离），注意并发访问使用 asyncio.Lock
        self.search_cache: Dict[Tuple[str, str], Dict[str, Dict]] = {}
        self.last_called_times: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._cache_lock = asyncio.Lock()

        # HTTP client session （在 initialize 创建并复用）
        self._http_session: aiohttp.ClientSession = None  # type: ignore

        # 请求头模板（Referer 在请求中动态设置）
        self.request_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,imageapng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
        }

    # -------------------- 网络与会话管理 --------------------
    async def initialize(self):
        """在插件初始化时创建复用的 aiohttp session"""
        logger.info("硬件信息查询插件初始化（已应用安全策略）")

        # 管理 SSL 验证策略：默认 True（验证）；若用户显式配置 insecure_skip_verify=True，则警告并跳过验证
        try:
            if self.insecure_skip_verify:
                # 如果用户要求跳过证书校验，使用 connector ssl=False
                connector = TCPConnector(ssl=False)
                logger.warning("已配置 insecure_skip_verify=True，插件将跳过 TLS/SSL 验证（存在安全风险）")
            else:
                connector = TCPConnector(ssl=True)
            timeout = ClientTimeout(total=15)
            self._http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        except Exception:
            logger.exception("创建 HTTP 会话失败，将在每次请求时临时创建 session")

        # 字体提示
        if not Path(self.mandatory_font).exists():
            logger.warning(
                "未找到强制字体文件 simhei.ttf，请将中文字体放到 fonts 目录，或确保系统字体包含中文。"
            )

        # Cookie 提示（不打印实际值）
        if self.custom_cookies:
            logger.info("已加载自定义 Cookie（未在日志中显示其值）；请勿将敏感 cookie 提交到版本控制。")
        else:
            logger.info("未配置自定义 Cookie，若触发验证请参考说明谨慎添加（推荐使用环境变量注入）")

    async def terminate(self):
        """关闭会话等清理工作"""
        try:
            if self._http_session and not self._http_session.closed:
                await self._http_session.close()
        except Exception:
            logger.exception("关闭 HTTP 会话时发生异常")
        logger.info("硬件信息查询插件已卸载")

    # -------------------- 工具函数 --------------------
    def _get_identity(self, event: AstrMessageEvent) -> Tuple[str, str]:
        """获取并标准化用户/群组身份，用于隔离缓存"""
        user_id = getattr(event, "user_id", None) or getattr(getattr(event, "sender", None), "user_id", None) or getattr(getattr(event, "from_user", None), "id", None) or getattr(getattr(event, "author", None), "id", None)
        user_id = str(user_id) if user_id else f"temp_{int(time.time() // self.temp_id_expire)}"

        group_id = getattr(event, "group_id", None)
        if not group_id:
            session = getattr(event, "session", None)
            group_id = getattr(session, "group_id", None) if session else None
        if not group_id:
            group = getattr(event, "group", None)
            group_id = getattr(group, "id", None) if group else None
        if not group_id:
            private_mark = abs(hash(f"{user_id}_{int(time.time() // 3600)}"))  # 保证正数
            group_id = f"private_{private_mark}"
        else:
            group_id = str(group_id)

        logger.debug(f"[身份隔离] 用户ID：{user_id} | 群组ID：{group_id}")
        return user_id, group_id

    def _clean_text(self, text: str) -> str:
        """清理消息文本，去掉 at 标记等，并去除开头的斜杠以兼容 /cpu /gpu 形式"""
        if not isinstance(text, str):
            return ""
        # 去掉常见的 At 标签或 mention 表示
        text = re.sub(r"\[At:[^\]]+\]", "", text)
        text = re.sub(r"<at[^>]*>.*?</at>", "", text, flags=re.I | re.S)
        text = text.strip()
        # 去掉开头的多种斜杠或分隔符（半角/反斜杠/全角斜杠等）
        text = text.lstrip("/\\／﹨")
        # 将连续空白归一
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_on_cooldown(self, identity: Tuple[str, str], hardware_type: str) -> Tuple[bool, int]:
        current_time = time.time()
        user_cooldowns = self.last_called_times.get(identity, {})
        last_call_time = user_cooldowns.get(hardware_type, 0.0)
        remaining_time = self.cooldown_period - (current_time - last_call_time)
        return (True, max(0, int(round(remaining_time)))) if remaining_time > 0 else (False, 0)

    def _clean_html(self, html: str) -> str:
        if not html:
            return ""
        # 一些简单清洗，避免非常长的 class/id 并让解析更稳健
        html = re.sub(r'(<\w+)(class|id|href|src)(=)', r"\1 \2\3", html)
        html = re.sub(r'(class|id|href|src)=([^\s>"]+)', r'\1="\2"', html)
        html = re.sub(r'class="[\w\d]{15,}"', "", html)
        return html

    def _is_verification_page(self, html: str) -> bool:
        verification_keywords = [
            "Automated bot check",
            "Your browser must support Javascript",
            "机器人验证",
            "cloudflare",
            "please enable JavaScript",
        ]
        for kw in verification_keywords:
            if kw.lower() in html.lower():
                return True
        return False

    # -------------------- 网络请求 --------------------
    async def _fetch_tpu_html(self, url: str, hardware_type: str) -> Tuple[str, bool]:
        """
        使用 aiohttp 获取网页内容并返回 (cleaned_html, is_verification_page)
        采用复用 session，如失败会做重试（指数退避）。
        """
        headers = dict(self.request_headers)
        headers["Referer"] = self.tpu_base.get(hardware_type, self.tpu_base["cpu"])

        # 如果 session 未创建，临时创建（不推荐长期使用）
        session = self._http_session
        temp_session = None
        try:
            if session is None or getattr(session, "closed", True):
                connector = TCPConnector(ssl=False) if self.insecure_skip_verify else TCPConnector(ssl=True)
                temp_session = aiohttp.ClientSession(connector=connector, timeout=ClientTimeout(total=15))
                session = temp_session

            for attempt in range(self.max_retries):
                try:
                    # cookies 不要记录具体内容到日志
                    async with session.get(url, headers=headers, cookies=self.custom_cookies or None) as resp:
                        text = await resp.text()
                        cleaned_html = self._clean_html(text)
                        if self._is_verification_page(cleaned_html):
                            logger.info(f"[{hardware_type.upper()}] 识别到验证页面：{url}")
                            return "", True
                        if resp.status == 200:
                            return cleaned_html, False
                        else:
                            logger.warning(f"请求 {url} 返回状态 {resp.status}（尝试 {attempt+1}/{self.max_retries}）")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"异步请求异常（尝试{attempt+1}/{self.max_retries}）：{e}")
                # 指数退避
                await asyncio.sleep(2 ** attempt)

            logger.error(f"请求 URL={url} 在 {self.max_retries} 次尝试后失败")
            return "", False
        finally:
            if temp_session:
                try:
                    await temp_session.close()
                except Exception:
                    logger.debug("关闭临时 session 时发生异常")

    async def _get_ranking_image(self, hardware_type: str) -> str:
        """下载或复用本地天梯图（异步）"""
        ranking = self.hardware_ranking.get(hardware_type)
        if not ranking:
            logger.error(f"未知硬件类型：{hardware_type}")
            return ""

        local_path = Path(ranking["local_path"])
        if local_path.exists() and local_path.stat().st_size > 1024:
            logger.info(f"[{hardware_type}] 使用本地天梯图：{local_path}")
            return str(local_path)

        url = ranking["url"]
        logger.info(f"[{hardware_type}] 下载天梯图：{url}")

        session = self._http_session
        temp_session = None
        try:
            if session is None or getattr(session, "closed", True):
                connector = TCPConnector(ssl=False) if self.insecure_skip_verify else TCPConnector(ssl=True)
                temp_session = aiohttp.ClientSession(connector=connector, timeout=ClientTimeout(total=30))
                session = temp_session

            for attempt in range(self.max_retries):
                try:
                    async with session.get(url, headers=self.request_headers or None) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            if not content or len(content) < 512:
                                logger.warning(f"[{hardware_type}] 下载内容过小，尝试重试（尝试{attempt+1}）")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            # 写文件
                            try:
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(local_path, "wb") as f:
                                    f.write(content)
                                if sys.platform.startswith("linux"):
                                    try:
                                        os.chmod(str(local_path), 0o644)
                                    except Exception:
                                        logger.debug("设置天梯图文件权限失败")
                                logger.info(f"[{hardware_type}] 天梯图保存成功：{local_path}")
                                return str(local_path)
                            except Exception:
                                logger.exception(f"[{hardware_type}] 保存天梯图到磁盘失败")
                                return ""
                        else:
                            logger.warning(f"[{hardware_type}] 下载失败：状态码 {resp.status}（尝试{attempt+1}）")
                except Exception as e:
                    logger.warning(f"[{hardware_type}] 下载异常（尝试{attempt+1}）：{e}")
                await asyncio.sleep(2 ** attempt)

            logger.error(f"[{hardware_type}] 下载天梯图在 {self.max_retries} 次尝试后失败")
            return ""
        finally:
            if temp_session:
                try:
                    await temp_session.close()
                except Exception:
                    logger.debug("关闭临时 session 时发生异常")

    # -------------------- 解析逻辑 --------------------
    def _parse_tpu_results(self, hardware_type: str, html: str) -> List[Dict[str, str]]:
        """解析搜索列表页面，返回 name + detail_url 列表"""
        results: List[Dict[str, str]] = []
        if not html:
            return results
        soup = BeautifulSoup(html, "lxml")

        try:
            if hardware_type == "cpu":
                cpu_table = soup.select_one("table.items-desktop-table")
                if not cpu_table:
                    logger.warning("[CPU] 未找到结果表格（可能是验证页面或结构变化）")
                    return results
                all_trs = cpu_table.find_all("tr", recursive=False)
                data_trs = [tr for tr in all_trs if not tr.find("th", recursive=False)]
                for tr in data_trs:
                    first_td = tr.find("td", recursive=False)
                    if not first_td:
                        continue
                    name_tag = first_td.find("a", class_=lambda x: x != "item-image-link")
                    if not name_tag:
                        continue
                    results.append({"name": name_tag.get_text(strip=True), "detail_url": urljoin(self.tpu_base["cpu"], name_tag.get("href", ""))})
            else:
                gpu_table = soup.select_one("div#list table.items-desktop-table")
                if not gpu_table:
                    logger.warning("[GPU] 未找到结果表格（可能是验证页面或结构变化）")
                    return results
                all_tds = [td for td in gpu_table.find_all("td", recursive=False) if not td.find_parent("thead")]
                if len(all_tds) < 6:
                    logger.warning(f"[GPU] 有效数据单元不足：{len(all_tds)}")
                    return results
                groups = [all_tds[i : i + 6] for i in range(0, len(all_tds), 6)]
                for td_group in groups:
                    try:
                        name_td = td_group[0]
                        item_name_div = name_td.find("div", class_="item-name")
                        if not item_name_div:
                            continue
                        name_tag = item_name_div.find("a", recursive=False)
                        if not name_tag:
                            continue
                        gpu_name = name_tag.get_text(strip=True)
                        relative_href = name_tag.get("href", "")
                        full_url = urljoin(self.tpu_base["gpu"], relative_href)
                        if gpu_name:
                            results.append({"name": gpu_name, "detail_url": full_url})
                    except Exception:
                        logger.debug("解析 GPU 表格单元时遇到异常，跳过该项")
        except Exception:
            logger.exception("解析搜索结果时发生异常")
        logger.info(f"[{hardware_type.upper()}] 搜索完成，找到 {len(results)} 条结果")
        return results

    def _translate_param_name(self, param_name: str) -> str:
        cleaned = param_name.strip().rstrip(":")
        return self.param_cn_map.get(cleaned, cleaned)

    def _get_chinese_font(self, size: int) -> ImageFont.FreeTypeFont:
        """加载中文字体，优先使用插件目录的 simhei.ttf，回退系统字体或默认字体"""
        # 尝试插件内的强制字体
        try:
            if Path(self.mandatory_font).exists():
                return ImageFont.truetype(self.mandatory_font, size)
        except Exception:
            logger.debug("加载强制字体失败，尝试系统字体", exc_info=True)

        # 尝试已知系统路径
        for p in self.system_fonts:
            try:
                if Path(p).exists():
                    return ImageFont.truetype(p, size)
                # 有些平台接受字体名直接加载
                return ImageFont.truetype(p, size)
            except Exception:
                continue

        # 最后回退为默认字体（可能不支持中文）
        logger.error("无法加载到合适的中文字体，将使用默认字体（可能导致中文显示乱码）")
        return ImageFont.load_default()

    def _generate_param_image(self, hardware_type: str, hardware_name: str, param_data: List[str]) -> str:
        """将参数文本渲染到图片并保存，返回图片路径"""
        img_width = 800
        line_height = 30
        padding = 40
        title_font_size = 24
        section_font_size = 18
        param_font_size = 14

        total_lines = len(param_data) + 2
        img_height = padding * 2 + total_lines * line_height
        img = Image.new("RGB", (img_width, img_height), color="#ffffff")
        draw = ImageDraw.Draw(img)

        try:
            title_font = self._get_chinese_font(title_font_size)
            section_font = self._get_chinese_font(section_font_size)
            param_font = self._get_chinese_font(param_font_size)
        except Exception:
            logger.exception("加载字体时发生异常")
            title_font = section_font = param_font = ImageFont.load_default()

        title = f"{hardware_type.upper()} 参数详情：{hardware_name}"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_x = (img_width - (title_bbox[2] - title_bbox[0])) // 2
        title_y = padding
        draw.text((title_x, title_y), title, font=title_font, fill="#333333")

        # 分隔线
        line_y = title_y + line_height + 10
        draw.line([(padding, line_y), (img_width - padding, line_y)], fill="#dddddd", width=2)

        current_y = line_y + 20
        for line in param_data:
            if line.startswith("【") and line.endswith("】"):
                section_bbox = draw.textbbox((0, 0), line, font=section_font)
                section_x = (img_width - (section_bbox[2] - section_bbox[0])) // 2
                draw.text((section_x, current_y), line, font=section_font, fill="#2c3e50")
                current_y += line_height + 5
            else:
                if "：" in line:
                    param_name, param_value = line.split("：", 1)
                    draw.text((padding, current_y), param_name, font=param_font, fill="#34495e")
                    value_bbox = draw.textbbox((0, 0), param_value, font=param_font)
                    value_x = img_width - padding - (value_bbox[2] - value_bbox[0])
                    draw.text((value_x, current_y), param_value, font=param_font, fill="#7f8c8d")
                else:
                    draw.text((padding, current_y), line, font=param_font, fill="#95a5a6")
                current_y += line_height

        # 保存图片（安全化文件名）
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", hardware_name)[:120]
        img_path = self.param_image_dir / f"{safe_name}_{hardware_type}_param.png"
        try:
            img.save(str(img_path), format="PNG", optimize=True)
            if sys.platform.startswith("linux"):
                try:
                    os.chmod(str(img_path), 0o644)
                except Exception:
                    logger.debug("设置参数图片权限失败")
            logger.info(f"[{hardware_type.upper()}] 参数图片生成成功：{img_path}")
            return str(img_path)
        except Exception:
            logger.exception("保存参数图片时发生异常")
            return ""

    def _parse_detail_info(self, hardware_type: str, html: str) -> Tuple[List[str], str]:
        """从详情页解析参数并返回 (param_data, image_path)"""
        if not html:
            return ["详情页获取失败"], ""
        soup = BeautifulSoup(html, "lxml")
        param_data: List[str] = []

        # 获取名称
        hardware_name = "未知硬件"
        title_tag = soup.select_one("h1.pagetitle") or soup.select_one("h1.page-title") or soup.select_one("div#content h1") or soup.select_one("h1")
        if title_tag:
            hardware_name = title_tag.get_text(strip=True)
            if len(hardware_name) < 2 or hardware_name.lower() in ["specifications", "details"]:
                hardware_name = "获取名称失败"

        param_data.append(f"【{hardware_name}】")
        param_data.append("=" * 20)

        detail_sections = soup.find_all("section", class_="details")
        if not detail_sections:
            param_data.append("未找到参数区域")
            return param_data, ""

        for section in detail_sections:
            section_title_tag = section.find("h1") or section.find("h2")
            if not section_title_tag:
                continue
            section_title_en = section_title_tag.get_text(strip=True)
            section_title_cn = {
                "Clock Speeds": "频率参数",
                "Memory": "显存参数",
                "Render Config": "渲染配置",
                "Theoretical Performance": "理论性能",
                "Board Design": "显卡设计",
                "Graphics Features": "图形特性",
                "Physical": "物理参数",
                "Processor": "处理器信息",
                "Performance": "性能参数",
                "Architecture": "架构参数",
                "Core Config": "核心配置",
                "Cache": "缓存配置",
                "Memory Specifications": "内存规格",
                "Expansion": "扩展接口",
                "Power Management": "电源管理",
                "Thermals": "散热参数",
                "General Specifications": "基本规格",
            }.get(section_title_en, section_title_en)

            if section_title_en in ["Notes", "Features", "GB202 GPU Notes"]:
                continue

            param_data.append(f"【{section_title_cn}】")

            if hardware_type == "cpu":
                section_table = section.find("table")
                if section_table:
                    for row in section_table.find_all("tr"):
                        th = row.find("th")
                        td = row.find("td")
                        if th and td:
                            param_name_en = th.get_text(strip=True).rstrip(":")
                            param_name_cn = self._translate_param_name(param_name_en)
                            param_value = td.get_text(strip=True).replace("\n", " ")
                            if param_name_cn and param_value:
                                param_data.append(f"{param_name_cn}：{param_value}")
            else:
                dl_containers = section.find_all("dl", class_="clearfix")
                if dl_containers:
                    for dl in dl_containers:
                        dt = dl.find("dt")
                        dd = dl.find("dd")
                        if dt and dd:
                            param_name_en = dt.get_text(strip=True)
                            param_name_cn = self._translate_param_name(param_name_en)
                            param_value = dd.get_text(strip=True).replace("\n", " / ")
                            if param_name_cn and param_value:
                                param_data.append(f"{param_name_cn}：{param_value}")
                else:
                    section_table = section.find("table")
                    if section_table:
                        for row in section_table.find_all("tr"):
                            th = row.find("th")
                            td = row.find("td")
                            if th and td:
                                param_name_en = th.get_text(strip=True).rstrip(":")
                                param_name_cn = self._translate_param_name(param_name_en)
                                param_value = td.get_text(strip=True).replace("\n", " ")
                                if param_name_cn and param_value:
                                    param_data.append(f"{param_name_cn}：{param_value}")

        if len(param_data) > 2:
            img_path = self._generate_param_image(hardware_type, hardware_name, param_data)
        else:
            param_data.append("未提取到详细参数")
            img_path = ""
        return param_data, img_path

    async def _get_hardware_detail(self, hardware_type: str, detail_url: str) -> Tuple[List[str], str, bool]:
        """获取详情页并解析，返回 (param_list, img_path, is_verification_page)"""
        logger.info(f"[{hardware_type.upper()}] 请求详情页：{detail_url}")
        html, is_verify = await self._fetch_tpu_html(detail_url, hardware_type)
        if is_verify:
            return ["触发机器人验证，无法获取详情"], "", True
        params, img_path = self._parse_detail_info(hardware_type, html)
        return params, img_path, False

    # -------------------- 主处理逻辑 --------------------
    async def _handle_hardware_query(self, event: AstrMessageEvent, hardware_type: str):
        identity = self._get_identity(event)
        # 兼容不同 event 字段名：优先 message_str，其次 message（部分平台）
        raw_message = getattr(event, "message_str", None)
        if not raw_message:
            raw_message = getattr(event, "message", "") or ""
        raw_message = raw_message or ""
        clean_message = self._clean_text(raw_message)
        logger.info(f"[用户{identity[0]}@群组{identity[1]}] 指令：{clean_message}")

        if not clean_message:
            yield event.plain_result(
                f"[{hardware_type.upper()}] 指令格式错误！\n"
                f"正确用法：\n"
                f"  1. 查天梯图：{hardware_type}\n"
                f"  2. 搜型号：{hardware_type} 关键词（例：{hardware_type} RTX 5090）\n"
                f"  3. 看详情：{hardware_type} 序号（例：{hardware_type} 1）"
            )
            return

        parts = clean_message.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        param = parts[1].strip() if len(parts) >= 2 else None

        # 如果用户直接发送 "cpu" 或 "/cpu"（clean_text 已处理斜杠），cmd 会是 "cpu"
        if cmd != hardware_type:
            yield event.plain_result(f"[{hardware_type.upper()}] 指令格式错误！正确用法如上")
            return

        # 获取缓存（并检查有效性）
        async with self._cache_lock:
            user_cache = self.search_cache.get(identity, {}).get(hardware_type, {})
            cache_valid = bool(user_cache) and time.time() < user_cache.get("expire", 0)

        # 无参数：返回天梯图
        if param is None:
            img_path = await self._get_ranking_image(hardware_type)
            if img_path:
                yield event.image_result(img_path)
            else:
                yield event.plain_result(f"[{hardware_type.upper()}] 天梯图获取失败")
            return

        # 检查是否是索引
        is_valid_index = False
        if param.isdigit():
            idx = int(param) - 1
            if cache_valid:
                results = user_cache.get("results", [])
                if 0 <= idx < len(results):
                    is_valid_index = True

        if is_valid_index:
            idx = int(param) - 1
            results = user_cache.get("results", [])
            selected = results[idx]
            yield event.plain_result(f"[{hardware_type.upper()}] 正在生成「{selected['name']}」参数图片...")
            param_data, img_path, is_verify = await self._get_hardware_detail(hardware_type, selected["detail_url"])

            if is_verify:
                # 提示用户如何操作（不要求或记录敏感 cookie）
                yield event.plain_result(
                    f"[{hardware_type.upper()}] 触发 techpowerup 机器人验证！\n"
                    "请按以下步骤解决：\n"
                    f"1. 用浏览器访问：{selected['detail_url']}\n"
                    "2. 完成验证码（滑块/图片验证）\n"
                    "3. 将需要的 Cookie 以安全方式注入（推荐使用环境变量 ASTR_PLUGIN_HW_COOKIES，不要直接写入仓库）\n"
                    "4. 重启插件后重试"
                )
                return

            if img_path and Path(img_path).exists():
                yield event.image_result(img_path)
            else:
                yield event.plain_result(f"[{hardware_type.upper()}] 参数图片生成失败，参数如下：\n" + "\n".join(param_data))
            return

        # 非索引 -> 搜索
        on_cooldown, remaining = self._is_on_cooldown(identity, hardware_type)
        if on_cooldown:
            yield event.plain_result(f"[{hardware_type.upper()}] 冷却中，请 {remaining} 秒后再试")
            return

        yield event.plain_result(f"[{hardware_type.upper()}] 正在搜索：{param}（{self.cache_ttl} 秒有效）")
        search_url = f"{self.tpu_base[hardware_type]}?q={quote(param)}"
        html, is_verify = await self._fetch_tpu_html(search_url, hardware_type)

        if is_verify:
            yield event.plain_result(
                f"[{hardware_type.upper()}] 触发 techpowerup 机器人验证！\n"
                "请按以下步骤解决：\n"
                f"1. 用浏览器访问：{search_url}\n"
                "2. 完成验证码（滑块/图片验证）\n"
                "3. 将需要的 Cookie 以安全方式注入（推荐使用环境变量 ASTR_PLUGIN_HW_COOKIES）\n"
                "4. 重启插件后重试"
            )
            # 更新冷却时间以防止用户短时间内重复触发
            async with self._cache_lock:
                self.last_called_times.setdefault(identity, {})[hardware_type] = time.time()
            return

        results = self._parse_tpu_results(hardware_type, html) if html else []

        if not results:
            yield event.plain_result(f"[{hardware_type.upper()}] 未找到「{param}」相关型号")
            async with self._cache_lock:
                self.last_called_times.setdefault(identity, {})[hardware_type] = time.time()
            return

        # 缓存当前身份的搜索结果（修复覆盖问题）
        async with self._cache_lock:
            self.search_cache.setdefault(identity, {})[hardware_type] = {"results": results, "expire": time.time() + self.cache_ttl}
            self.last_called_times.setdefault(identity, {})[hardware_type] = time.time()

        logger.debug(f"[缓存更新] {identity} | {hardware_type} | 结果数：{len(results)}")

        list_text = f"[{hardware_type.upper()}] 搜索结果（共{len(results)}条）：\n"
        for i, item in enumerate(results, 1):
            list_text += f"{i}. {item['name']}\n"
        list_text += f"回复「{hardware_type} 序号」查看详情"
        yield event.plain_result(list_text)

    # -------------------- 命令绑定 --------------------
    @filter.command("cpu")
    async def cpu_info(self, event: AstrMessageEvent):
        """查询CPU天梯榜图或型号信息"""
        async for r in self._handle_hardware_query(event, "cpu"):
            yield r

    @filter.command("gpu")
    async def gpu_info(self, event: AstrMessageEvent):
        """查询GPU天梯榜图或型号信息"""
        async for r in self._handle_hardware_query(event, "gpu"):
            yield r