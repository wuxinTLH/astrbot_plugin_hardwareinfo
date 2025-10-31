import os
import time
import re
import asyncio
import sys
from urllib.parse import urljoin, quote
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.core.utils.io import download_image_by_url

import aiohttp
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional


@register(
    "astrbot_plugin_hardwareinfo",
    "SakuraMikku",
    "硬件信息查询（CPU/GPU搜索+天梯图+参数图片）",
    "0.0.1",
    "https://github.com/wuxinTLH/astrbot_plugin_hardwareinfo"
)
class HardwareInfoPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        # 基础配置
        self.cooldown_period = config.get("cooldown_period", 30)  # 搜索冷却时间（秒）
        self.cache_ttl = config.get("cache_ttl", 60)  # 缓存有效期（秒）
        self.temp_id_expire = 600
        self.base_dir = os.path.dirname(__file__)
        
        # 目录初始化
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.param_image_dir = os.path.join(self.base_dir, "param_images")
        self.font_dir = os.path.join(self.base_dir, "fonts")
        for dir_path in [self.cache_dir, self.param_image_dir, self.font_dir]:
            os.makedirs(dir_path, exist_ok=True)
            if sys.platform.startswith('linux'):
                os.chmod(dir_path, 0o755)
        
        # 字体配置
        self.mandatory_font = os.path.join(self.font_dir, "simhei.ttf")
        self.system_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "simhei.ttf", "wqy-microhei.ttc", "noto-sans-cjk-sc.ttc"
        ]
        
        # 硬件配置
        self.hardware_ranking = {
            "cpu": {
                "url": "https://pica.zhimg.com/v2-18344d446f16199d4208dd9149528834_1440w.webp?consumer=ZHI_MENG",
                "local_path": os.path.join(self.cache_dir, "cpu_ranking.webp")
            },
            "gpu": {
                "url": "https://pic1.zhimg.com/v2-ca6724487c3bcd007598b20eb8693bc4_r.jpg",
                "local_path": os.path.join(self.cache_dir, "gpu_ranking.jpg")
            }
        }
        
        self.tpu_base = {
            "cpu": "https://www.techpowerup.com/cpu-specs/",
            "gpu": "https://www.techpowerup.com/gpu-specs/"
        }
        
        # 参数映射
        self.param_cn_map: Dict[str, str] = {
            "Base Clock": "基础频率", "Boost Clock": "加速频率",
            "Memory Clock": "显存频率", "Memory Size": "显存容量",
            "Memory Type": "显存类型", "Memory Bus": "显存位宽",
            "Bandwidth": "显存带宽", "TDP": "功耗(TDP)",
            "Socket": "接口类型", "Foundry": "生产工艺",
            "Process Size": "制程工艺", "Transistors": "晶体管数量",
            "Die Size": "核心面积", "Market": "适用市场",
            "Release Date": "发布日期", "Launch Price": "首发价格",
            "# of Cores": "核心数", "# of Threads": "线程数",
            "Cache L1": "L1缓存", "Cache L2": "L2缓存",
            "Cache L3": "L3缓存", "Frequency": "基础频率",
            "Turbo Clock": "加速频率", "Multiplier": "倍频",
            "Memory Support": "内存支持", "Rated Speed": "内存频率",
            "Memory Bandwidth": "内存带宽", "Shading Units": "流处理器",
            "TMUs": "纹理单元", "ROPs": "光栅单元",
            "SM Count": "SM单元数", "Tensor Cores": "张量核心",
            "RT Cores": "光线追踪核心", "Slot Width": "插槽宽度",
            "Length": "显卡长度", "Height": "显卡高度",
            "Width": "显卡厚度", "Suggested PSU": "建议电源",
            "Outputs": "视频输出", "Power Connectors": "供电接口",
            "DirectX": "DirectX版本", "OpenGL": "OpenGL版本",
            "OpenCL": "OpenCL版本", "Vulkan": "Vulkan版本",
            "CUDA": "CUDA版本", "Shader Model": "着色器模型"
        }
        
        # 数据存储
        self.search_cache = {}
        self.last_called_times = {}

    def _get_user_id(self, event: AstrMessageEvent) -> str:
        user_id = getattr(event, "user_id", None) or \
                  getattr(getattr(event, "sender", None), "user_id", None) or \
                  getattr(getattr(event, "from_user", None), "id", None)
        
        if user_id:
            return str(user_id)
        
        timestamp_trunc = int(time.time()) // self.temp_id_expire * self.temp_id_expire
        temp_id = f"temp_qqwebhook_{timestamp_trunc}"
        logger.warning(f"未获取到用户ID，使用临时标识：{temp_id}")
        return temp_id

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'\[At:[^\]]+\]', "", text)
        text = re.sub(r'<at[^>]+>.*?</at>', "", text)
        return text.strip()

    def _is_on_cooldown(self, user_id: str, hardware_type: str) -> tuple[bool, float]:
        current_time = time.time()
        user_cooldowns = self.last_called_times.get(user_id, {})
        last_call_time = user_cooldowns.get(hardware_type, 0)
        remaining_time = self.cooldown_period - (current_time - last_call_time)
        
        return (True, round(remaining_time)) if remaining_time > 0 else (False, 0)

    async def _get_ranking_image(self, hardware_type: str) -> str:
        ranking = self.hardware_ranking[hardware_type]
        local_path = ranking["local_path"]
        
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            logger.info(f"[{hardware_type}] 使用本地天梯图")
            return local_path
        
        try:
            logger.info(f"[{hardware_type}] 下载天梯图：{ranking['url']}")
            resp = requests.get(ranking["url"], timeout=15)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                if sys.platform.startswith('linux'):
                    os.chmod(local_path, 0o644)
                logger.info(f"[{hardware_type}] 天梯图保存成功")
                return local_path
            else:
                logger.error(f"[{hardware_type}] 下载失败：状态码{resp.status_code}")
                return ""
        except Exception as e:
            logger.error(f"[{hardware_type}] 天梯图处理异常：{str(e)}")
            return ""

    def _clean_html(self, html: str) -> str:
        if not html:
            return ""
        html = re.sub(r'(<\w+)(class|id|href|src)(=)', r'\1 \2\3', html)
        html = re.sub(r'(class|id|href|src)=([^\s>"]+)', r'\1="\2"', html)
        html = re.sub(r'class="[\w\d]{15,}"', '', html)
        return html

    async def _fetch_tpu_html(self, url: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        cleaned_html = self._clean_html(html)
                        return cleaned_html
                    logger.error(f"请求失败：URL={url}，状态码{resp.status}")
                    return ""
        except Exception as e:
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    cleaned_html = self._clean_html(resp.text())
                    return cleaned_html
                logger.error(f"同步请求失败：URL={url}，状态码{resp.status_code}")
                return ""
            except Exception as e2:
                logger.error(f"请求异常：{str(e2)}")
                return ""

    def _parse_tpu_results(self, hardware_type: str, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        results = []
        
        if hardware_type == "cpu":
            cpu_table = soup.select_one('table.items-desktop-table')
            if not cpu_table:
                logger.warning("[CPU] 未找到结果表格")
                return results
            
            all_trs = cpu_table.find_all("tr", recursive=False)
            data_trs = [tr for tr in all_trs if not (tr.find("th", recursive=False))]
            
            for tr in data_trs:
                first_td = tr.find("td", recursive=False)
                if not first_td:
                    continue
                
                name_tag = first_td.find("a", class_=lambda x: x != "item-image-link")
                if not name_tag:
                    continue
                
                results.append({
                    "name": name_tag.get_text(strip=True),
                    "detail_url": urljoin(self.tpu_base["cpu"], name_tag.get("href", ""))
                })
        
        else:
            gpu_table = soup.select_one('div#list table.items-desktop-table')
            if not gpu_table:
                logger.warning("[GPU] 未找到结果表格")
                return results
            
            all_tds = gpu_table.find_all("td", recursive=False)
            if len(all_tds) < 6:
                logger.warning(f"[GPU] 有效TD数量不足：{len(all_tds)}个")
                return results
            
            gpu_td_groups = [all_tds[i:i+6] for i in range(0, len(all_tds), 6)]
            
            for td_group in gpu_td_groups:
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
                
                if gpu_name and any(keyword in gpu_name.lower() for keyword in ["geforce", "radeon", "rtx", "rx"]):
                    results.append({
                        "name": gpu_name,
                        "detail_url": full_url
                    })
        
        logger.info(f"[{hardware_type.upper()}] 解析完成：{len(results)}条结果")
        return results

    def _translate_param_name(self, param_name: str) -> str:
        cleaned_name = param_name.strip().rstrip(':')
        return self.param_cn_map.get(cleaned_name, cleaned_name)

    def _get_chinese_font(self, size: int) -> ImageFont.FreeTypeFont:
        if os.path.exists(self.mandatory_font):
            try:
                font = ImageFont.truetype(self.mandatory_font, size, encoding="utf-8")
                logger.debug(f"使用强制字体：{self.mandatory_font}")
                return font
            except Exception as e:
                logger.warning(f"强制字体加载失败：{str(e)}，尝试系统字体")
        
        for font_path in self.system_fonts:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size, encoding="utf-8")
                    logger.debug(f"使用系统字体：{font_path}")
                    return font
                except Exception as e:
                    logger.warning(f"字体{font_path}加载失败：{str(e)}")
            else:
                try:
                    font = ImageFont.truetype(font_path, size, encoding="utf-8")
                    logger.debug(f"使用字体名加载：{font_path}")
                    return font
                except Exception as e:
                    continue
        
        error_msg = (
            "无法加载中文字体！请按以下步骤解决：\n"
            "1. 下载黑体字体文件(simhei.ttf)\n"
            f"2. 放置到插件目录下的fonts文件夹：{self.font_dir}\n"
            "3. 确保文件权限正确（Linux下执行：chmod 644 simhei.ttf）"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _generate_param_image(self, hardware_type: str, hardware_name: str, param_data: List[str]) -> str:
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
        except RuntimeError as e:
            error_font = ImageFont.load_default(size=16)
            draw.text((padding, padding), str(e), font=error_font, fill="#ff0000")
            img_path = os.path.join(self.param_image_dir, "font_error.png")
            img.save(img_path, "PNG")
            return img_path

        # 绘制标题（确保硬件名称正确显示）
        title = f"{hardware_type.upper()} 参数详情：{hardware_name}"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_x = (img_width - (title_bbox[2] - title_bbox[0])) // 2
        title_y = padding
        draw.text((title_x, title_y), title, font=title_font, fill="#333333")

        # 绘制分隔线
        line_y = title_y + line_height + 10
        draw.line([(padding, line_y), (img_width - padding, line_y)], fill="#dddddd", width=2)

        # 绘制参数列表
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

        # 保存图片
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", hardware_name)
        img_path = os.path.join(self.param_image_dir, f"{safe_name}_{hardware_type}_param.png")
        img.save(img_path, "PNG", quality=95)
        if sys.platform.startswith('linux'):
            os.chmod(img_path, 0o644)
        logger.info(f"[{hardware_type.upper()}] 参数图片生成成功：{img_path}")
        return img_path

    def _parse_detail_info(self, hardware_type: str, html: str) -> tuple[List[str], str]:
        """修复硬件名称显示为"未知参数"的问题"""
        if not html:
            return ["详情页获取失败"], ""
        
        soup = BeautifulSoup(html, "lxml")
        param_data = []
        
        # 核心修复：增强硬件名称获取逻辑，避免"未知参数"
        hardware_name = "未知硬件"  # 默认值改为"未知硬件"更合理
        
        # 尝试多种方式获取标题，提高成功率
        title_tag = soup.select_one('h1.pagetitle') or \
                   soup.select_one('h1.page-title') or \
                   soup.select_one('div#content h1') or \
                   soup.select_one('h1')
        
        if title_tag:
            hardware_name = title_tag.get_text(strip=True)
            # 过滤可能的无效标题
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
                "Cache": "缓存配置"
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
                            param_name_en = th.get_text(strip=True).rstrip(':')
                            param_name_cn = self._translate_param_name(param_name_en)
                            param_value = td.get_text(strip=True).replace('\n', ' ')
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
                            param_value = dd.get_text(strip=True).replace('\n', ' / ')
                            if param_name_cn and param_value:
                                param_data.append(f"{param_name_cn}：{param_value}")
                else:
                    section_table = section.find("table")
                    if section_table:
                        for row in section_table.find_all("tr"):
                            th = row.find("th")
                            td = row.find("td")
                            if th and td:
                                param_name_en = th.get_text(strip=True).rstrip(':')
                                param_name_cn = self._translate_param_name(param_name_en)
                                param_value = td.get_text(strip=True).replace('\n', ' ')
                                if param_name_cn and param_value:
                                    param_data.append(f"{param_name_cn}：{param_value}")

        if len(param_data) > 2:
            img_path = self._generate_param_image(hardware_type, hardware_name, param_data)
        else:
            param_data.append("未提取到详细参数")
            img_path = ""

        return param_data, img_path

    async def _get_hardware_detail(self, hardware_type: str, detail_url: str) -> tuple[List[str], str]:
        logger.info(f"[{hardware_type.upper()}] 请求详情页：{detail_url}")
        html = await self._fetch_tpu_html(detail_url)
        return self._parse_detail_info(hardware_type, html)

    async def _handle_hardware_query(self, event: AstrMessageEvent, hardware_type: str) -> None:
        """修复冷却时间逻辑：只在搜索型号时应用CD，查看详情无CD"""
        user_id = self._get_user_id(event)
        raw_message = getattr(event, "message_str", "") or ""
        clean_message = self._clean_text(raw_message)
        logger.info(f"[用户{user_id}] 指令：{clean_message}")
        
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
        
        if cmd != hardware_type:
            yield event.plain_result(
                f"[{hardware_type.upper()}] 指令格式错误！正确用法如上"
            )
            return
        
        user_cache = self.search_cache.get(user_id, {}).get(hardware_type, {})
        cache_valid = user_cache and time.time() < user_cache.get("expire", 0)
        
        # 无参数：返回天梯图（无CD）
        if param is None:
            img_path = await self._get_ranking_image(hardware_type)
            if img_path:
                yield event.image_result(img_path)
            else:
                yield event.plain_result(f"[{hardware_type.upper()}] 天梯图获取失败")
            return
        
        # 有参数：区分是搜索还是查看详情
        if param.isdigit() and cache_valid:
            # 查看详情：无CD限制
            idx = int(param) - 1
            results = user_cache["results"]
            if 0 <= idx < len(results):
                selected = results[idx]
                yield event.plain_result(f"[{hardware_type.upper()}] 正在生成「{selected['name']}」参数图片...")
                param_data, img_path = await self._get_hardware_detail(hardware_type, selected["detail_url"])
                if img_path and os.path.exists(img_path):
                    yield event.image_result(img_path)
                else:
                    yield event.plain_result(f"[{hardware_type.upper()}] 参数图片生成失败，参数如下：\n" + "\n".join(param_data))
            else:
                yield event.plain_result(f"[{hardware_type.upper()}] 序号无效！可选范围1-{len(results)}")
        else:
            # 搜索型号：应用CD限制
            on_cooldown, remaining = self._is_on_cooldown(user_id, hardware_type)
            if on_cooldown:
                yield event.plain_result(f"[{hardware_type.upper()}] 冷却中，请{remaining}秒后再试")
                return
                
            yield event.plain_result(f"[{hardware_type.upper()}] 正在搜索：{param}（1分钟有效）")
            search_url = f"{self.tpu_base[hardware_type]}?q={quote(param)}"
            html = await self._fetch_tpu_html(search_url)
            results = self._parse_tpu_results(hardware_type, html) if html else []
            
            if not results:
                yield event.plain_result(f"[{hardware_type.upper()}] 未找到「{param}」相关型号")
                # 搜索无结果也记录冷却时间，防止恶意查询
                self.last_called_times.setdefault(user_id, {})[hardware_type] = time.time()
                return
            
            self.search_cache[user_id] = {
                hardware_type: {"results": results, "expire": time.time() + self.cache_ttl}
            }
            
            list_text = f"[{hardware_type.upper()}] 搜索结果（共{len(results)}条）：\n"
            for i, item in enumerate(results, 1):
                list_text += f"{i}. {item['name']}\n"
            list_text += f"回复「{hardware_type} 序号」查看详情"
            yield event.plain_result(list_text)
            
            # 只有搜索操作才更新冷却时间
            self.last_called_times.setdefault(user_id, {})[hardware_type] = time.time()

    @filter.command("cpu")
    async def cpu_info(self, event: AstrMessageEvent):
        async for result in self._handle_hardware_query(event, "cpu"):
            yield result

    @filter.command("gpu")
    async def gpu_info(self, event: AstrMessageEvent):
        async for result in self._handle_hardware_query(event, "gpu"):
            yield result

    async def initialize(self):
        logger.info("硬件信息查询插件初始化完成（v2.0，修复名称和冷却逻辑）")
        logger.info("依赖库：aiohttp、requests、beautifulsoup4、lxml、pillow>=9.0.0")
        
        if not os.path.exists(self.mandatory_font):
            logger.warning(
                f"未找到强制字体文件：{self.mandatory_font}\n"
                "请下载simhei.ttf并放置到该目录，否则中文可能显示异常"
            )

    async def terminate(self):
        logger.info("硬件信息查询插件已卸载")
