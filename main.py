import os
import time
import re
import asyncio
from urllib.parse import urljoin, quote
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.core.utils.io import download_image_by_url

import aiohttp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


@register(
    "astrbot_plugin_hardwareinfo",
    "SakuraMikku",
    "硬件信息查询（支持CPU/GPU搜索、天梯图、详情截图）",
    "1.0",
    "https://github.com/SakuraMikku/astrbot_plugins"
)
class HardwareInfoPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        # 基础配置
        self.cooldown_period = config.get("cooldown_period", 60)  # 指令冷却时间（秒）
        self.cache_ttl = config.get("cache_ttl", 60)  # 搜索结果缓存时间（秒）
        self.temp_id_expire = 600  # 临时用户ID有效期（10分钟，确保缓存共享）
        self.base_dir = os.path.dirname(__file__)
        
        # 目录初始化（缓存+截图）
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.screenshot_dir = os.path.join(self.base_dir, "screenshots")
        for dir_path in [self.cache_dir, self.screenshot_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 天梯图配置（用户提供的链接）
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
        
        # TechPowerUp搜索基址（带斜杠，适配/cpu-specs/?q=格式）
        self.tpu_base = {
            "cpu": "https://www.techpowerup.com/cpu-specs/",
            "gpu": "https://www.techpowerup.com/gpu-specs/"
        }
        
        # 缓存存储（用户ID → 硬件类型 → {结果列表, 过期时间}）
        self.search_cache = {}
        # 冷却时间存储（用户ID → 硬件类型 → 上次调用时间）
        self.last_called_times = {}

    def _get_user_id(self, event: AstrMessageEvent) -> str:
        """获取用户ID，临时ID确保10分钟内不变（解决缓存共享）"""
        # 优先获取QQ官方用户ID字段
        user_id = getattr(event, "user_id", None) or \
        getattr(getattr(event, "sender", None), "user_id", None) or \
        getattr(getattr(event, "from_user", None), "id", None)
        
        if user_id:
            return str(user_id)
        
        # 生成临时ID（时间戳截断，10分钟内同一用户不变）
        timestamp_trunc = int(time.time()) // self.temp_id_expire * self.temp_id_expire
        temp_id = f"temp_qqwebhook_{timestamp_trunc}"
        logger.warning(f"未获取到用户ID，使用临时标识（10分钟有效）：{temp_id}")
        return temp_id

    def _clean_text(self, text: str) -> str:
        """仅移除QQ@标签，保留指令和参数完整性"""
        text = re.sub(r'\[At:[^\]]+\]', "", text)  # 移除[At:xxx]
        text = re.sub(r'<at[^>]+>.*?</at>', "", text)  # 移除<at>标签
        return text.strip()  # 仅去前后空格，不修改其他内容

    def _is_on_cooldown(self, user_id: str, hardware_type: str) -> tuple[bool, float]:
        """检查用户指令是否在冷却中"""
        current_time = time.time()
        user_cooldowns = self.last_called_times.get(user_id, {})
        last_call = user_cooldowns.get(hardware_type, 0)
        remaining = self.cooldown_period - (current_time - last_call)
        
        if remaining > 0:
            return True, round(remaining)
        return False, 0

    async def _get_ranking_image(self, hardware_type: str) -> str:
        """获取天梯图（本地优先，本地无则下载）"""
        ranking = self.hardware_ranking[hardware_type]
        local_path = ranking["local_path"]
        
        # 本地有有效文件（>1KB）直接返回
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            logger.info(f"[{hardware_type}] 使用本地天梯图：{os.path.basename(local_path)}")
            return local_path
        
        # 下载天梯图到本地
        try:
            logger.info(f"[{hardware_type}] 下载天梯图：{ranking['url']}")
            temp_path = await download_image_by_url(ranking["url"])
            if not temp_path or not os.path.exists(temp_path):
                logger.error(f"[{hardware_type}] 下载失败：未获取临时文件")
                return ""
            
            # 保存到缓存目录
            with open(local_path, "wb") as f_out, open(temp_path, "rb") as f_in:
                f_out.write(f_in.read())
            logger.info(f"[{hardware_type}] 天梯图保存成功：{local_path}")
            return local_path
        except Exception as e:
            logger.error(f"[{hardware_type}] 天梯图处理异常：{str(e)}")
            return ""

    async def _fetch_tpu_html(self, hardware_type: str, keyword: str) -> str:
        """请求TechPowerUp搜索页面（正确拼接URL）"""
        encoded_keyword = quote(keyword)
        # 最终URL格式：https://www.techpowerup.com/cpu-specs/?q=xxx
        search_url = f"{self.tpu_base[hardware_type]}?q={encoded_keyword}"
        logger.info(f"[{hardware_type}] 搜索URL：{search_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, timeout=15) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        logger.info(f"[{hardware_type}] 成功获取页面（长度：{len(html)//1024}KB）")
                        return html
                    logger.error(f"[{hardware_type}] 搜索失败：状态码{resp.status}")
                    return ""
        except Exception as e:
            logger.error(f"[{hardware_type}] 搜索网络异常：{str(e)}")
            return ""

    def _parse_tpu_results(self, hardware_type: str, html: str) -> list[dict]:
        """解析搜索结果（CPU/GPU分别适配页面结构）"""
        soup = BeautifulSoup(html, "lxml")
        results = []
        
        if hardware_type == "cpu":
            # CPU页面：表格无<tbody>，直接解析table下的tr
            cpu_table = soup.select_one('table.items-desktop-table')
            if not cpu_table:
                logger.warning("[CPU] 未找到结果表格（类名：items-desktop-table）")
                return results
            
            # 过滤表头行（子元素为<th>的是表头）
            all_trs = cpu_table.find_all("tr", recursive=False)
            data_trs = [tr for tr in all_trs if not (tr.find(recursive=False) and tr.find(recursive=False).name == "th")]
            
            if not data_trs:
                logger.warning(f"[CPU] 未找到数据行（共{len(all_trs)}行，均为表头）")
                return results
            
            # 提取CPU名称和详情链接
            for tr in data_trs:
                first_td = tr.find("td", recursive=False)
                if not first_td:
                    continue
                
                name_tag = first_td.find("a", recursive=False)
                if not name_tag:
                    continue
                
                results.append({
                    "name": name_tag.get_text(strip=True),
                    "detail_url": urljoin(self.tpu_base["cpu"], name_tag.get("href", ""))
                })
        
        else:  # GPU页面解析
            gpu_table = soup.select_one('table.items-desktop-table')
            if not gpu_table:
                logger.warning("[GPU] 未找到结果表格（类名：items-desktop-table）")
                logger.warning(f"[GPU] 页面片段：{str(soup.select_one('body'))[:1000]}...")
                return results
            
            # 过滤数据行（含item-name的是数据行）
            all_trs = gpu_table.find_all("tr", recursive=False)
            data_trs = [tr for tr in all_trs if tr.find("div", class_="item-name", recursive=True)]
            
            if not data_trs:
                logger.warning(f"[GPU] 未找到数据行（共{len(all_trs)}行）")
                logger.warning(f"[GPU] 表格片段：{str(gpu_table)[:2000]}...")
                return results
            
            # 提取GPU名称和详情链接
            for tr in data_trs:
                name_div = tr.find("div", class_="item-name", recursive=True)
                if not name_div:
                    continue
                
                name_tag = name_div.find("a", recursive=False)
                if not name_tag:
                    logger.warning(f"[GPU] 数据行无链接：{str(tr)[:500]}...")
                    continue
                
                results.append({
                    "name": name_tag.get_text(strip=True),
                    "detail_url": urljoin(self.tpu_base["gpu"], name_tag.get("href", ""))
                })
                logger.debug(f"[GPU] 提取结果：{name_tag.get_text(strip=True)}")
        
        logger.info(f"[{hardware_type.upper()}] 解析完成，共获取{len(results)}条有效结果")
        return results

    def _screenshot_tpu_detail(self, hardware_type: str, detail_url: str, name: str) -> str:
        """截图硬件详情页（容错配置，适配不同环境）"""
        # 处理非法文件名
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", name)
        screenshot_path = os.path.join(self.screenshot_dir, f"{safe_name}_{hardware_type}.png")
        
        # 本地有截图直接返回
        if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 2048:
            logger.info(f"[{hardware_type}] 使用本地截图：{os.path.basename(screenshot_path)}")
            return screenshot_path
        
        # Chrome浏览器配置（解决截图失败核心）
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # 无头模式
        chrome_options.add_argument("--disable-gpu")  # 禁用GPU（兼容Linux）
        chrome_options.add_argument("--window-size=1920,1080")  # 增大窗口，避免元素遮挡
        chrome_options.add_argument("--no-sandbox")  # 关闭沙箱（Linux必加）
        chrome_options.add_argument("--disable-dev-shm-usage")  # 解决共享内存不足
        chrome_options.add_argument("--start-maximized")  # 最大化窗口
        chrome_options.add_argument("--ignore-certificate-errors")  # 忽略证书错误
        # 模拟正常浏览器UA，避免被识别为爬虫
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        try:
            logger.info(f"[{hardware_type}] 启动浏览器，访问详情页：{detail_url}")
            # --------------------------
            # 关键配置：指定ChromeDriver路径（需根据服务器实际路径修改！）
            # Linux示例：executable_path="/usr/bin/chromedriver"
            # Windows示例：executable_path="C:/Program Files/ChromeDriver/chromedriver.exe"
            # --------------------------
            driver = webdriver.Chrome(
                executable_path="/usr/bin/chromedriver",  # ← 替换为你的ChromeDriver路径
                options=chrome_options
            )
            driver.get(detail_url)
            wait = WebDriverWait(driver, 15)  # 缩短等待时间，避免msg_id过期
            
            # 等待页面加载完成（通过标题判断）
            wait.until(lambda d: d.title != "")
            logger.info(f"[{hardware_type}] 页面标题：{driver.title}")
            
            # 定位截图区域（优先sectioncontainer，无则用body）
            try:
                target_element = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "sectioncontainer"))
                )
                logger.info(f"[{hardware_type}] 找到sectioncontainer元素")
            except TimeoutException:
                logger.warning(f"[{hardware_type}] 未找到sectioncontainer，使用body截图")
                target_element = driver.find_element(By.TAG_NAME, "body")
            
            # 滚动到元素可见位置
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'start'});", 
                target_element
            )
            time.sleep(0.8)  # 等待滚动稳定
            
            # 截图并保存
            screenshot_data = target_element.screenshot_as_png
            with open(screenshot_path, "wb") as f:
                f.write(screenshot_data)
            
            # 验证截图有效性
            if os.path.getsize(screenshot_path) > 2048:
                logger.info(f"[{hardware_type}] 截图成功：{screenshot_path}（{len(screenshot_data)//1024}KB）")
                return screenshot_path
            else:
                logger.error(f"[{hardware_type}] 截图无效（大小<2KB），已删除")
                os.remove(screenshot_path)
                return ""
        
        except Exception as e:
            logger.error(f"[{hardware_type}] 截图异常：{type(e).__name__}:{str(e)}")
            # 打印页面源码，便于排查问题
            if "driver" in locals():
                logger.error(f"[{hardware_type}] 页面源码片段：{driver.page_source[:2000]}...")
            return ""
        finally:
            if "driver" in locals():
                driver.quit()
                logger.info(f"[{hardware_type}] 关闭浏览器")

    async def _handle_hardware_query(self, event: AstrMessageEvent, hardware_type: str) -> None:
        """核心处理逻辑（整合搜索、序号识别、缓存管理）"""
        user_id = self._get_user_id(event)
        raw_message = getattr(event, "message_str", "")
        clean_message = self._clean_text(raw_message)
        logger.info(f"[用户{user_id}] 清理后消息：{clean_message}")
        
        # 分离指令和参数（maxsplit=1：保留参数中的空格）
        parts = clean_message.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        param = parts[1].strip() if len(parts) >= 2 else None
        
        # 1. 校验指令格式
        if cmd != hardware_type:
            yield event.plain_result(
                f"[{hardware_type.upper()}] 指令格式错误！\n"
                f"正确用法：\n"
                f"  1. 查天梯图：{hardware_type}（直接发送）\n"
                f"  2. 搜型号：{hardware_type} 关键词（如：{hardware_type} 9800x3d）\n"
                f"  3. 看详情：{hardware_type} 序号（如：{hardware_type} 1，需先搜索）"
            )
            return
        
        # 2. 冷却检查
        on_cooldown, remaining = self._is_on_cooldown(user_id, hardware_type)
        if on_cooldown:
            yield event.plain_result(f"[{hardware_type.upper()}] 冷却中，请{remaining}秒后再试")
            return
        
        # 3. 读取用户缓存
        user_cache = self.search_cache.get(user_id, {}).get(hardware_type, {