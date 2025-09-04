# encoding:utf-8
import os
import re
import json
import time
import base64
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import tempfile
from io import BytesIO
from datetime import datetime, timedelta
import threading
from PIL import Image
from typing import Optional

import plugins
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from plugins import *
from config import conf

@plugins.register(
    name="NanoBananaImage",
    desire_priority=200,
    hidden=False,
    desc="A plugin for generating images using NanoBanana API",
    version="1.0",
    author="Lingyuzhou",
)
class NanoBananaImage(Plugin):
    def __init__(self):
        super().__init__()
        try:
            self.config = super().load_config()
            if not self.config:
                raise Exception("配置未找到")
            
            # 从配置文件加载api_key配置
            self.api_key = self.config.get("api_key")
            if not self.api_key:
                raise Exception("在配置中未找到API密钥")

            # 从配置文件加载base_url配置
            self.base_url = self.config.get("base_url", "https://api.agentify.top/v1/chat/completions")

            # 从配置文件加载指令关键词
            self.commands = self.config.get("commands", ["N画图"])
            self.edit_last_commands = self.config.get("edit_last_commands", ["N编辑"])
            self.reference_edit_commands = self.config.get("reference_edit_commands", ["N改图"])
            self.merge_commands = self.config.get("merge_commands", ["O融图"])
            
            # 初始化图片编辑状态管理
            self.waiting_for_image = {}
            self.image_prompts = {}
            self.conversations = {}
            self.conversation_timestamps = {}
            self.reference_image_wait_timeout = 180  # 等待参考图片超时时间（秒）
            
            # 融图功能相关变量
            self.waiting_for_merge_image = {}  # 用户ID -> 等待融图的提示词
            self.waiting_for_merge_image_time = {}  # 用户ID -> 开始等待融图的时间戳
            self.merge_image_wait_timeout = 180  # 等待融图的超时时间(秒)
            self.merge_image_first = {}  # 用户ID -> 第一张图片的数据
            
            # 初始化图片缓存
            self.image_cache = {}  # 会话ID/用户ID -> {"data": 图片数据, "timestamp": 时间戳}
            self.image_cache_timeout = 600  # 图片缓存过期时间(秒)
            
            # 初始化图片处理相关变量
            self.max_image_size = 10 * 1024 * 1024  # 最大图片大小限制(10MB)
            self.allowed_image_formats = ['jpeg', 'jpg', 'png', 'webp']  # 允许的图片格式
            self.max_image_dimension = 4096  # 最大图片尺寸
            
            # 临时文件管理
            self.temp_files = []  # 存储创建的临时文件路径
            self.temp_file_cleanup_interval = 3600  # 临时文件清理间隔(秒)
            self.last_cleanup_time = time.time()
                        
            # 配置 requests session
            self.session = requests.Session()
            retries = Retry(
                total=5,  # 总共重试5次
                backoff_factor=1.0,  # 重试间隔时间
                status_forcelist=[500, 502, 503, 504, 429],  # 需要重试的HTTP状态码
                allowed_methods=["GET", "POST"]  # 允许重试的请求方法
            )
            self.session.mount('http://', HTTPAdapter(max_retries=retries))
            self.session.mount('https://', HTTPAdapter(max_retries=retries))
            
            # 注册消息处理器
            self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
            
            logger.info("[NanoBananaImage] 插件初始化成功")
            
        except Exception as e:
            logger.error(f"[NanoBananaImage] 初始化失败: {e}")
            logger.exception(e)
            raise e
    
    def _cleanup_merge_image_state(self, user_id: str):
        """清理用户的融图状态"""
        try:
            if user_id in self.merge_image_first:
                del self.merge_image_first[user_id]
            if user_id in self.waiting_for_merge_image:
                del self.waiting_for_merge_image[user_id]
            if user_id in self.waiting_for_merge_image_time:
                del self.waiting_for_merge_image_time[user_id]
            logger.info(f"[NanoBananaImage] 已清理用户 {user_id} 的融图状态")
        except Exception as e:
            logger.error(f"[NanoBananaImage] 清理用户 {user_id} 的融图状态失败: {str(e)}")
            logger.exception(e)
    
    def _cleanup_image_cache(self):
        """清理过期的图片缓存"""
        try:
            current_time = time.time()
            expired_keys = []
            
            # 找出所有过期的缓存项
            for key, cache_item in self.image_cache.items():
                if current_time - cache_item["timestamp"] > self.image_cache_timeout:
                    expired_keys.append(key)
            
            # 删除过期项
            for key in expired_keys:
                del self.image_cache[key]
                logger.info(f"[NanoBananaImage] 已清理过期图片缓存: {key}")
        except Exception as e:
            logger.error(f"[NanoBananaImage] 清理图片缓存失败: {str(e)}")
            logger.exception(e)
    
    def _cleanup_temp_files(self):
        """清理过期的临时文件"""
        try:
            current_time = time.time()
            
            # 检查是否需要清理
            if current_time - self.last_cleanup_time < self.temp_file_cleanup_interval:
                return
            
            cleaned_files = []
            remaining_files = []
            
            for temp_file_path in self.temp_files:
                try:
                    if os.path.exists(temp_file_path):
                        # 检查文件创建时间
                        file_mtime = os.path.getmtime(temp_file_path)
                        if current_time - file_mtime > self.temp_file_cleanup_interval:
                            os.remove(temp_file_path)
                            cleaned_files.append(temp_file_path)
                            logger.info(f"[NanoBananaImage] 已清理过期临时文件: {temp_file_path}")
                        else:
                            remaining_files.append(temp_file_path)
                    else:
                        # 文件已不存在，从列表中移除
                        cleaned_files.append(temp_file_path)
                except Exception as e:
                    logger.error(f"[NanoBananaImage] 清理临时文件失败 {temp_file_path}: {e}")
                    remaining_files.append(temp_file_path)
            
            # 更新临时文件列表
            self.temp_files = remaining_files
            self.last_cleanup_time = current_time
            
            if cleaned_files:
                logger.info(f"[NanoBananaImage] 临时文件清理完成，清理了 {len(cleaned_files)} 个文件")
                
        except Exception as e:
            logger.error(f"[NanoBananaImage] 临时文件清理失败: {str(e)}")
            logger.exception(e)
            
    def _handle_api_result(self, result, e_context):
        """统一处理API返回结果"""
        try:
            # 定期清理临时文件和缓存
            self._cleanup_temp_files()
            self._cleanup_image_cache()
            
            user_id = e_context["context"].get("session_id") or e_context["context"].get("from_user_id")
            request_id = f"req_{int(time.time() * 1000)}" # 为处理结果生成ID
            logger.info(f"[NanoBananaImage] [{request_id}] 开始处理API结果，用户ID: {user_id}")
            logger.debug(f"[NanoBananaImage] [{request_id}] 原始结果: {self._safe_json_dumps(result)}")

            if not isinstance(result, dict):
                logger.error(f"[NanoBananaImage] [{request_id}] 无效的结果类型: {type(result)}")
                reply = Reply(ReplyType.ERROR, "内部错误：无法处理API响应。")
                e_context["reply"] = reply
                return # 返回，因为无法处理

            final_reply_set = False # 标记是否已设置最终回复
            image_urls = [] # 存储所有找到的图片URL

            # 从choices中提取图片和文本内容
            if "choices" in result and result["choices"]:
                for choice in result["choices"]:
                    if "message" in choice:
                        message = choice["message"]
                        
                        # 首先尝试从新的API格式中提取图片（message.images）
                        if "images" in message and message["images"]:
                            logger.info(f"[NanoBananaImage] [{request_id}] 发现新的API格式图片数组，数量: {len(message['images'])}")
                            for image_item in message["images"]:
                                if image_item.get("type") == "image_url" and "image_url" in image_item:
                                    image_url = image_item["image_url"].get("url")
                                    if image_url:
                                        # 处理base64格式的图片URL
                                        if image_url.startswith("data:image/"):
                                            try:
                                                # 提取base64数据
                                                if ";base64," in image_url:
                                                    header, base64_data = image_url.split(";base64,", 1)
                                                    logger.info(f"[NanoBananaImage] [{request_id}] 检测到base64格式图片，长度: {len(base64_data)}, 预览: {self._truncate_base64_log(base64_data)}")
                                                    
                                                    # 将base64数据转换为临时文件
                                                    try:
                                                        # 解码base64数据
                                                        image_data = base64.b64decode(base64_data)
                                                        logger.info(f"[NanoBananaImage] [{request_id}] base64解码成功，图片大小: {len(image_data)} 字节")
                                                        
                                                        # 验证图片数据
                                                        img = Image.open(BytesIO(image_data))
                                                        img_format = img.format.lower()
                                                        img_size = img.size
                                                        logger.info(f"[NanoBananaImage] [{request_id}] 图片验证成功: 格式={img_format}, 尺寸={img_size}")
                                                        
                                                        # 创建临时文件
                                                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{img_format}') as temp_file:
                                                            temp_file.write(image_data)
                                                            temp_file_path = temp_file.name
                                                        
                                                        # 记录临时文件路径
                                                        self.temp_files.append(temp_file_path)
                                                        logger.info(f"[NanoBananaImage] [{request_id}] 成功创建临时图片文件: {temp_file_path}")
                                                        image_urls.append(temp_file_path)
                                                        
                                                    except Exception as decode_error:
                                                        logger.error(f"[NanoBananaImage] [{request_id}] base64解码或文件创建失败: {decode_error}")
                                                        # 尝试进行容错解码，避免把data URL直接传给下游导致发送失败
                                                        try:
                                                            safe_data = re.sub(r'[^A-Za-z0-9+/=]', '', base64_data)
                                                            missing_padding = (-len(safe_data)) % 4
                                                            if missing_padding:
                                                                safe_data += '=' * missing_padding
                                                            image_data = base64.b64decode(safe_data)
                                                            img = Image.open(BytesIO(image_data))
                                                            img_format = img.format.lower()
                                                            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{img_format}') as temp_file:
                                                                temp_file.write(image_data)
                                                                temp_file_path = temp_file.name
                                                            self.temp_files.append(temp_file_path)
                                                            logger.info(f"[NanoBananaImage] [{request_id}] 容错解码成功，已创建临时图片文件: {temp_file_path}")
                                                            image_urls.append(temp_file_path)
                                                        except Exception as e2:
                                                            logger.error(f"[NanoBananaImage] [{request_id}] 容错解码仍失败，跳过该图片: {e2}")
                                                            # 不再将data URL当作URL传递，避免下游失败
                                                            pass
                                                else:
                                                    image_urls.append(image_url)
                                            except Exception as e:
                                                logger.error(f"[NanoBananaImage] [{request_id}] 处理base64图片URL失败: {e}")
                                        else:
                                            # 处理普通HTTP/HTTPS URL
                                            image_urls.append(image_url)
                                            logger.info(f"[NanoBananaImage] [{request_id}] 添加HTTP图片URL: {image_url}")
                        
                        # 如果没有从新格式找到图片，尝试从content中提取markdown格式的图片URL（向后兼容）
                        if not image_urls and "content" in message:
                            content = message["content"]
                            markdown_images = re.findall(r'!\[.*?\]\((.*?)\)', content)
                            if markdown_images:
                                image_urls.extend(markdown_images)
                                logger.info(f"[NanoBananaImage] [{request_id}] 从markdown内容中提取到 {len(markdown_images)} 个图片URL")
                        
                        # 处理文本内容
                        if "content" in message:
                            content = message["content"]
                            # 如果有图片，从content中移除图片标记，保留纯文本
                            if image_urls:
                                # 移除markdown格式的图片标记
                                url_matches = list(re.finditer(r'!\[.*?\]\((.*?)\)|\[.*?\]\((https?://[^\s\)]+)\)', content))
                                for match in url_matches:
                                    content = content.replace(match.group(0), '')
                            
                            cleaned_content = content.strip()
                            # 如果没有找到图片URL，但有文本内容，将其作为文本回复
                            if not image_urls and not ("image_urls" in result and result["image_urls"]) and cleaned_content:
                                text_reply = Reply(ReplyType.TEXT, cleaned_content)
                                e_context["reply"] = text_reply
                                final_reply_set = True
                                logger.info(f"[NanoBananaImage] [{request_id}] 使用API返回的文本内容作为回复")
                                return

            # 处理文本响应
            if "text_responses" in result and result["text_responses"]:
                logger.info(f"[NanoBananaImage] [{request_id}] 发现 {len(result['text_responses'])} 个文本响应")
                full_text_response = "\n".join(text for text in result["text_responses"] if text)
                if full_text_response:
                    text_reply = Reply(ReplyType.TEXT, full_text_response)
                    # 如果没有图片，文本就是最终回复；如果有图片，文本通过channel发送
                    if not image_urls and not ("image_urls" in result and result["image_urls"]):
                        logger.info(f"[NanoBananaImage] [{request_id}] 设置最终文本回复")
                        e_context["reply"] = text_reply
                        final_reply_set = True
                    else:
                        logger.info(f"[NanoBananaImage] [{request_id}] 通过channel发送附加文本回复")
                        e_context["channel"].send(text_reply, e_context["context"])
                else:
                    logger.info(f"[NanoBananaImage] [{request_id}] 文本响应为空")

            # 添加API直接返回的图片URL
            if "image_urls" in result and result["image_urls"]:
                image_urls.extend(result["image_urls"])

            # 处理所有收集到的图片URL
            if image_urls:
                logger.info(f"[NanoBananaImage] [{request_id}] 总共发现 {len(image_urls)} 个图片URL")
                # 通常我们只取第一个URL作为主要回复
                image_url_to_send = image_urls[0]
                try:
                    # 判断是文件路径还是URL
                    if os.path.isfile(image_url_to_send):
                        # 如果是临时文件路径，使用IMAGE类型
                        logger.info(f"[NanoBananaImage] [{request_id}] 检测到临时文件路径，使用IMAGE类型回复: {image_url_to_send}")
                        image_reply = Reply(ReplyType.IMAGE, image_url_to_send)
                    else:
                        # 如果是URL，使用IMAGE_URL类型
                        logger.info(f"[NanoBananaImage] [{request_id}] 检测到图片URL，使用IMAGE_URL类型回复: {image_url_to_send}")
                        image_reply = Reply(ReplyType.IMAGE_URL, image_url_to_send)
                    
                    logger.info(f"[NanoBananaImage] [{request_id}] 设置最终图片回复: {image_url_to_send}")
                    e_context["reply"] = image_reply # 图片优先作为最终回复
                    final_reply_set = True

                    # 如果有多张图片，可以通过channel额外发送
                    if len(image_urls) > 1:
                        logger.info(f"[NanoBananaImage] [{request_id}] 发现多张图片，将通过channel发送其余图片")
                        for url in image_urls[1:]:
                            if os.path.isfile(url):
                                additional_reply = Reply(ReplyType.IMAGE, url)
                            else:
                                additional_reply = Reply(ReplyType.IMAGE_URL, url)
                            e_context["channel"].send(additional_reply, e_context["context"])
                except Exception as e:
                    logger.error(f"[NanoBananaImage] [{request_id}] 处理图片URL时出错: {str(e)}")
                    logger.exception(e)
                    reply = Reply(ReplyType.ERROR, f"处理图片时出错: {str(e)}")
                    e_context["reply"] = reply
                    final_reply_set = True
            else:
                logger.info(f"[NanoBananaImage] [{request_id}] 未找到任何图片URL")

            # 如果没有设置任何回复，检查是否有错误信息或其他内容可以返回
            if not final_reply_set:
                error_message = None
                # 从choices中提取错误信息
                if "choices" in result and result["choices"]:
                    for choice in result["choices"]:
                        if "message" in choice:
                            message = choice["message"]
                            # 优先从content字段提取错误信息
                            if "content" in message:
                                content = message["content"]
                                # 如果有图片，从content中移除图片标记
                                if image_urls:
                                    url_matches = list(re.finditer(r'!\[.*?\]\((.*?)\)|\[.*?\]\((https?://[^\s\)]+)\)', content))
                                    for match in url_matches:
                                        content = content.replace(match.group(0), '')
                                error_message = content.strip()
                                break
                
                if error_message:
                    logger.warning(f"[NanoBananaImage] [{request_id}] API返回错误信息: {error_message}")
                    reply = Reply(ReplyType.ERROR, error_message)
                else:
                    logger.warning(f"[NanoBananaImage] [{request_id}] 未能从API结果中生成有效回复")
                    reply = Reply(ReplyType.ERROR, "未能生成有效内容，请检查您的输入或稍后再试。")
                
                e_context["reply"] = reply

        except Exception as e:
            logger.error(f"[NanoBananaImage] 处理API结果时发生意外错误: {e}")
            logger.exception(e)
            reply = Reply(ReplyType.ERROR, f"处理结果时出错: {str(e)}")
            e_context["reply"] = reply # 覆盖之前的回复，报告处理错误

        return

    def on_handle_context(self, e_context: EventContext):
        context = e_context["context"]
        user_id = context.get("session_id") or context.get("from_user_id")
        content = context.content
        
        # 处理图片消息
        if context.type == ContextType.IMAGE:
            # 检查是否在等待融图
            if user_id in self.waiting_for_merge_image:
                try:
                    # 获取图片数据
                    image_data = self._get_image_data(context)
                    if not image_data:
                        reply = Reply(ReplyType.TEXT, "无法获取图片数据，请重新上传图片。")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                    
                    # 检查是否超时
                    current_time = time.time()
                    start_time = self.waiting_for_merge_image_time.get(user_id, 0)
                    if current_time - start_time > self.merge_image_wait_timeout:
                        # 清理状态
                        self._cleanup_merge_image_state(user_id)
                        reply = Reply(ReplyType.TEXT, f"等待上传图片超时（超过{self.merge_image_wait_timeout//60}分钟），请重新开始融图操作")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                    
                    # 处理融图逻辑
                    if user_id not in self.merge_image_first:
                        # 保存第一张图片
                        self.merge_image_first[user_id] = image_data
                        # 更新等待时间戳
                        self.waiting_for_merge_image_time[user_id] = time.time()
                        # 发送成功获取第一张图片的提示
                        reply = Reply(ReplyType.TEXT, "✅ 成功获取图一，请继续发送图二")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                    else:
                        # 处理第二张图片
                        prompt = self.waiting_for_merge_image[user_id]
                        first_image_data = self.merge_image_first[user_id]
                        
                        # 清除状态
                        self._cleanup_merge_image_state(user_id)
                        
                        # 发送成功获取第二张图片的提示
                        success_reply = Reply(ReplyType.TEXT, "⏳ 成功获取图二，正在处理中...")
                        e_context["channel"].send(success_reply, e_context["context"])
                        
                        # 记录融图API请求信息
                        request_id = f"merge_{int(time.time() * 1000)}"
                        logger.info(f"[NanoBananaImage] [{request_id}] 准备发送融图API请求，用户ID: {user_id}, 提示词: {prompt}")
                        logger.info(f"[NanoBananaImage] [{request_id}] 图片1大小: {len(first_image_data)} 字节, 图片2大小: {len(image_data)} 字节")

                        # 检查图片大小是否过大
                        max_size = 10 * 1024 * 1024  # 10MB
                        if len(first_image_data) > max_size or len(image_data) > max_size:
                            logger.warning(f"[NanoBananaImage] 图片大小超过10MB，尝试压缩图片")
                            try:
                                def compress_image(image_data, max_size=1024*1024):
                                    img = Image.open(BytesIO(image_data))
                                    quality = 95
                                    output = BytesIO()
                                    img.save(output, format='JPEG', quality=quality)
                                    while output.tell() > max_size and quality > 30:
                                        output = BytesIO()
                                        quality -= 5
                                        img.save(output, format='JPEG', quality=quality)
                                    output.seek(0)
                                    return output.getvalue()
                                
                                if len(first_image_data) > max_size:
                                    first_image_data = compress_image(first_image_data)
                                    logger.info(f"[NanoBananaImage] 压缩后图片1大小: {len(first_image_data)} 字节")
                                
                                if len(image_data) > max_size:
                                    image_data = compress_image(image_data)
                                    logger.info(f"[NanoBananaImage] 压缩后图片2大小: {len(image_data)} 字节")
                            except Exception as e:
                                logger.error(f"[NanoBananaImage] 压缩图片失败: {str(e)}")
                                logger.exception(e)
                                reply = Reply(ReplyType.TEXT, "图片压缩失败，请尝试上传较小的图片。")
                                e_context["reply"] = reply
                                e_context.action = EventAction.BREAK_PASS
                                return
                        
                        # 调用API进行融图
                        try:
                            # 转换图片为base64
                            first_image_base64 = base64.b64encode(first_image_data).decode('utf-8')
                            second_image_base64 = base64.b64encode(image_data).decode('utf-8')
                            
                            # 构建API请求参数
                            headers = {
                                'Accept': 'application/json',
                                'Content-Type': 'application/json',
                                'Authorization': f'Bearer {self.api_key}'
                            }
                            
                            # 构建请求数据
                            data = {
                                "model": self.config.get("model", "google/gemini-2.5-flash-image-preview:free"),
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": f"{prompt}\n\n 请务必生成一个高质量的融合图像作为回复。"
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{first_image_base64}"}
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{second_image_base64}"}
                                            }
                                        ]
                                    }
                                ],
                                "stream": False,
                                "max_tokens": self.config.get("max_tokens", 8192),
                                "temperature": self.config.get("temperature", 0.7)
                            }
                            
                            # 记录完整的API请求信息
                            logger.info(f"[NanoBananaImage] 发送融图API请求，请求URL: {self.base_url}")
                            # 记录请求头信息（移除Authorization敏感信息）
                            safe_headers = headers.copy()
                            safe_headers['Authorization'] = '***'
                            logger.debug(f"[NanoBananaImage] 请求头信息: {json.dumps(safe_headers)}")
                            # 记录请求数据（使用安全的JSON序列化）
                            logger.debug(f"[NanoBananaImage] 请求数据: {self._safe_json_dumps(data)}")
                            
                            # 发送API请求
                            response = self.session.post(
                                self.base_url,
                                headers=headers,
                                json=data,
                                timeout=180  # 3分钟超时
                            )
                            response.raise_for_status()
                            result = response.json()
                            
                            # 记录API响应信息
                            logger.info(f"[NanoBananaImage] API响应状态码: {response.status_code}")
                            logger.debug(f"[NanoBananaImage] API响应头: {dict(response.headers)}")
                            logger.debug(f"[NanoBananaImage] API响应内容: {self._safe_json_dumps(result)}")
                            
                            # 检查model字段不一致的情况
                            if result.get('model') != data['model']:
                                logger.warning(f"[NanoBananaImage] 请求的model({data['model']})与响应的model({result.get('model')})不一致")
                            
                            # 处理API返回结果
                            self._handle_api_result(result, e_context)
                            e_context.action = EventAction.BREAK_PASS
                            return
                            
                        except Exception as e:
                            logger.error(f"[NanoBananaImage] 融图API调用失败: {e}")
                            reply = Reply(ReplyType.ERROR, f"融图失败: {str(e)}")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                except Exception as e:
                    logger.error(f"[NanoBananaImage] 处理融图图片消息失败: {e}")
                    reply = Reply(ReplyType.ERROR, f"处理图片失败: {str(e)}")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
            
                    
            # 如果不是融图，检查是否在等待其他图片上传
            if user_id not in self.waiting_for_image:
                return
            
            try:
                # 检查是否超时
                current_time = time.time()
                start_time = self.conversation_timestamps.get(user_id, 0)
                
                # 记录当前状态
                logger.debug(f"[NanoBananaImage] 处理图片消息，用户ID: {user_id}, 等待状态: {self.waiting_for_image.get(user_id)}, 开始时间: {start_time}")
                
                if current_time - start_time > self.reference_image_wait_timeout:
                    # 超时，清理状态
                    del self.waiting_for_image[user_id]
                    if user_id in self.image_prompts:
                        del self.image_prompts[user_id]
                    if user_id in self.conversation_timestamps:
                        del self.conversation_timestamps[user_id]
                    
                    logger.info(f"[NanoBananaImage] 用户 {user_id} 等待上传参考图片超时，自动结束流程")
                    reply = Reply(ReplyType.TEXT, f"等待上传参考图片超时（超过{self.reference_image_wait_timeout//60}分钟），已自动取消操作。如需继续，请重新发送参考图编辑命令。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 验证等待状态
                if not self.waiting_for_image.get(user_id):
                    logger.error(f"[NanoBananaImage] 用户 {user_id} 的等待状态已丢失")
                    reply = Reply(ReplyType.TEXT, "会话状态已丢失，请重新发送参考图编辑命令。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 获取图片数据
                logger.debug(f"[NanoBananaImage] 开始获取图片数据，用户ID: {user_id}")
                image_data = self._get_image_data(context)
                
                # 提示用户图片获取成功
                success_reply = Reply(ReplyType.TEXT, "⏳ 成功获取图片，正在处理中...")
                e_context["channel"].send(success_reply, e_context["context"])
                
                # 验证图片数据
                if not image_data:
                    logger.error(f"[NanoBananaImage] 获取图片数据失败，用户ID: {user_id}")
                    reply = Reply(ReplyType.TEXT, "无法获取图片数据，请重新上传图片。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    # 保持等待状态，让用户可以重试
                    return
                                
                if len(image_data) < 100:
                    logger.error(f"[NanoBananaImage] 图片数据过小，大小: {len(image_data)} 字节，用户ID: {user_id}")
                    reply = Reply(ReplyType.TEXT, "图片数据无效，请确保上传的是有效的图片文件。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    # 保持等待状态，让用户可以重试
                    return
                
                try:
                    # 验证图片格式和大小
                    img = Image.open(BytesIO(image_data))
                    img_format = img.format.lower()
                    img_size = img.size
                    
                    # 检查图片格式
                    allowed_formats = ['jpeg', 'jpg', 'png', 'webp']
                    if img_format not in allowed_formats:
                        logger.error(f"不支持的图片格式: {img_format}")
                        reply = Reply(ReplyType.TEXT, f"不支持的图片格式: {img_format}，仅支持: {', '.join(allowed_formats)}")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                    
                    # 检查图片尺寸
                    max_dimension = 4096
                    if img_size[0] > max_dimension or img_size[1] > max_dimension:
                        logger.error(f"图片尺寸过大: {img_size}")
                        reply = Reply(ReplyType.TEXT, f"图片尺寸过大: {img_size[0]}x{img_size[1]}，最大允许: {max_dimension}x{max_dimension}")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                    
                    logger.info(f"收到有效的图片数据，格式: {img_format}, 尺寸: {img_size}, 大小: {len(image_data)} 字节")
                                                    
                except Exception as e:
                    logger.error(f"[NanoBananaImage] 图片验证失败: {e}")
                    reply = Reply(ReplyType.TEXT, "无法处理图片，请确保上传的是有效的图片文件。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 获取之前保存的提示词
                prompt = self.image_prompts.get(user_id)
                if not prompt:
                    logger.error(f"未找到用户{user_id}的提示词")
                    reply = Reply(ReplyType.TEXT, "会话已过期，请重新发送编辑命令。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
               
                # 调用API编辑图片
                result = self.generate_image(prompt, True, image_data, user_id, e_context)
                
                # 清理状态
                del self.waiting_for_image[user_id]
                del self.image_prompts[user_id]
                
                if not result:
                    reply = Reply(ReplyType.ERROR, "图片编辑失败，请稍后重试")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 处理API返回结果
                has_sent_image = False
                
                # 检查是否需要等待图片生成
                if result.get("wait_for_image"):
                    logger.warning(f"[NanoBananaImage] 检测到需要等待图片生成，图片ID: {result.get('image_id')}")
                    reply = Reply(ReplyType.TEXT, "图片正在生成中，请稍等片刻后重试，或检查API配置是否正确。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 先处理图片URL
                if "image_urls" in result and result["image_urls"]:
                    for image_url in result["image_urls"]:
                        try:
                            # 构建图片回复（本地文件用IMAGE，URL用IMAGE_URL）
                            image_url = self._materialize_data_url(image_url)
                            reply_type = ReplyType.IMAGE if os.path.isfile(image_url) else ReplyType.IMAGE_URL
                            image_reply = Reply(reply_type, image_url)
                            # 发送图片
                            e_context["channel"].send(image_reply, e_context["context"])
                            logger.info(f"[NanoBananaImage] 发送图片: {image_url}")
                            has_sent_image = True
                            break  # 只处理第一个图片URL
                        except Exception as e:
                            logger.error(f"[NanoBananaImage] 发送图片URL失败: {e}")
                            continue
                
                # 再处理文本响应
                if "text_responses" in result and result["text_responses"]:
                    for text in result["text_responses"]:
                        if text:
                            text_reply = Reply(ReplyType.TEXT, text)
                            e_context["channel"].send(text_reply, e_context["context"])
                
                if has_sent_image:
                    # 发送耗时信息
                    duration_reply = Reply(ReplyType.TEXT, result["duration_msg"])
                    e_context["channel"].send(duration_reply, e_context["context"])
                    e_context.action = EventAction.BREAK_PASS
                    return
                
            except Exception as e:
                logger.error(f"[NanoBananaImage] 处理图片消息失败: {e}")
                logger.exception(e)
                reply = Reply(ReplyType.ERROR, f"处理图片失败: {str(e)}")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return

        # 处理文本消息
        if context.type != ContextType.TEXT:
            return
            
        content = context.content
        # 检查是否是支持的指令
        all_commands = self.commands + self.edit_last_commands + self.reference_edit_commands + self.merge_commands
        if not content.startswith(tuple(all_commands)):
            return
            
        logger.debug(f"[NanoBananaImage] 收到消息: {content}")
        
        # 引用图片的“N改图”快捷改图：检测消息对象的引用信息并直接处理
        actual_msg_object = context.kwargs.get('msg') if hasattr(context, 'kwargs') and context.kwargs else None
        if (actual_msg_object and
            hasattr(actual_msg_object, 'is_processed_image_quote') and getattr(actual_msg_object, 'is_processed_image_quote') and
            hasattr(actual_msg_object, 'referenced_image_path') and getattr(actual_msg_object, 'referenced_image_path') and
            content and content.startswith(tuple(self.reference_edit_commands))):
            logger.info(f"[NanoBananaImage] 检测到引用图片的N改图命令: {content}")
            try:
                self._handle_referenced_image_to_image(e_context, content, getattr(actual_msg_object, 'referenced_image_path'))
                e_context.action = EventAction.BREAK_PASS
                return
            except Exception as ex:
                logger.error(f"[NanoBananaImage] 引用图片改图处理失败: {ex}")
        
        try:
            # 判断是文生图、编辑参考图、编辑最近生成的图片还是融图
            is_edit_reference_image = content.startswith(tuple(self.reference_edit_commands))
            is_edit_last_image = content.startswith(tuple(self.edit_last_commands))
            is_merge_image = content.startswith(tuple(self.merge_commands))
            
            # 移除前缀
            for prefix in self.commands:
                if content.startswith(prefix):
                    content = content[len(prefix):].strip()
                    break
            
            # 处理编辑最近生成的图片
            if is_edit_last_image:
                # 检查是否有缓存的图片URL
                if user_id not in self.image_cache or "urls" not in self.image_cache[user_id]:
                    reply = Reply(ReplyType.TEXT, "没有找到最近生成的图片，请先使用生成图片功能。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 获取最近生成的图片URL
                image_urls = self.image_cache[user_id]["urls"]
                if not image_urls:
                    reply = Reply(ReplyType.TEXT, "最近生成的图片已过期，请重新生成图片。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 获取第一张图片的数据（兼容本地文件路径或HTTP/HTTPS）
                try:
                    first_url = image_urls[0]
                    if os.path.isfile(first_url):
                        with open(first_url, "rb") as f:
                            image_data = f.read()
                    else:
                        response = self.session.get(first_url, timeout=30)
                        response.raise_for_status()
                        image_data = response.content
                except Exception as e:
                    logger.error(f"[NanoBananaImage] 获取缓存图片失败: {e}")
                    reply = Reply(ReplyType.ERROR, "获取最近生成的图片失败，请重新生成图片。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                               
                # 调用API编辑图片（去除指令前缀）
                prompt_for_edit_last = content
                for prefix in self.edit_last_commands:
                    if prompt_for_edit_last.startswith(prefix):
                        prompt_for_edit_last = prompt_for_edit_last[len(prefix):].strip()
                        break
                result = self.generate_image(prompt_for_edit_last, True, image_data, user_id, e_context)
                
                if not result:
                    reply = Reply(ReplyType.ERROR, "图片编辑失败，请稍后重试")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 处理API返回结果
                # 检查是否需要等待图片生成
                if result.get("wait_for_image"):
                    logger.warning(f"[NanoBananaImage] 检测到需要等待图片生成，图片ID: {result.get('image_id')}")
                    reply = Reply(ReplyType.TEXT, "图片正在生成中，请稍等片刻后重试，或检查API配置是否正确。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                if "text_responses" in result and result["text_responses"]:
                    for text in result["text_responses"]:
                        if text:
                            text_reply = Reply(ReplyType.TEXT, text)
                            e_context["channel"].send(text_reply, e_context["context"])
                
                if "image_urls" in result and result["image_urls"]:
                    for image_url in result["image_urls"]:
                        try:
                            # 构建图片回复（本地文件用IMAGE，URL用IMAGE_URL）
                            image_url = self._materialize_data_url(image_url)
                            reply_type = ReplyType.IMAGE if os.path.isfile(image_url) else ReplyType.IMAGE_URL
                            image_reply = Reply(reply_type, image_url)
                            # 设置回复到e_context["reply"]
                            e_context["reply"] = image_reply
                            logger.info(f"[NanoBananaImage] 设置图片URL回复: {image_url}")
                        except Exception as e:
                            logger.error(f"[NanoBananaImage] 设置图片URL回复失败: {e}")
                            continue
                
                # 发送耗时信息
                if "duration_msg" in result:
                    duration_reply = Reply(ReplyType.TEXT, result["duration_msg"])
                    e_context["channel"].send(duration_reply, e_context["context"])
                
                e_context.action = EventAction.BREAK_PASS
                return
            
            if is_edit_reference_image:
                # 保存提示词和等待状态（去除指令前缀）
                self.waiting_for_image[user_id] = True
                ref_prompt = content
                for prefix in self.reference_edit_commands:
                    if ref_prompt.startswith(prefix):
                        ref_prompt = ref_prompt[len(prefix):].strip()
                        break
                self.image_prompts[user_id] = ref_prompt
                self.conversation_timestamps[user_id] = time.time()
                
                # 提示用户上传图片
                reply = Reply(ReplyType.TEXT, "请发送需要编辑的参考图片，等待时间3分钟")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
            if is_merge_image:
                # 从命令中提取提示词
                for prefix in self.merge_commands:
                    if content.startswith(prefix):
                        prompt = content[len(prefix):].strip()
                        break
                
                # 保存提示词和等待状态
                self.waiting_for_merge_image[user_id] = prompt
                self.waiting_for_merge_image_time[user_id] = time.time()
                
                # 提示用户上传图片
                reply = Reply(ReplyType.TEXT, "请发送需要融图的第一张图片，等待时间3分钟")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            # 提示用户任务已提交
            submit_reply = Reply(ReplyType.TEXT, "正在调用NanoBanana生成图片，请耐心等待...")
            e_context["channel"].send(submit_reply, e_context["context"])
            
            # 调用API生成图片
            result = self.generate_image(content, is_edit_reference_image, None, user_id, e_context)
            
            if not result:
                reply = Reply(ReplyType.ERROR, "生成失败，请稍后重试")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return

            # 检查是否需要等待图片生成
            if result.get("wait_for_image"):
                logger.warning(f"[NanoBananaImage] 检测到需要等待图片生成，图片ID: {result.get('image_id')}")
                reply = Reply(ReplyType.TEXT, "图片正在生成中，请稍等片刻后重试，或检查API配置是否正确。")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return

            # 先处理图片URL
            if "image_urls" in result and result["image_urls"]:
                for image_url in result["image_urls"]:
                    try:
                        # 构建图片回复（本地文件用IMAGE，URL用IMAGE_URL）并直接发送
                        image_url = self._materialize_data_url(image_url)
                        reply_type = ReplyType.IMAGE if os.path.isfile(image_url) else ReplyType.IMAGE_URL
                        image_reply = Reply(reply_type, image_url)
                        e_context["channel"].send(image_reply, e_context["context"])
                        logger.info(f"[NanoBananaImage] 发送图片: {image_url}")
                        break  # 只处理第一个图片URL
                    except Exception as e:
                        logger.error(f"[NanoBananaImage] 发送图片URL失败: {e}")
                        continue

            # 再处理文本消息
            if "text_responses" in result and result["text_responses"]:
                for text in result["text_responses"]:
                    if text:
                        text_reply = Reply(ReplyType.TEXT, text)
                        e_context["channel"].send(text_reply, e_context["context"])

            # 发送耗时信息
            if "duration_msg" in result:
                duration_reply = Reply(ReplyType.TEXT, result["duration_msg"])
                e_context["channel"].send(duration_reply, e_context["context"])

            e_context.action = EventAction.BREAK_PASS
            return
            
            # 验证图片URL并发送
            has_sent_content = False
            if "image_urls" in result and result["image_urls"]:
                for image_url in result["image_urls"]:
                    try:
                        img_response = self.session.head(image_url, timeout=10)
                        img_response.raise_for_status()
                        if img_response.headers.get('content-type', '').startswith('image/'):
                            image_reply = Reply(ReplyType.IMAGE_URL, image_url)
                            e_context["channel"].send(image_reply, e_context["context"])
                            has_sent_content = True
                        else:
                            logger.warning(f"[NanoBananaImage] 跳过无效的图片URL: {image_url}")
                    except Exception as e:
                        logger.error(f"[NanoBananaImage] 图片URL验证失败: {e}")
                        continue

                if has_sent_content:
                    # 发送耗时信息
                    duration_reply = Reply(ReplyType.TEXT, result["duration_msg"])
                    e_context["channel"].send(duration_reply, e_context["context"])
                    e_context.action = EventAction.BREAK_PASS
                    return

            # 如果没有任何内容被发送
            reply = Reply(ReplyType.ERROR, "生成的内容无效或被过滤")
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS
            
        except Exception as e:
            logger.error(f"[NanoBananaImage] 发生错误: {e}")
            reply = Reply(ReplyType.ERROR, f"发生错误: {str(e)}")
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS


    def generate_image(self, prompt: str, is_edit_reference_image: bool = False, image_data: bytes =None, user_id: str = None, e_context: Optional[EventContext] = None, op_mode: Optional[str] = None) -> dict:
        """调用NanoBanana API生成图片"""
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"
        logger.info(f"[NanoBananaImage] [{request_id}] 开始生成图片请求")
        logger.debug(f"[NanoBananaImage] [{request_id}] 请求参数: prompt='{prompt}', is_edit_reference_image={is_edit_reference_image}")
        
        # 验证编辑参考图模式下的图片数据
        if is_edit_reference_image:
            if not image_data:
                logger.error(f"[NanoBananaImage] [{request_id}] 编辑参考图模式缺少图片数据")
                raise ValueError("编辑参考图模式需要提供图片数据")
            try:
                # 验证图片数据是否有效
                img = Image.open(BytesIO(image_data))
                img_format = img.format.lower()
                img_size = img.size
                logger.info(f"[NanoBananaImage] [{request_id}] 图片验证成功: 格式={img_format}, 尺寸={img_size}")
                
                # 检查图片大小和大小限制
                allowed_formats = ['jpeg', 'jpg', 'png', 'webp']
                if img_format not in allowed_formats:
                    raise ValueError(f"不支持的图片格式: {img_format}，仅支持: {', '.join(allowed_formats)}")
                
                max_dimension = 4096  # 最大尺寸限制
                if img_size[0] > max_dimension or img_size[1] > max_dimension:
                    raise ValueError(f"图片尺寸过大: {img_size}，最大允许: {max_dimension}x{max_dimension}")
                
                # 将图片数据转换为Base64编码
                image_base64 = base64.b64encode(image_data).decode("utf-8")
                # 只记录Base64编码的前20和后20个字符，以及总长度和图片大小
                preview_base64 = f"{image_base64[:20]}...{image_base64[-20:]}"
                img_size_kb = len(image_data) / 1024
                logger.info(f"[NanoBananaImage] [{request_id}] 成功编码图片数据 - 大小: {img_size_kb:.2f}KB, Base64预览: {preview_base64}")
            except ValueError as ve:
                logger.error(f"[NanoBananaImage] [{request_id}] 图片验证失败: {ve}")
                raise
            except Exception as e:
                logger.error(f"[NanoBananaImage] [{request_id}] 图片处理异常: {e}")
                logger.exception(e)
                raise ValueError(f"图片处理失败: {str(e)}")

        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # 构建请求数据
        if is_edit_reference_image:
            # 获取图片MIME类型
            img = Image.open(BytesIO(image_data))
            mime_type = f"image/{img.format.lower()}"
            
            # 检查图片尺寸和格式
            if min(img.size) <= 10:
                raise ValueError("图片尺寸过小，宽度和高度均应大于10像素")
                
            # 检查宽高比
            aspect_ratio = max(img.size) / min(img.size)
            if aspect_ratio > 200:
                raise ValueError("图片宽高比不合适，不应超过200:1或1:200")
            
            # 转换为base64并构建image_url
            base64_data = base64.b64encode(image_data).decode('utf-8')
            image_url = f"data:image/{img.format.lower()};base64,{base64_data}"
            
            data = {
                "model": self.config.get("model", "google/gemini-2.5-flash-image-preview:free"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt} "},
                            {"type": "image_url","image_url": {"url": image_url} },
                        ]
                    }
                ],
                "stream": False,
                "max_tokens": self.config.get("max_tokens", 8192),
                "temperature": self.config.get("temperature", 0.7)
            }
        else:
            data = {
                "model": self.config.get("model", "google/gemini-2.5-flash-image-preview:free"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt}"}
                        ]
                    }
                ],
                "stream": False,
                "max_tokens": self.config.get("max_tokens", 8192),
                "temperature": self.config.get("temperature", 0.7)
                }
        
        try:
            # 记录请求基本信息
            command_type = op_mode if op_mode else ("N改图" if is_edit_reference_image and user_id in self.waiting_for_image else "N编辑" if is_edit_reference_image else "N画图")
            logger.info(f"[NanoBananaImage] [{request_id}] 开始{command_type} - 提示词: '{prompt}'")
            
            # 发送处理提示
            if command_type == "N改图":
                processing_reply = Reply(ReplyType.TEXT, "正在调用NanoBanana进行图生图，请耐心等待...")
                e_context["channel"].send(processing_reply, e_context["context"])
            
            # 创建不包含敏感信息的请求头副本用于日志记录
            masked_headers = headers.copy()
            masked_headers['Authorization'] = '***'
            
            # 记录关键请求参数，省略敏感信息和冗长数据
            request_info = {
                'url': self.base_url,
                'headers': masked_headers,
                'body': data,
                'command_type': command_type
            }
            logger.debug(f"[NanoBananaImage] [{request_id}] 请求参数: {self._safe_json_dumps(request_info)}")
            
            # 设置较长的超时时间，因为图片生成可能需要较长时间
            logger.info(f"[NanoBananaImage] [{command_type}] [{request_id}] 正在发送API请求...")
            start_time = time.time()
            
            try:
                response = self.session.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=180  # 3分钟超时
                )
                response.raise_for_status()
                result = response.json()
                
                # 计算请求耗时
                request_duration = time.time() - start_time
                logger.info(f"[NanoBananaImage] [{command_type}] [{request_id}] API请求成功，耗时: {request_duration:.2f}秒")
                
                # 记录API响应信息
                response_info = {
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'body': result,
                    'duration': f"{request_duration:.2f}秒"
                }
                logger.debug(f"[NanoBananaImage] [{command_type}] [{request_id}] 完整API响应:\n{self._safe_json_dumps(response_info)}")
                
                # 记录响应结果统计
                if 'choices' in result and result['choices']:
                    content = result['choices'][0].get('message', {}).get('content', '')
                    image_count = len(re.findall(r'!\[.*?\]\((.*?)\)|\[.*?\]\((https?://[^\s)]+)\)', content))
                    text_length = len(re.sub(r'!\[.*?\]\((.*?)\)|\[.*?\]\((https?://[^\s)]+)\)', '', content).strip())
                    logger.info(f"[NanoBananaImage] [{command_type}] [{request_id}] 响应统计:\n图片数量: {image_count}\n文本长度: {text_length}字符")
            
                # 检查API响应格式
                if not isinstance(result, dict) or not result:
                    logger.error(f"[NanoBananaImage] [{request_id}] API响应格式无效或为空: {result}")
                    raise ValueError("API响应格式无效或为空，请稍后重试")
                
                error_msg = None
                if 'error' in result:
                    error_msg = result['error']
                if isinstance(error_msg, dict):
                    error_code = error_msg.get('code', 'unknown')
                    error_message = error_msg.get('message', str(error_msg))
                    error_details = error_msg.get('details', [])
                    
                    # 记录详细的错误信息
                    logger.error(f"[NanoBananaImage] [{request_id}] API错误:\n代码: {error_code}\n消息: {error_message}\n详情: {error_details}")
                    
                    # 根据错误代码分类处理
                    if error_code == 'rate_limit_exceeded':
                        raise ValueError("API请求频率超限，请等待几分钟后重试")
                    elif error_code == 'invalid_api_key':
                        raise ValueError("API密钥无效或已过期，请检查配置")
                    elif error_code == 'content_filter':
                        raise ValueError("内容被过滤，可能包含不适当的内容")
                    elif error_code == 'model_overloaded':
                        raise ValueError("模型负载过高，请稍后重试")
                    elif error_code == 'context_length_exceeded':
                        raise ValueError("输入内容过长，请缩短提示词或减少图片大小")
                    else:
                        raise ValueError(f"API错误: {error_message}")
                elif error_msg is not None:
                    logger.error(f"[NanoBananaImage] [{request_id}] 未知API错误: {error_msg}")
                    raise ValueError(f"未知API错误: {error_msg}")
            
                # 验证API响应的完整性
                if 'choices' not in result or not result['choices']:
                    logger.error(f"[NanoBananaImage] [{request_id}] API响应缺少choices字段:\n{self._safe_json_dumps(result)}")
                    raise ValueError("API响应格式无效：缺少必要的响应内容")
            
                # 获取第一个选择项的内容
                first_choice = result['choices'][0]
                if 'message' not in first_choice or 'content' not in first_choice['message']:
                    logger.error(f"[NanoBananaImage] [{request_id}] API响应格式错误:\n{self._safe_json_dumps(first_choice)}")
                    raise ValueError("API响应格式无效：响应内容结构异常")
            
                content = first_choice['message']['content']
                # 安全地记录API返回内容，避免base64数据刷屏
                safe_content = content
                if len(content) > 1000 and self._is_likely_base64(content):
                    safe_content = self._truncate_base64_log(content, 200)
                logger.debug(f"[NanoBananaImage] [{request_id}] API返回内容:\n{safe_content}")
            
                # 处理返回结果
                text_responses = []
                image_urls = []
                
                # 首先尝试从新的API格式中提取图片（choices[0].message.images）
                if 'images' in first_choice['message'] and first_choice['message']['images']:
                    logger.info(f"[NanoBananaImage] [{request_id}] 发现新的API格式图片数组，数量: {len(first_choice['message']['images'])}")
                    for image_item in first_choice['message']['images']:
                        if image_item.get('type') == 'image_url' and 'image_url' in image_item:
                            image_url = image_item['image_url'].get('url')
                            if image_url:
                                # 处理base64格式的图片URL
                                if image_url.startswith('data:image/'):
                                    # 对于base64格式，强制落地为本地文件再发送
                                    try:
                                        if ';base64,' in image_url:
                                            header, base64_data = image_url.split(';base64,', 1)
                                            # 解析mime用于后缀
                                            mime = header.replace('data:', '')
                                            ext = 'png'
                                            try:
                                                ext = mime.split('/')[1].split(';')[0]
                                            except Exception:
                                                pass
                                            # 写入指定缓存目录
                                            cache_dir = os.path.join(tempfile.gettempdir(), "wx859_img_cache")
                                            os.makedirs(cache_dir, exist_ok=True)
                                            file_name = f"wx859_{int(time.time()*1000)}_{len(base64_data)}.{ext}"
                                            temp_file_path = os.path.join(cache_dir, file_name)
                                            with open(temp_file_path, "wb") as f:
                                                f.write(base64.b64decode(base64_data))
                                            image_urls.append(temp_file_path)
                                            logger.info(f"[NanoBananaImage] [{request_id}] 已落地base64图片 -> {temp_file_path} (长度: {len(base64_data)})")
                                        else:
                                            # 没有;base64,分隔符，仍尝试按data URL整体解码
                                            cache_dir = os.path.join(tempfile.gettempdir(), "wx859_img_cache")
                                            os.makedirs(cache_dir, exist_ok=True)
                                            file_name = f"wx859_{int(time.time()*1000)}.bin"
                                            temp_file_path = os.path.join(cache_dir, file_name)
                                            payload = image_url.split(',', 1)[1] if ',' in image_url else image_url
                                            with open(temp_file_path, "wb") as f:
                                                f.write(base64.b64decode(payload))
                                            image_urls.append(temp_file_path)
                                            logger.info(f"[NanoBananaImage] [{request_id}] 已落地base64图片(无显式分隔) -> {temp_file_path}")
                                    except Exception as e:
                                        logger.error(f"[NanoBananaImage] [{request_id}] 处理base64图片URL失败: {e}")
                                else:
                                    # 处理普通HTTP/HTTPS URL
                                    image_urls.append(image_url)
                                    logger.info(f"[NanoBananaImage] [{request_id}] 添加HTTP图片URL: {image_url}")
                
                # 如果没有从新格式找到图片，检查是否需要等待图片生成
                if not image_urls:
                    # 检查API响应中是否有图片生成的提示信息
                    content_lower = content.lower()
                    image_generation_keywords = ['图片', 'image', '生成', 'generate', '创建', 'create', '制作', 'make']
                    has_image_generation_hint = any(keyword in content_lower for keyword in image_generation_keywords)
                    
                    if has_image_generation_hint:
                        logger.info(f"[NanoBananaImage] [{request_id}] 检测到图片生成提示，可能需要等待图片生成完成")
                        # 检查是否有图片ID或其他标识符
                        if 'id' in result:
                            image_id = result['id']
                            logger.info(f"[NanoBananaImage] [{request_id}] 发现图片ID: {image_id}")
                            
                            # 尝试从content中提取base64数据
                            try:
                                # 查找可能的base64数据
                                base64_pattern = r'([A-Za-z0-9+/]{100,}={0,2})'
                                base64_matches = re.findall(base64_pattern, content)
                                
                                for base64_data in base64_matches:
                                    try:
                                        # 尝试解码base64数据
                                        image_bytes = base64.b64decode(base64_data)
                                        # 尝试作为图片打开
                                        pil_image = Image.open(BytesIO(image_bytes))
                                        pil_image.verify()  # 验证图片完整性
                                        
                                        # 如果验证成功，重新打开图片
                                        pil_image = Image.open(BytesIO(image_bytes))
                                        logger.info(f"[NanoBananaImage] [{request_id}] 从content中成功提取base64图片，尺寸: {pil_image.size}, 预览: {self._truncate_base64_log(base64_data)}")
                                        
                                        # 直接落地为文件并加入路径（而非回写data URL）
                                        cache_dir = os.path.join(tempfile.gettempdir(), "wx859_img_cache")
                                        os.makedirs(cache_dir, exist_ok=True)
                                        ext = (pil_image.format or "png").lower()
                                        file_name = f"wx859_{int(time.time()*1000)}_{len(base64_data)}.{ext}"
                                        temp_file_path = os.path.join(cache_dir, file_name)
                                        with open(temp_file_path, "wb") as f:
                                            f.write(image_bytes)
                                        image_urls.append(temp_file_path)
                                        logger.info(f"[NanoBananaImage] [{request_id}] 从content中提取图片并落地 -> {temp_file_path}")
                                        break
                                    except Exception as e:
                                        logger.debug(f"[NanoBananaImage] [{request_id}] base64数据 {self._truncate_base64_log(base64_data)} 不是有效图片: {e}")
                                        continue
                            except Exception as e:
                                logger.debug(f"[NanoBananaImage] [{request_id}] 提取base64数据失败: {e}")
                    
                    # 如果仍然没有找到图片，尝试从content中提取markdown格式的图片URL（向后兼容）
                    if not image_urls:
                        logger.info(f"[NanoBananaImage] [{request_id}] 新格式未找到图片，尝试从content中提取markdown格式")
                        url_matches = list(re.finditer(r'!\[(.*?)\]\((.+?)\)|\[.*?\]\((https?://[^\s\)]+)\)', content))
                        
                        # 处理图片URL
                        for match in url_matches:
                            url = match.group(2) or match.group(3)
                            if url and url.startswith(('http://', 'https://')):  # 验证URL格式
                                image_urls.append(url)
                            else:
                                logger.warning(f"[NanoBananaImage] [{request_id}] 跳过无效的图片URL: {url}")
                
                # 处理文本内容
                cleaned_content = content.strip()
                
                # 如果有图片，从content中移除图片标记，保留纯文本
                if image_urls:
                    # 移除markdown格式的图片标记
                    url_matches = list(re.finditer(r'!\[(.*?)\]\((.+?)\)|\[.*?\]\((https?://[^\s\)]+)\)', content))
                    for match in url_matches:
                        content = content.replace(match.group(0), '')
                    cleaned_content = content.strip()
                
                # 如果有文本内容，将其添加到响应列表
                if cleaned_content:
                    text_responses.append(cleaned_content)
                    logger.info(f"[NanoBananaImage] [{request_id}] 添加文本内容: {cleaned_content[:100]}...")
                else:
                    logger.info(f"[NanoBananaImage] [{request_id}] 未发现文本内容")
            
                # 记录处理结果统计
                end_time = time.time()
                duration = end_time - start_time
                duration_msg = f"图片处理完成! 耗时: {duration:.2f}秒"
                logger.info(f"[NanoBananaImage] [{request_id}] 处理结果:\n文本数量: {len(text_responses)}\n图片数量: {len(image_urls)}\n{duration_msg}")
                            
                # 如果没有任何有效内容，记录警告
                if not text_responses and not image_urls:
                    logger.warning(f"[NanoBananaImage] [{request_id}] API返回了空响应")

            
                # 缓存生成的图片URL
                if user_id and image_urls:
                    self.image_cache[user_id] = {
                    "urls": image_urls,
                    "timestamp": time.time()
                }
                logger.info(f"[NanoBananaImage] [{request_id}] 已缓存用户{user_id}的图片URL，数量: {len(image_urls)}")
            
                # 根据内容类型返回不同的结果
                if image_urls:
                    # 当有图片URL时，只返回图片URL
                    logger.info(f"[NanoBananaImage] [{request_id}] API返回图片消息，共{len(image_urls)}张图片")
                    return {"image_urls": image_urls, "duration_msg": duration_msg}
                elif text_responses:
                    # 检查是否应该等待图片生成
                    content_lower = content.lower()
                    image_generation_keywords = ['图片', 'image', '生成', 'generate', '创建', 'create', '制作', 'make']
                    has_image_generation_hint = any(keyword in content_lower for keyword in image_generation_keywords)
                    
                    if has_image_generation_hint and 'id' in result:
                        # 检测到图片生成提示，可能需要等待
                        logger.warning(f"[NanoBananaImage] [{request_id}] 检测到图片生成提示但未找到图片，可能需要等待或重试")
                        logger.info(f"[NanoBananaImage] [{request_id}] 建议：检查API是否支持异步图片生成，或尝试使用不同的API端点")
                        # 返回特殊标记，表示需要等待图片生成
                        return {"text_responses": text_responses, "duration_msg": duration_msg, "wait_for_image": True, "image_id": result.get('id')}
                    else:
                        # 当没有图片URL时，返回文本内容
                        logger.info(f"[NanoBananaImage] [{request_id}] API返回纯文本消息")
                        return {"text_responses": text_responses, "duration_msg": duration_msg}
                else:
                    # 没有任何有效内容
                    logger.info(f"[NanoBananaImage] [{request_id}] API返回空响应")
                    return {"text_responses": ["API返回内容为空"], "duration_msg": duration_msg}
            
            except requests.Timeout:
                execution_time = time.time() - start_time
                logger.error(f"[NanoBananaImage] API请求超时，耗时: {execution_time:.2f}秒")
                raise Exception("API请求超时，请稍后重试")
        except requests.RequestException as e:
            execution_time = time.time() - start_time
            logger.error(f"[NanoBananaImage] API请求失败，耗时: {execution_time:.2f}秒，错误: {e}")
            if isinstance(e, requests.ConnectionError):
                raise Exception("无法连接到API服务器，请检查网络连接")
            elif isinstance(e, requests.TooManyRedirects):
                raise Exception("API请求重定向次数过多")
            else:
                raise Exception(f"API请求失败: {str(e)}")
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[NanoBananaImage] 图片生成失败，耗时: {execution_time:.2f}秒，错误: {e}")
            raise

    def _get_image_data(self, context) -> Optional[bytes]:
        """
        统一的图片数据获取方法，支持多种数据源
        
        Args:
            context: 消息上下文对象
            
        Returns:
            bytes: 图片二进制数据，获取失败则返回None
        """
        try:
            logger.debug(f"[NanoBananaImage] 开始获取图片数据，context类型: {context.type}")
            
            # 统一的文件读取函数
            def read_file(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                        logger.debug(f"[NanoBananaImage] 成功读取文件: {file_path}, 大小: {len(data)} 字节")
                        return data
                except Exception as e:
                    logger.error(f"[NanoBananaImage] 读取文件失败 {file_path}: {e}")
                    return None
            
            # 1. 从context.content获取
            if hasattr(context, 'content'):
                logger.debug(f"[NanoBananaImage] 检查context.content: {type(context.content)}")
                if isinstance(context.content, bytes):
                    logger.debug(f"[NanoBananaImage] 从context.content获取到二进制数据，大小: {len(context.content)} 字节")
                    return context.content
                elif isinstance(context.content, str):
                    if os.path.isfile(context.content):
                        data = read_file(context.content)
                        if data:
                            return data
                    elif context.content.startswith(('http://', 'https://')):
                        try:
                            logger.debug(f"[NanoBananaImage] 尝试从URL下载图片: {context.content}")
                            response = requests.get(context.content, timeout=10)
                            if response.status_code == 200:
                                data = response.content
                                if data and len(data) > 1000:
                                    logger.debug(f"[NanoBananaImage] 从URL下载图片成功，大小: {len(data)} 字节")
                                    return data
                        except Exception as e:
                            logger.error(f"[NanoBananaImage] 从URL下载图片失败: {e}")
                    else:
                        logger.debug(f"[NanoBananaImage] 文件不存在或非URL: {context.content}")
            
            # 2. 从context.kwargs获取
            if hasattr(context, 'kwargs'):
                logger.debug(f"[NanoBananaImage] 检查context.kwargs: {context.kwargs.keys() if context.kwargs else 'None'}")
                
                # 2.1 检查image_base64
                image_base64 = context.kwargs.get('image_base64')
                if image_base64:
                    try:
                        image_data = base64.b64decode(image_base64)
                        logger.debug(f"[NanoBananaImage] 从image_base64解码图片数据成功，大小: {len(image_data)} 字节")
                        return image_data
                    except Exception as e:
                        logger.error(f"[NanoBananaImage] Base64解码失败: {e}")
                
                # 2.2 获取消息对象
                msg = context.kwargs.get('msg')
                if msg:
                    logger.debug(f"[NanoBananaImage] 检查msg对象: {type(msg)}")
                    
                    # 检查file_path属性
                    if hasattr(msg, 'file_path') and msg.file_path:
                        logger.debug(f"[NanoBananaImage] 发现file_path: {msg.file_path}")
                        data = read_file(msg.file_path)
                        if data:
                            return data
                    
                    # 检查download_image方法
                    if hasattr(msg, 'download_image') and callable(getattr(msg, 'download_image')):
                        retry_count = 3
                        retry_delay = 1  # 秒
                        
                        for attempt in range(retry_count):
                            try:
                                logger.debug(f"[NanoBananaImage] 尝试使用download_image方法 (尝试 {attempt + 1}/{retry_count})")
                                image_data = msg.download_image()
                                if image_data and len(image_data) > 1000:
                                    logger.debug(f"[NanoBananaImage] 通过download_image获取图片成功，大小: {len(image_data)} 字节")
                                    return image_data
                                else:
                                    logger.warning(f"[NanoBananaImage] download_image返回的数据无效或过小: {len(image_data) if image_data else 0} 字节")
                            except Exception as e:
                                logger.error(f"[NanoBananaImage] download_image调用失败 (尝试 {attempt + 1}/{retry_count}): {e}")
                                if attempt < retry_count - 1:
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # 指数退避
                    
                    # 检查img属性
                    if hasattr(msg, 'img') and msg.img:
                        logger.debug(f"[NanoBananaImage] 检查msg.img属性")
                        image_data = msg.img
                        if image_data and len(image_data) > 1000:
                            logger.debug(f"[NanoBananaImage] 从msg.img获取图片成功，大小: {len(image_data)} 字节")
                            return image_data
                        else:
                            logger.warning(f"[NanoBananaImage] msg.img数据无效或过小: {len(image_data) if image_data else 0} 字节")
                    
                    # 检查msg_data属性
                    if hasattr(msg, 'msg_data'):
                        try:
                            msg_data = msg.msg_data
                            if isinstance(msg_data, dict) and 'image' in msg_data:
                                image_data = msg_data['image']
                                if image_data and len(image_data) > 1000:
                                    logger.debug(f"[NanoBananaImage] 从msg_data['image']获取到图片数据，大小: {len(image_data)} 字节")
                                    return image_data
                            elif isinstance(msg_data, bytes):
                                logger.debug(f"[NanoBananaImage] 从msg_data(bytes)获取到图片数据，大小: {len(msg_data)} 字节")
                                return msg_data
                        except Exception as e:
                            logger.error(f"[NanoBananaImage] 获取msg_data失败: {e}")
                    
                    # 检查_rawmsg属性
                    if hasattr(msg, '_rawmsg') and isinstance(msg._rawmsg, dict):
                        try:
                            rawmsg = msg._rawmsg
                            logger.debug(f"[NanoBananaImage] 获取到_rawmsg: {type(rawmsg)}")
                            
                            if 'file' in rawmsg and rawmsg['file']:
                                file_path = rawmsg['file']
                                logger.debug(f"[NanoBananaImage] 从_rawmsg获取到文件路径: {file_path}")
                                data = read_file(file_path)
                                if data:
                                    return data
                        except Exception as e:
                            logger.error(f"[NanoBananaImage] 处理_rawmsg失败: {e}")
                    
                    # 检查image_url属性
                    if hasattr(msg, 'image_url') and msg.image_url:
                        try:
                            image_url = msg.image_url
                            logger.debug(f"[NanoBananaImage] 从msg.image_url获取图片URL: {image_url}")
                            response = requests.get(image_url, timeout=10)
                            if response.status_code == 200:
                                data = response.content
                                if data and len(data) > 1000:
                                    logger.debug(f"[NanoBananaImage] 从image_url下载图片成功，大小: {len(data)} 字节")
                                    return data
                        except Exception as e:
                            logger.error(f"[NanoBananaImage] 从image_url下载图片失败: {e}")
                    
                    # 如果文件未下载，尝试下载
                    if hasattr(msg, '_prepare_fn') and hasattr(msg, '_prepared') and not msg._prepared:
                        logger.debug("[NanoBananaImage] 尝试调用msg._prepare_fn()下载图片...")
                        try:
                            msg._prepare_fn()
                            msg._prepared = True
                            time.sleep(1)  # 等待文件准备完成
                            
                            # 再次尝试获取内容
                            if hasattr(msg, 'content'):
                                if isinstance(msg.content, bytes):
                                    return msg.content
                                elif isinstance(msg.content, str) and os.path.isfile(msg.content):
                                    data = read_file(msg.content)
                                    if data:
                                        return data
                        except Exception as e:
                            logger.error(f"[NanoBananaImage] 调用_prepare_fn下载图片失败: {e}")
            
            logger.error("[NanoBananaImage] 无法获取图片数据，已尝试所有可能的获取方式")
            return None
            
        except Exception as e:
            logger.error(f"[NanoBananaImage] 获取图片数据失败: {e}")
            return None

    def _materialize_data_url(self, image_url: str) -> str:
        """
        如果是 data:image/...;base64,xxx 的图片URL，则解码并落地为临时文件，返回本地文件路径；
        否则原样返回。避免将 data URL 当作网络URL发送导致适配器错误。
        """
        try:
            if isinstance(image_url, str) and image_url.startswith('data:image/') and ';base64,' in image_url:
                header, data_part = image_url.split(';base64,', 1)
                mime_type = header.replace('data:', '')
                # 扩展名推断
                ext = '.png'
                if 'jpeg' in mime_type or 'jpg' in mime_type:
                    ext = '.jpg'
                elif 'webp' in mime_type:
                    ext = '.webp'
                elif 'png' in mime_type:
                    ext = '.png'
                # 解码并落地
                try:
                    decoded = base64.b64decode(data_part, validate=False)
                except Exception:
                    decoded = base64.b64decode(data_part + '==')  # 容错填充
                cache_dir = os.path.join(tempfile.gettempdir(), "wx859_img_cache")
                os.makedirs(cache_dir, exist_ok=True)
                filename = f"nb_{int(time.time()*1000)}_{len(data_part)}{ext}"
                file_path = os.path.join(cache_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(decoded)
                logger.info(f"[NanoBananaImage] 已将data URL落地为文件: {file_path}, 大小: {len(decoded)} 字节, 类型: {mime_type}")
                return file_path
        except Exception as e:
            logger.error(f"[NanoBananaImage] data URL落地失败: {e}")
        return image_url

    def _handle_referenced_image_to_image(self, e_context: EventContext, content: str, referenced_image_path: str):
        """
        处理引用图片的“N改图”命令：直接使用被引用的图片进行编辑，无需再次上传
        """
        try:
            user_id = e_context["context"].get("session_id") or e_context["context"].get("from_user_id")
            # 提取提示词（去掉前缀）
            prompt = content
            for prefix in self.reference_edit_commands:
                if prompt.startswith(prefix):
                    prompt = prompt[len(prefix):].strip()
                    break
            if not prompt:
                e_context["reply"] = Reply(ReplyType.TEXT, f"请在“{self.reference_edit_commands[0]}”后输入描述，例如：{self.reference_edit_commands[0]} 把背景换成夜景")
                return

            # 读取引用图片数据
            image_data = self._get_referenced_image_data(referenced_image_path)
            if not image_data:
                e_context["reply"] = Reply(ReplyType.TEXT, "未能读取被引用的图片，请重试或直接上传图片。")
                return

            # 验证图片（与上传流程一致）
            try:
                img = Image.open(BytesIO(image_data))
                img_format = img.format.lower()
                img_size = img.size
                allowed_formats = ['jpeg', 'jpg', 'png', 'webp']
                if img_format not in allowed_formats:
                    e_context["reply"] = Reply(ReplyType.TEXT, f"不支持的图片格式: {img_format}，仅支持: {', '.join(allowed_formats)}")
                    return
                max_dimension = 4096
                if img_size[0] > max_dimension or img_size[1] > max_dimension:
                    e_context["reply"] = Reply(ReplyType.TEXT, f"图片尺寸过大: {img_size[0]}x{img_size[1]}，最大允许: {max_dimension}x{max_dimension}")
                    return
            except Exception as ve:
                logger.error(f"[NanoBananaImage] 引用图片验证失败: {ve}")
                e_context["reply"] = Reply(ReplyType.TEXT, "无法处理引用的图片，请确保图片有效。")
                return

            # 调用生成接口进行改图（明确标注为N改图）
            result = self.generate_image(prompt, True, image_data, user_id, e_context, op_mode="N改图")
            if not result:
                e_context["reply"] = Reply(ReplyType.ERROR, "图片编辑失败，请稍后重试")
                return

            # 如需等待生成
            if result.get("wait_for_image"):
                e_context["reply"] = Reply(ReplyType.TEXT, "图片正在生成中，请稍等片刻后重试，或检查API配置是否正确。")
                return

            # 发送图片优先
            sent_any = False
            if "image_urls" in result and result["image_urls"]:
                for image_url in result["image_urls"]:
                    try:
                        image_url = self._materialize_data_url(image_url)
                        reply_type = ReplyType.IMAGE if os.path.isfile(image_url) else ReplyType.IMAGE_URL
                        image_reply = Reply(reply_type, image_url)
                        e_context["channel"].send(image_reply, e_context["context"])
                        sent_any = True
                        break
                    except Exception as se:
                        logger.error(f"[NanoBananaImage] 发送引用改图图片失败: {se}")
                        continue

            # 发送文本
            if "text_responses" in result and result["text_responses"]:
                for text in result["text_responses"]:
                    if text:
                        e_context["channel"].send(Reply(ReplyType.TEXT, text), e_context["context"])

            if sent_any and "duration_msg" in result:
                e_context["channel"].send(Reply(ReplyType.TEXT, result["duration_msg"]), e_context["context"])

        except Exception as e:
            logger.error(f"[NanoBananaImage] 处理引用图片改图命令失败: {e}")
            e_context["reply"] = Reply(ReplyType.ERROR, f"引用图片改图失败: {str(e)}")

    def _get_referenced_image_data(self, referenced_image_path: str) -> Optional[bytes]:
        """
        获取引用图片的数据，支持本地路径/HTTP(S) URL/临时目录回退
        """
        try:
            if not referenced_image_path:
                return None
            path = str(referenced_image_path)
            logger.info(f"[NanoBananaImage] 获取引用图片数据: {path}")

            # 1) 本地文件
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read()

            # 2) URL 下载
            if path.startswith(("http://", "https://")):
                resp = self.session.get(path, timeout=30)
                resp.raise_for_status()
                return resp.content if resp.content else None

            # 3) 尝试在系统临时缓存目录 wx859_img_cache 查找
            cache_dir = os.path.join(tempfile.gettempdir(), "wx859_img_cache")
            candidate = os.path.join(cache_dir, os.path.basename(path))
            if os.path.isfile(candidate):
                with open(candidate, "rb") as f:
                    return f.read()

            # 4) 再尝试直接在系统临时目录
            candidate2 = os.path.join(tempfile.gettempdir(), os.path.basename(path))
            if os.path.isfile(candidate2):
                with open(candidate2, "rb") as f:
                    return f.read()

            logger.error(f"[NanoBananaImage] 无法找到引用图片: {path}")
            return None
        except Exception as e:
            logger.error(f"[NanoBananaImage] 读取引用图片失败: {e}")
            return None

    def _truncate_base64_log(self, base64_str, max_length=50):
        """
        截断base64字符串用于日志记录，避免刷屏
        """
        if not base64_str:
            return ""
        
        # 如果是data URL格式，提取更多有用信息
        if base64_str.startswith('data:image/') and ';base64,' in base64_str:
            header, data_part = base64_str.split(';base64,', 1)
            mime_type = header.replace('data:', '')
            data_size_kb = len(data_part) * 3 / 4 / 1024  # base64解码后的大概大小
            if len(data_part) <= max_length:
                return f"{header};base64,{data_part}"
            else:
                return f"{header};base64,{data_part[:max_length]}... (数据长度: {len(data_part)}, 约{data_size_kb:.1f}KB)"
        
        # 普通base64字符串处理
        if len(base64_str) <= max_length:
            return base64_str
        return f"{base64_str[:max_length]}... (总长度: {len(base64_str)})"

    def _safe_json_dumps(self, obj, ensure_ascii=False, indent=2):
        """
        安全地序列化JSON，自动截断base64字符串避免刷屏
        """
        def _process_value(value):
            if isinstance(value, str) and len(value) > 100 and self._is_likely_base64(value):
                return self._truncate_base64_log(value)
            elif isinstance(value, dict):
                return {k: _process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_process_value(v) for v in value]
            else:
                return value
        
        processed_obj = _process_value(obj)
        return json.dumps(processed_obj, ensure_ascii=ensure_ascii, indent=indent)

    def _is_likely_base64(self, text):
        """
        判断字符串是否可能是base64编码
        """
        if not text:
            return False
        
        # 检查是否是data URL格式
        if text.startswith('data:image/') and ';base64,' in text:
            return True
        
        # 检查是否包含base64特征字符
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        text_chars = set(text)
        
        # 如果字符串主要由base64字符组成且长度较长，可能是base64
        if len(text) > 100:  # 降低长度阈值，提高检测敏感度
            non_base64_chars = text_chars - base64_chars
            # 如果非base64字符少于5%，认为是base64
            return len(non_base64_chars) < len(text_chars) * 0.05
        
        return False

    def get_help_text(self, **kwargs):
        help_text = "NanoBananaImage绘图插件使用指南：\n"
        help_text += f"1. 调用NanoBanana生成图片：发送 {self.commands[0]} + 描述，例如：{self.commands[0]} 用一张信息图表详细解释牛顿的棱镜实验\n"
        help_text += f"2. 编辑用户上传的图片：发送 {self.reference_edit_commands[0]} + 描述，然后上传图片\n"
        help_text += f"3. 融合两张图片：发送 {self.merge_commands[0]} + 描述，然后依次上传两张图片\n"
        return help_text