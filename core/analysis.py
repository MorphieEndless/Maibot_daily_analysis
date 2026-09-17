"""
聊天分析服务

把聊天记录交给 LLM 做各类分析（话题、群友称号、金句、炫压抑评级、个人画像等）。
所有 LLM 调用通过宿主注入的 ``ctx.llm`` 能力完成；纯数据处理保持为静态方法。

消息字典遵循插件运行时的扁平结构（由 plugin.py 的归一化层提供）：
    {
        "user_id": str, "user_nickname": str, "user_cardname": str,
        "processed_plain_text": str, "time": float,
        "is_command": bool, "is_notify": bool,
    }
"""

import re
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import Counter

from .constants import AnalysisConfig


# LLM 各任务的输出 token 上限。
# 注意：SDK→宿主的能力调用 RPC 固定 30 秒超时，单次 LLM 必须在 30 秒内返回，
# 因此默认使用快速模型(utils/flash)并适度限制输出长度，避免超时。
_SUMMARY_MAX_TOKENS = 1200
_JSON_MAX_TOKENS = 2500
# 多用户 JSON（群友称号/炫压抑评级）输出较长，但要兼顾 30 秒 RPC 超时，控制在 2500
_MULTI_USER_JSON_MAX_TOKENS = 2500

# LLM 输入消息上限：取最近 N 条参与总结/话题/金句，避免超大群 prompt 过长拖慢生成
_MAX_INPUT_MESSAGES = 400

# 并发 LLM 调用上限。设为 2：兼顾速度与"上游串行时排队不耗尽超时预算"。
_LLM_MAX_CONCURRENCY = 2

# 单次 LLM 调用的最长等待（秒），到点放弃该次分析项。注意：宿主对插件的单次能力调用
# 约有 30 秒 RPC 硬上限，设大于 30 通常无额外效果；此值仅作客户端侧的等待上限。
_DEFAULT_CALL_TIMEOUT_S = 60

# 默认模型任务：utils 对应快速非思考模型，速度快、稳定
_DEFAULT_MODEL_TASK = "utils"


class AnalysisService:
    """聊天记录分析服务（绑定插件 ctx，统一走 ctx.llm / ctx.logger）"""

    # Emoji 正则（精确匹配，避免误伤中文字符）
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # symbols
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002600-\U000026FF"  # misc symbols
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0001F000-\U0001F02F"  # mahjong tiles
        "\U0001F0A0-\U0001F0FF"  # playing cards
        "]+",
        flags=re.UNICODE,
    )

    def __init__(
        self,
        ctx: Any,
        model: str = _DEFAULT_MODEL_TASK,
        call_timeout_s: int = _DEFAULT_CALL_TIMEOUT_S,
    ):
        self.ctx = ctx
        self.logger = ctx.logger
        # 模型任务名（可由插件配置覆盖）。默认 utils=快速模型，确保 30 秒内返回
        self.model = model or _DEFAULT_MODEL_TASK
        # 单次 LLM 调用的客户端等待上限（秒），可由插件配置覆盖
        self.call_timeout_s = max(5, int(call_timeout_s or _DEFAULT_CALL_TIMEOUT_S))
        # 限制并发 LLM 调用数：每个能力调用有约 30 秒 RPC 硬超时，若上游串行处理，
        # 一次放出过多调用会让排队靠后的调用把等待时间算进自己的超时预算而被掐断。
        # 信号量在"真正发起 ctx.llm.generate 之前"获取，确保每次调用的 30 秒计时
        # 从有空闲槽位时才开始，避免排队耗尽预算。
        self._llm_semaphore = asyncio.Semaphore(_LLM_MAX_CONCURRENCY)

    # ==================== LLM 调用封装 ====================\n
    async def _llm(
        self,
        prompt: str,
        *,
        request_type: str,
        max_tokens: int = _JSON_MAX_TOKENS,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """调用宿主 LLM 能力，成功返回文本，失败返回 None"""
        try:
            async with self._llm_semaphore:
                try:
                    # 适配 MaiBot 1.2.x / maibot_sdk 2.8+（显式传递 task_name，model 设为空避免被宿主误认为具体模型名）
                    generate_coro = self.ctx.llm.generate(
                        prompt,
                        task_name=self.model,
                        model="",
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except TypeError:
                    # 兼容早期旧版 SDK 签名
                    generate_coro = self.ctx.llm.generate(
                        prompt,
                        model=self.model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                result = await asyncio.wait_for(
                    generate_coro,
                    timeout=self.call_timeout_s,
                )
        except asyncio.TimeoutError:
            self.logger.warning(f"LLM 调用超时 ({request_type}, >{self.call_timeout_s}s)")
            return None
        except Exception as e:
            self.logger.error(f"LLM 调用异常 ({request_type}): {e}", exc_info=True)
            return None

        if not isinstance(result, dict) or not result.get("success", False):
            err = result.get("error") if isinstance(result, dict) else result
            self.logger.error(f"LLM 生成失败 ({request_type}): {err}")
            return None

        # 兼容 SDK 不同版本的字段名：优先 response，兜底 content / text
        content = (
            result.get("response")
            or result.get("content")
            or result.get("text")
            or ""
        )
        return str(content).strip() if content else None

    # ==================== 纯数据统计（无 LLM，秒级完成） ====================

    @staticmethod
    def calculate_chat_time_range(messages: List[dict]) -> str:
        """计算最早消息到最晚消息的时间跨度（如 '08:30 - 23:45'）"""
        if not messages:
            return "00:00 - 23:59"
        valid_times = [m["time"] for m in messages if isinstance(m.get("time"), (int, float)) and m["time"] > 0]
        if not valid_times:
            return "00:00 - 23:59"
        earliest = datetime.fromtimestamp(min(valid_times)).strftime("%H:%M")
        latest = datetime.fromtimestamp(max(valid_times)).strftime("%H:%M")
        return f"{earliest} - {latest}"

    @staticmethod
    def calculate_peak_time(messages: List[dict]) -> str:
        """计算发言最活跃的时段（一小时区间，如 '14:00 - 15:00'）"""
        if not messages:
            return "12:00 - 13:00"
        hours = []
        for m in messages:
            t = m.get("time")
            if isinstance(t, (int, float)) and t > 0:
                hours.append(datetime.fromtimestamp(t).hour)
        if not hours:
            return "12:00 - 13:00"
        peak_hour = Counter(hours).most_common(1)[0][0]
        return f"{peak_hour:02d}:00 - {(peak_hour + 1) % 24:02d}:00"

    @staticmethod
    def calculate_hourly_distribution(messages: List[dict]) -> List[int]:
        """计算 24 小时每小时的消息数"""
        counts = [0] * 24
        for m in messages:
            t = m.get("time")
            if isinstance(t, (int, float)) and t > 0:
                counts[datetime.fromtimestamp(t).hour] += 1
        return counts

    @staticmethod
    def calculate_emoji_count(messages: List[dict]) -> int:
        """统计消息中真实的 Emoji 数量"""
        total = 0
        for m in messages:
            text = m.get("processed_plain_text", "")
            if text:
                total += len(AnalysisService.EMOJI_PATTERN.findall(text))
        return total

    @staticmethod
    def calculate_total_words(messages: List[dict]) -> int:
        """统计所有消息的实际总字符数"""
        return sum(len(m.get("processed_plain_text", "")) for m in messages)

    @staticmethod
    def find_active_users(messages: List[dict], min_count: int, max_users: int) -> List[Tuple[str, str, int]]:
        """找出活跃用户 (user_id, display_name, msg_count)

        按发言数降序，过滤少于 min_count 条的，最多取 max_users 人。
        display_name 优先取 user_cardname，其次 user_nickname，最后 user_id。
        """
        user_counts = Counter(m["user_id"] for m in messages if m.get("user_id"))
        # 记录每个 user_id 最新的显示昵称
        names: Dict[str, str] = {}
        for m in messages:
            uid = m.get("user_id")
            if uid and uid not in names:
                name = m.get("user_cardname") or m.get("user_nickname") or uid
                names[uid] = name

        results = []
        for uid, count in user_counts.most_common():
            if count < min_count:
                break
            results.append((uid, names.get(uid, uid), count))
            if len(results) >= max_users:
                break
        return results

    # ==================== 提示词准备辅助 ====================

    @staticmethod
    def _prepare_chat_lines(messages: List[dict], max_lines: int = _MAX_INPUT_MESSAGES) -> str:
        """格式化聊天记录为适合 LLM 读取的文本行

        取最近 max_lines 条，过滤命令/通知和纯空白。
        格式：`昵称: 消息内容`
        """
        recent = messages[-max_lines:] if len(messages) > max_lines else messages
        lines = []
        for m in recent:
            if m.get("is_command") or m.get("is_notify"):
                continue
            text = (m.get("processed_plain_text") or "").strip()
            if not text:
                continue
            name = m.get("user_cardname") or m.get("user_nickname") or m.get("user_id", "某人")
            lines.append(f"{name}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _safe_parse_json(text: Optional[str]) -> Optional[Any]:
        """尝试从 LLM 输出中提取并解析 JSON（支持 markdown 围栏）"""
        if not text:
            return None
        # 去除开头的 ```json 与结尾的 ```
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # 去掉第一行（```json 或 ```）
            first = lines[0].strip()
            if first.startswith("```"):
                lines = lines[1:]
            # 去掉最后一行若为 ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # 优先全量解析
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 尝试正则提取最外层 [ ... ] 或 { ... }
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        return None

    # ==================== 业务分析项（LLM 驱动） ====================

    async def analyze_group_summary(self, messages: List[dict], total_count: int) -> Optional[str]:
        """群聊故事型总结（自然段落，像朋友闲聊般讲述）"""
        chat_text = self._prepare_chat_lines(messages, max_lines=_MAX_INPUT_MESSAGES)
        if not chat_text:
            return None

        prompt = f"""你是这群里一个幽默敏锐的朋友。请读读今天这 {total_count} 条聊天，写一篇像聊天一样的群聊日常总结。

【要求】
1. 像跟朋友讲故事一样，自然、口语化、有梗，不要官话、不要公文腔
2. 抓今天的核心事件、槽点、争议、高光时刻
3. 可以点出关键人物的精彩表现（用他们发言时的昵称）
4. 篇幅 200~350 字，分为 2-3 个小段落，纯文本，不要包含 Markdown 标题符号

【聊天记录】
{chat_text}
"""
        return await self._llm(
            prompt,
            request_type="plugin.chat_summary",
            max_tokens=_SUMMARY_MAX_TOKENS,
            temperature=0.7,
        )

    async def extract_topics(self, messages: List[dict]) -> List[dict]:
        """提取今日热门话题（2~4 个，带热度、关键词、参与者）"""
        chat_text = self._prepare_chat_lines(messages, max_lines=_MAX_INPUT_MESSAGES)
        if not chat_text:
            return []

        prompt = f"""从以下群聊记录中提炼出 2 到 4 个今天讨论最充分的热门话题。

【输出要求】严格输出 JSON 数组，不要任何多余文字、不要 markdown 说明。每个元素格式如下：
[
  {{
    "title": "话题简短标题（8字以内，口语生动）",
    "desc": "一两句话概括群友在讨论什么、有什么亮点或分歧（30字以内）",
    "hot": 95,
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "participants": ["主要参与者昵称1", "主要参与者昵称2"]
  }}
]
hot 为 50-100 的整数，表示该话题在群内的热烈程度。participants 最多 4 人。

【聊天记录】
{chat_text}
"""
        res = await self._llm(
            prompt,
            request_type="plugin.chat_topics",
            max_tokens=_JSON_MAX_TOKENS,
            temperature=0.5,
        )
        parsed = self._safe_parse_json(res)
        if isinstance(parsed, list):
            valid = []
            for item in parsed:
                if isinstance(item, dict) and "title" in item:
                    valid.append({
                        "title": str(item.get("title", ""))[:12],
                        "desc": str(item.get("desc", ""))[:60],
                        "hot": int(item.get("hot", 80)),
                        "keywords": [str(k)[:8] for k in item.get("keywords", [])][:4],
                        "participants": [str(p)[:12] for p in item.get("participants", [])][:4],
                    })
            return valid[:4]
        return []

    async def generate_character_sketch(
        self,
        messages: List[dict],
        active_users: List[Tuple[str, str, int]],
    ) -> List[dict]:
        """为活跃群友生成称号 + MBTI + 简评"""
        if not active_users:
            return []

        # 收集这几位用户的发言样本（每人最多 15 条代表性发言）
        user_samples: Dict[str, List[str]] = {uid: [] for uid, _, _ in active_users}
        for m in messages:
            uid = m.get("user_id")
            if uid in user_samples and len(user_samples[uid]) < 15:
                text = (m.get("processed_plain_text") or "").strip()
                if text and not m.get("is_command"):
                    user_samples[uid].append(text)

        # 拼接用户画像 prompt
        profiles_text = []
        for uid, name, count in active_users:
            samples = " / ".join(user_samples.get(uid, [])[:10]) or "(无典型发言)"
            profiles_text.append(f"- 用户：{name} (发言{count}条)\n  发言摘录：{samples}")

        prompt = f"""根据以下几位活跃群友今天的发言特点，为每人生成一个独特的个性化称号、推测一个 MBTI 类型，并给出一句风趣入木的点评。

【输出要求】严格输出 JSON 数组，不要任何多余文字。格式：
[
  {{
    "user_name": "用户昵称（必须严格对齐输入里的名字）",
    "title": "个性称号（4-8字，如『深夜哲思大师』『人形复读机』『摸鱼艺术总监』）",
    "mbti": "INTJ",
    "reason": "一句话点评，幽默切中要害，30字以内"
  }}
]
MBTI 必须是标准的 16 种类型之一（四个大写英文字母）。

【候选群友】
{"\n".join(profiles_text)}
"""
        res = await self._llm(
            prompt,
            request_type="plugin.character_sketch",
            max_tokens=_MULTI_USER_JSON_MAX_TOKENS,
            temperature=0.7,
        )
        parsed = self._safe_parse_json(res)
        results = []
        if isinstance(parsed, list):
            # 建名字映射
            name_to_uid = {name: uid for uid, name, _ in active_users}
            for item in parsed:
                if isinstance(item, dict) and "user_name" in item:
                    name = str(item.get("user_name", ""))
                    uid = name_to_uid.get(name, "")
                    results.append({
                        "user_id": uid,
                        "user_name": name,
                        "title": str(item.get("title", "群聊观察员"))[:12],
                        "mbti": str(item.get("mbti", "ENFP")).upper()[:4],
                        "reason": str(item.get("reason", ""))[:50],
                    })
        return results

    async def extract_quotes(self, messages: List[dict]) -> List[dict]:
        """提取群聊金句（语出惊人）2~4 条"""
        chat_text = self._prepare_chat_lines(messages, max_lines=_MAX_INPUT_MESSAGES)
        if not chat_text:
            return []

        prompt = f"""从以下聊天记录中挑选 2 到 4 句最具幽默感、哲理感或语出惊人的金句摘录（群聊高光时刻）。

【要求】
1. 必须是聊天记录中真实说过的话，不可凭空编造
2. 原话长度最好在 5-60 字之间，有爆点或有深度
3. 严格输出 JSON 数组，格式如下：
[
  {{
    "quote": "原话摘录",
    "user_name": "说话者的昵称",
    "reason": "推荐理由（风趣简短，20字以内）"
  }}
]

【聊天记录】
{chat_text}
"""
        res = await self._llm(
            prompt,
            request_type="plugin.extract_quotes",
            max_tokens=_JSON_MAX_TOKENS,
            temperature=0.6,
        )
        parsed = self._safe_parse_json(res)
        valid = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "quote" in item:
                    valid.append({
                        "quote": str(item.get("quote", "")).strip()[:100],
                        "user_name": str(item.get("user_name", "群友"))[:16],
                        "reason": str(item.get("reason", ""))[:40],
                    })
        return valid[:4]

    async def analyze_depression(
        self,
        messages: List[dict],
        active_users: List[Tuple[str, str, int]],
        max_display: int = AnalysisConfig.MAX_DEPRESSION_DISPLAY,
        show_bottom: bool = True,
    ) -> List[dict]:
        """炫压抑评级（娱乐向：根据发言风格分析压抑指数）"""
        if not active_users:
            return []

        # 抽取各用户发言片段
        user_samples: Dict[str, List[str]] = {uid: [] for uid, _, _ in active_users}
        for m in messages:
            uid = m.get("user_id")
            if uid in user_samples and len(user_samples[uid]) < 12:
                text = (m.get("processed_plain_text") or "").strip()
                if text and not m.get("is_command"):
                    user_samples[uid].append(text)

        profiles = []
        for uid, name, _ in active_users:
            samples = " / ".join(user_samples.get(uid, [])[:8]) or "(无典型发言)"
            profiles.append(f"- {name}: {samples}")

        prompt = f"""根据以下群友今天的发言风格与文字情绪，给每位群友做一个娱乐向的「压抑/emo 指数」评估（纯属群聊娱乐，调侃风格）。

【输出要求】严格输出 JSON 数组，格式：
[
  {{
    "user_name": "群友昵称（与输入严格对齐）",
    "score": 78,
    "rank": "评级称号（如：轻度疲惫 / 现充阳光 / 灵魂出窍 / 重度抑郁爆发）",
    "desc": "一句文言风或幽默调侃短评（25字以内）"
  }}
]
score 范围 0 到 150。越疲惫、打工人怨气、熬夜发疯、自嘲的，分数越高；越阳光开朗、乐呵水群的分数越低。

【待评群友】
{"\n".join(profiles)}
"""
        res = await self._llm(
            prompt,
            request_type="plugin.depression_analysis",
            max_tokens=_MULTI_USER_JSON_MAX_TOKENS,
            temperature=0.7,
        )
        parsed = self._safe_parse_json(res)
        items = []
        if isinstance(parsed, list):
            name_to_uid = {name: uid for uid, name, _ in active_users}
            for it in parsed:
                if isinstance(it, dict) and "user_name" in it:
                    name = str(it.get("user_name", ""))
                    score = int(it.get("score", 50))
                    items.append({
                        "user_id": name_to_uid.get(name, ""),
                        "user_name": name,
                        "score": score,
                        "rank": str(it.get("rank", "平静自然"))[:12],
                        "desc": str(it.get("desc", ""))[:40],
                    })

        # 按 score 降序排序
        items.sort(key=lambda x: x["score"], reverse=True)
        if not items:
            return []

        # 根据配置决定截取哪些展示
        total_avail = len(items)
        if total_avail <= max_display or not show_bottom:
            return items[:max_display]

        # 开启展示倒数：前 top_k + 后 bottom_k
        top_k = (max_display + 1) // 2
        bottom_k = max_display // 2
        selected = items[:top_k] + items[-bottom_k:]
        # 去重保持顺序
        seen = set()
        final_list = []
        for x in selected:
            if x["user_name"] not in seen:
                seen.add(x["user_name"])
                final_list.append(x)
        return final_list

    # ==================== 个人总结专用分析 ====================

    async def analyze_user_summary(
        self,
        user_messages: List[dict],
        user_name: str,
        time_range: str,
    ) -> Optional[dict]:
        """个人总结分析：故事段落 + 个性称号 + MBTI + 压抑评级 + 个人金句"""
        chat_text = self._prepare_chat_lines(user_messages, max_lines=200)
        if not chat_text:
            return None

        prompt = f"""你是观察这位群友的幽默朋友。请阅读用户「{user_name}」{time_range}的所有发言，生成一份个人总结报告。

【输出要求】严格输出单个 JSON 对象，不要多余文字：
{{
  "summary": "以第二人称『你』开头的一段风趣点评，回顾 TA 今天主要聊了什么、表现出怎样的性格或情绪（120-220字，像熟人说话，纯文本无Markdown标题）",
  "title": "专属称号（4-8字，如『熬夜修仙总工程师』）",
  "mbti": "INTP",
  "reason": "称号与MBTI的简要理由（25字以内）",
  "depression_score": 65,
  "depression_rank": "轻度emo",
  "depression_desc": "一句文言或戏谑点评（20字以内）",
  "quote": "TA 今天最精彩的一句发言（必须来自下面记录）",
  "quote_reason": "金句理由（15字以内）"
}}

【{user_name} 的发言记录】
{chat_text}
"""
        res = await self._llm(
            prompt,
            request_type="plugin.user_summary",
            max_tokens=_JSON_MAX_TOKENS,
            temperature=0.7,
        )
        parsed = self._safe_parse_json(res)
        if isinstance(parsed, dict):
            return {
                "summary": str(parsed.get("summary", "")).strip(),
                "title": str(parsed.get("title", "群聊旅人"))[:12],
                "mbti": str(parsed.get("mbti", "ENFP")).upper()[:4],
                "reason": str(parsed.get("reason", ""))[:40],
                "depression_score": int(parsed.get("depression_score", 50)),
                "depression_rank": str(parsed.get("depression_rank", "平静"))[:10],
                "depression_desc": str(parsed.get("depression_desc", ""))[:40],
                "quote": str(parsed.get("quote", "")).strip()[:80],
                "quote_reason": str(parsed.get("quote_reason", ""))[:30],
            }
        return None
