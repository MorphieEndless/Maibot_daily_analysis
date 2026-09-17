"""
每日分析插件（MaiBot 1.0 / maibot_sdk 2.x）

功能：
- /summary [今天|昨天]      生成群聊整体总结图片
- /mysummary [今天|昨天]    生成自己的个人总结图片
- /mysummary @某人 [今天|昨天] / /mysummary QQ号 [今天|昨天]  查看他人总结（需权限）
- 每日定时自动生成群聊总结

所有宿主能力通过 ctx.* 调用；图片由宿主内置 render.html2png 渲染（无需自带浏览器）。
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple
from collections import Counter

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase

from .core import AnalysisService, SummaryRenderer


# ==================== 模块选项（WebUI 中文下拉）====================

# 群聊总结可选模块；"无"=该槽位不显示任何模块
GroupModuleOption = Literal[
    "无", "24H活跃轨迹", "今日话题", "群友画像", "语出惊人", "炫压抑评级"
]
# 个人总结可选模块；额外提供"并排"组合项以保留横向并排能力
PersonalModuleOption = Literal[
    "无", "3H活跃轨迹", "群友画像", "炫压抑评级", "语出惊人", "群友画像+炫压抑评级(并排)"
]

# 中文模块名 → 渲染器内部代码
_GROUP_MODULE_MAP = {
    "24H活跃轨迹": "24H",
    "今日话题": "Topics",
    "群友画像": "Portraits",
    "语出惊人": "Quotes",
    "炫压抑评级": "Rankings",
}
_PERSONAL_MODULE_MAP = {
    "3H活跃轨迹": "3H",
    "群友画像": "Portraits",
    "炫压抑评级": "Rankings",
    "语出惊人": "Quotes",
    "群友画像+炫压抑评级(并排)": "Portraits,Rankings",
}

# 定时自动总结：单个群处理的整体超时（秒），避免某群卡住拖垮整轮
_AUTO_SUMMARY_PER_GROUP_TIMEOUT = 120


def _slots_to_display_order(slots, mapping: dict) -> List[str]:
    """把若干下拉槽位（中文模块名，"无"表示不显示）按顺序转成渲染器用的代码列表，按模块去重。

    组合项（如 "Portraits,Rankings"）按其成员逐个去重：若某成员已在前面出现过，则跳过该槽位，
    避免同一模块在"独立"和"并排"中重复渲染。
    """
    order: List[str] = []
    seen = set()
    for slot in slots:
        code = mapping.get(slot)
        if not code:
            continue
        members = code.split(",")
        if any(member in seen for member in members):
            continue
        seen.update(members)
        order.append(code)
    return order


# ==================== 配置模型 ====================\n

class PluginSection(PluginConfigBase):
    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0
    enabled: bool = Field(
        default=False,
        description="是否启用插件",
        json_schema_extra={"label": "启用插件"},
    )
    config_version: str = Field(
        default="2.4.0",
        description="配置文件版本，用于兼容性校验，请勿手动修改",
        json_schema_extra={"label": "配置版本", "disabled": True},
    )


class SummarySection(PluginConfigBase):
    __ui_label__ = "群聊总结"
    __ui_icon__ = "message-square"
    __ui_order__ = 1
    # 5 个下拉槽位，按槽位顺序从上到下显示；某槽位选"无"即隐藏该位置
    slot_1: GroupModuleOption = Field(
        default="24H活跃轨迹",
        description="第 1 个显示的模块（选『无』则此位置不显示）",
        json_schema_extra={"label": "显示模块 1"},
    )
    slot_2: GroupModuleOption = Field(
        default="今日话题",
        description="第 2 个显示的模块",
        json_schema_extra={"label": "显示模块 2"},
    )
    slot_3: GroupModuleOption = Field(
        default="群友画像",
        description="第 3 个显示的模块",
        json_schema_extra={"label": "显示模块 3"},
    )
    slot_4: GroupModuleOption = Field(
        default="语出惊人",
        description="第 4 个显示的模块",
        json_schema_extra={"label": "显示模块 4"},
    )
    slot_5: GroupModuleOption = Field(
        default="炫压抑评级",
        description="第 5 个显示的模块",
        json_schema_extra={"label": "显示模块 5"},
    )
    max_depression_display: int = Field(
        default=6,
        description="炫压抑评级最多展示人数",
        json_schema_extra={"label": "炫压抑最多展示人数"},
    )
    depression_show_bottom: bool = Field(
        default=True,
        description="是否展示倒数排名（开启：前N/2名+后N/2名；关闭：只展示前N名）",
        json_schema_extra={"label": "展示倒数排名"},
    )
    highlight_time_mode: Literal["消息时间跨度", "最活跃时段"] = Field(
        default="消息时间跨度",
        description="图片顶部 Highlight Time 的显示方式：消息时间跨度=今日最早消息到生成前最晚消息；最活跃时段=发言最多的那一小时",
        json_schema_extra={"label": "Highlight Time 显示"},
    )


class UserSummarySection(PluginConfigBase):
    __ui_label__ = "个人总结"
    __ui_icon__ = "user"
    __ui_order__ = 2
    enabled: bool = Field(
        default=True,
        description="是否启用个人总结功能（/mysummary）",
        json_schema_extra={"label": "启用个人总结"},
    )
    # 查看他人名单模式：白名单=仅名单内可看他人；黑名单=名单内禁止看他人，其余人可看
    view_others_mode: Literal["白名单", "黑名单"] = Field(
        default="白名单",
        description="控制『查看他人个人总结』的名单生效模式。白名单=仅名单内的用户能看他人；黑名单=名单内的用户禁止看他人，其余人可看",
        json_schema_extra={"label": "查看他人名单模式"},
    )
    # 配合上面模式控制谁能看他人；为空时若为白名单则所有人可看他人（缺省放行），若为黑名单则没人被禁
    allowed_users: List[str] = Field(
        default=[],
        description="配合上面的模式控制谁能查看他人总结。所有人始终可以查看自己的总结",
        json_schema_extra={"label": "查看他人权限名单（QQ号）"},
    )
    # 4 个下拉槽位；可选\"无\"以隐藏；并排项保留原 2 并排布局
    slot_1: PersonalModuleOption = Field(
        default="3H活跃轨迹",
        description="第 1 个显示的模块（选『无』则此位置不显示）",
        json_schema_extra={"label": "显示模块 1"},
    )
    slot_2: PersonalModuleOption = Field(
        default="群友画像+炫压抑评级(并排)",
        description="第 2 个显示的模块",
        json_schema_extra={"label": "显示模块 2"},
    )
    slot_3: PersonalModuleOption = Field(
        default="语出惊人",
        description="第 3 个显示的模块",
        json_schema_extra={"label": "显示模块 3"},
    )
    slot_4: PersonalModuleOption = Field(
        default="无",
        description="第 4 个显示的模块",
        json_schema_extra={"label": "显示模块 4"},
    )


class AutoSummarySection(PluginConfigBase):
    __ui_label__ = "自动总结"
    __ui_icon__ = "clock"
    __ui_order__ = 3
    enabled: bool = Field(
        default=False,
        description="是否启用每日定时自动总结",
        json_schema_extra={"label": "启用自动总结"},
    )
    time: str = Field(
        default="23:00",
        description="每日执行时间（HH:MM，24 小时制）",
        json_schema_extra={"label": "执行时间"},
    )
    timezone: str = Field(
        default="Asia/Shanghai",
        description="时区（IANA 名称，如 Asia/Shanghai、America/New_York）",
        json_schema_extra={"label": "时区"},
    )
    min_messages: int = Field(
        default=10,
        description="触发自动总结所需的群聊今日最少消息数，少于此条数的群自动跳过",
        json_schema_extra={"label": "最少消息数"},
    )
    target_chats: List[str] = Field(
        default=[],
        description="要执行自动总结的目标群号列表（留空表示所有活跃群聊）",
        json_schema_extra={"label": "目标群号列表（留空=所有群）"},
    )


class CommandPermissionSection(PluginConfigBase):
    __ui_label__ = "命令权限"
    __ui_icon__ = "shield"
    __ui_order__ = 4
    mode: Literal["黑名单", "白名单"] = Field(
        default="黑名单",
        description="群聊权限控制模式。黑名单：列表中的群禁用命令；白名单：只有列表中的群可用",
        json_schema_extra={"label": "群聊权限模式"},
    )
    target_chats: List[str] = Field(
        default=[],
        description="群号列表（配合上面的模式生效）",
        json_schema_extra={"label": "生效群号列表"},
    )
    admin_users: List[str] = Field(
        default=[],
        description="允许在群聊中使用 /summary 的管理员 QQ 号列表（留空表示群内所有人可用）",
        json_schema_extra={"label": "/summary 管理员（留空=所有人）"},
    )


class AdvancedSection(PluginConfigBase):
    __ui_label__ = "高级"
    __ui_icon__ = "sliders"
    __ui_order__ = 5
    # 模型任务名：宿主 model_task_config 下的任务键（如 utils / planner / replyer / lpmm 等）。
    # 插件使用 ctx.llm.generate(model=...) 只认【任务名】；建议填快速无思考任务（utils / flash）。
    model_task: str = Field(
        default="utils",
        description="分析使用的宿主模型任务名（不是具体模型名，是任务名，如 utils、planner、replyer）。填错会在加载时校验并回退到 utils",
        json_schema_extra={"label": "模型任务名 (task)"},
    )
    inject_memory: bool = Field(
        default=False,
        description="实验性：把生成的总结注入麦麦记忆（群聊总结注入该群记忆，个人总结注入对该用户的记忆），让麦麦能记起总结过的事",
        json_schema_extra={"label": "注入麦麦记忆 (实验性)"},
    )
    llm_timeout_seconds: int = Field(
        default=60,
        description="单次 LLM 调用的客户端最长等待时间（秒）。注意：受宿主约 30 秒 RPC 硬上限约束，设太大在宿主超时时无额外效果",
        json_schema_extra={"label": "LLM 调用超时(秒)"},
    )
    render_timeout_seconds: int = Field(
        default=25,
        description="单次 HTML 渲染为图片的超时时间（秒）。根据服务器性能调整",
        json_schema_extra={"label": "图片渲染超时(秒)"},
    )


class DailyAnalysisConfig(PluginConfigBase):
    """每日分析插件根配置模型"""

    plugin: PluginSection = Field(default_factory=PluginSection)
    summary: SummarySection = Field(default_factory=SummarySection)
    user_summary: UserSummarySection = Field(default_factory=UserSummarySection)
    auto_summary: AutoSummarySection = Field(default_factory=AutoSummarySection)
    command_permission: CommandPermissionSection = Field(
        default_factory=CommandPermissionSection
    )
    advanced: AdvancedSection = Field(default_factory=AdvancedSection)


# ==================== 插件主类 ====================\n

class DailyAnalysisPlugin(MaiBotPlugin):
    """每日分析插件主类"""

    config_model = DailyAnalysisConfig

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._scheduler_task: Optional[asyncio.Task] = None
        self._service: Optional[AnalysisService] = None
        self._renderer: Optional[SummaryRenderer] = None
        # 正在后台生成的 (stream_id, scope) 集合，防止同一聊天流重复触发多次
        # scope: "group:YYYY-MM-DD" 或 "user:target_uid:YYYY-MM-DD"
        self._generating: set = set()
        # 跟踪所有后台分析/发图任务，卸载/重载时统一取消，避免悬挂协程
        self._background_tasks: set = set()

    # ---------- 生命周期 ----------

    async def on_load(self) -> None:
        """插件加载：初始化服务，启动定时调度器"""
        self.logger.info("每日分析插件正在加载...")
        await self._init_runtime_services()
        self._start_scheduler()
        self.logger.info("每日分析插件加载完成")

    async def on_unload(self) -> None:
        """插件卸载：取消调度器及所有进行中的后台分析任务"""
        self.logger.info("每日分析插件正在卸载...")
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        # 取消所有后台总结/发图任务
        pending = [t for t in self._background_tasks if not t.done()]
        if pending:
            self.logger.info(f"正在取消 {len(pending)} 个后台总结任务...")
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        self._background_tasks.clear()
        self._generating.clear()

        self.logger.info("每日分析插件已卸载")

    async def on_config_change(self) -> None:
        """配置热更新：重新初始化服务与调度器"""
        self.logger.info("每日分析插件配置已更新，重新应用...")
        await self._init_runtime_services()
        self._start_scheduler()

    async def _init_runtime_services(self) -> None:
        """初始化底层分析与渲染服务（绑定最新的 ctx 与配置）"""
        # 校验并回退模型任务
        validated_task = await self._validated_model_task()
        self._service = AnalysisService(
            self.ctx,
            model=validated_task,
            call_timeout_s=self._llm_timeout_seconds(),
        )
        self._renderer = SummaryRenderer(
            self.ctx,
            timeout_ms=self._render_timeout_ms(),
        )

    def _llm_timeout_seconds(self) -> int:
        return max(5, int(self.config.advanced.llm_timeout_seconds or 60))

    def _render_timeout_ms(self) -> int:
        return max(5, int(self.config.advanced.render_timeout_seconds or 25)) * 1000

    async def _validated_model_task(self) -> str:
        """校验配置的模型任务名是否为宿主可用任务；非法（如误填模型名）则回退 utils 并告警。"""
        want = (self.config.advanced.model_task or "utils").strip() or "utils"
        try:
            res = await self.ctx.llm.get_available_models()
            models = res.get("models") if isinstance(res, dict) else res
            if isinstance(models, list) and models:
                if want in models:
                    return want
                fallback = "utils" if "utils" in models else str(models[0])
                self.logger.warning(
                    f"配置的模型任务 '{want}' 不在宿主可用任务列表 {models} 中，已自动回退到 '{fallback}'"
                )
                return fallback
        except Exception as exc:
            self.logger.warning(f"获取可用模型任务列表失败: {exc}，使用配置值 '{want}'")
            return want
        return want

    # ---------- WebUI 配置 Schema 覆盖（按配置节分标签页） ----------

    def get_webui_config_schema(self, **kwargs: Any) -> Dict[str, Any]:
        """在 SDK 自动生成的配置 Schema 基础上，把布局改为「每个配置节一个标签页」。"""
        try:
            schema = super().get_webui_config_schema(**kwargs)
        except Exception:
            return {}
        try:
            if isinstance(schema, dict):
                sections = schema.get("sections")
                if isinstance(sections, dict) and sections:
                    ordered = sorted(
                        sections.items(),
                        key=lambda kv: (kv[1].get("order", 0) if isinstance(kv[1], dict) else 0),
                    )
                    tabs = []
                    for name, sec in ordered:
                        sec = sec if isinstance(sec, dict) else {}
                        tabs.append(
                            {
                                "id": name,
                                "title": sec.get("title") or name,
                                "icon": sec.get("icon"),
                                "order": sec.get("order", 0),
                                "sections": [name],
                            }
                        )
                    schema["layout"] = {"type": "tabs", "tabs": tabs}
        except Exception:
            return schema
        return schema

    # ---------- 工具：能力返回解析 ----------

    @staticmethod
    def _extract_messages(capability_result: Any) -> List[dict]:
        """从 ctx.message.get_by_time_in_chat 的返回中提取消息列表"""
        if not capability_result:
            return []
        if isinstance(capability_result, list):
            return capability_result
        if isinstance(capability_result, dict):
            for key in ("messages", "data", "result", "items"):
                val = capability_result.get(key)
                if isinstance(val, list):
                    return val
        return []

    # ---------- 工具：消息归一化 ----------

    @staticmethod
    def _normalize_message(msg: Any) -> Optional[dict]:
        """将不同版本的消息结构归一化为插件内部使用的统一字典。

        注意：MaiBot 宿主的 process_reply_component 会把「被回复的原消息文本」
        直接拼进回复者的 processed_plain_text，造成回复者被记成说了被回复的话。
        这里检测到存在 reply 段时，从 raw_message 的非 reply 段（text/at）
        重新拼装出回复者本人的真实文本，丢弃被引用的原文。
        """
        if not isinstance(msg, dict):
            return None

        # 读取原始段列表（优先 raw_message，兜底 segments / content）
        raw_segs = (
            msg.get("raw_message")
            or msg.get("segments")
            or msg.get("content")
        )

        user_id = str(
            msg.get("user_id")
            or msg.get("sender_id")
            or msg.get("author_id")
            or ""
        )
        nickname = str(
            msg.get("user_nickname")
            or msg.get("nickname")
            or msg.get("sender_name")
            or msg.get("author_name")
            or ""
        )
        cardname = str(
            msg.get("user_cardname")
            or msg.get("cardname")
            or msg.get("card")
            or ""
        )

        # 宿主默认的清洗文本
        plain_text = str(
            msg.get("processed_plain_text")
            or msg.get("plain_text")
            or msg.get("text")
            or msg.get("message")
            or ""
        )

        # 修复回复引用拼接 bug：若存在 reply 段，用 raw_message 的 text/at 段重构
        if isinstance(raw_segs, list) and any(
            isinstance(s, dict) and s.get("type") in ("reply", "quote") for s in raw_segs
        ):
            rebuilt_parts = []
            for s in raw_segs:
                if not isinstance(s, dict):
                    continue
                stype = s.get("type")
                if stype in ("reply", "quote"):
                    continue
                sdata = s.get("data")
                if isinstance(sdata, dict):
                    if stype == "text":
                        t = str(sdata.get("text") or "")
                        if t:
                            rebuilt_parts.append(t)
                    elif stype == "at":
                        t = str(sdata.get("qq") or sdata.get("user_id") or "")
                        if t:
                            rebuilt_parts.append(f"@{t}")
                elif isinstance(s.get("text"), str) and stype != "reply":
                    rebuilt_parts.append(s["text"])
            if rebuilt_parts:
                plain_text = "".join(rebuilt_parts).strip()

        # 时间戳（秒）
        t = msg.get("time") or msg.get("timestamp") or msg.get("created_at") or 0
        try:
            t = float(t)
        except Exception:
            t = 0.0

        is_cmd = bool(msg.get("is_command", False))
        is_notify = bool(
            msg.get("is_notify", False)
            or msg.get("message_type") in ("notify", "notice")
        )

        return {
            "user_id": user_id,
            "user_nickname": nickname,
            "user_cardname": cardname,
            "processed_plain_text": plain_text,
            "time": t,
            "is_command": is_cmd,
            "is_notify": is_notify,
        }

    # ---------- 历史消息查询与归一化 ----------

    async def _fetch_messages_in_range(
        self, stream_id: str, start_dt: datetime, end_dt: datetime
    ) -> List[dict]:
        """查询指定聊天流在时间范围内的消息，按时间升序返回归一化列表"""
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        try:
            res = await self.ctx.message.get_by_time_in_chat(
                stream_id=stream_id,
                start_time=start_ts,
                end_time=end_ts,
                filter_command=True,
            )
        except Exception as e:
            self.logger.error(f"查询历史消息失败 (stream={stream_id}): {e}")
            return []

        raw_list = self._extract_messages(res)
        normalized = []
        for item in raw_list:
            norm = self._normalize_message(item)
            if norm and norm.get("processed_plain_text"):
                normalized.append(norm)

        normalized.sort(key=lambda m: m.get("time", 0))
        return normalized

    # ---------- 模块槽位解析 ----------

    def _get_group_slots_order(self) -> List[str]:
        cfg = self.config.summary
        slots = [cfg.slot_1, cfg.slot_2, cfg.slot_3, cfg.slot_4, cfg.slot_5]
        return _slots_to_display_order(slots, _GROUP_MODULE_MAP)

    def _get_personal_slots_order(self) -> List[str]:
        cfg = self.config.user_summary
        slots = [cfg.slot_1, cfg.slot_2, cfg.slot_3, cfg.slot_4]
        return _slots_to_display_order(slots, _PERSONAL_MODULE_MAP)

    # ---------- 群聊总结生图流程 ----------

    async def _build_group_summary_image(
        self,
        messages: List[dict],
        summary: str,
        time_range: str,
        target_date: datetime,
    ) -> Optional[str]:
        """组装分析数据并渲染群聊总结长图，返回纯 base64"""
        slots = self._get_group_slots_order()
        active_users = AnalysisService.find_active_users(
            messages,
            min_count=5,
            max_users=8,
        )

        async def _topics():
            if "Topics" in slots:
                try:
                    return await self._service.extract_topics(messages)
                except Exception as e:
                    self.logger.warning(f"话题分析异常: {e}")
            return []

        async def _titles():
            if "Portraits" in slots:
                try:
                    return await self._service.generate_character_sketch(messages, active_users)
                except Exception as e:
                    self.logger.warning(f"群友画像分析异常: {e}")
            return []

        async def _quotes():
            if "Quotes" in slots:
                try:
                    return await self._service.extract_quotes(messages)
                except Exception as e:
                    self.logger.warning(f"金句分析异常: {e}")
            return []

        async def _depression():
            if "Rankings" in slots:
                try:
                    cfg = self.config.summary
                    return await self._service.analyze_depression(
                        messages,
                        active_users,
                        max_display=int(cfg.max_depression_display or 6),
                        show_bottom=bool(cfg.depression_show_bottom),
                    )
                except Exception as e:
                    self.logger.warning(f"炫压抑评级异常: {e}")
            return []

        topics_res, titles_res, quotes_res, dep_res = await asyncio.gather(
            _topics(), _titles(), _quotes(), _depression()
        )

        summary_data = {
            "summary": summary,
            "topics": topics_res,
            "titles": titles_res,
            "quotes": quotes_res,
            "depression": dep_res,
        }

        return await self._renderer.render_group_summary(
            messages=messages,
            summary_data=summary_data,
            slots=slots,
            time_range=time_range,
            target_date=target_date,
            highlight_time_mode=self.config.summary.highlight_time_mode,
        )

    # ---------- 个人总结生图流程 ----------

    async def _build_user_summary_image(
        self,
        user_messages: List[dict],
        user_name: str,
        user_id: str,
        target_date: datetime,
    ) -> Tuple[Optional[str], Optional[str]]:
        """组装分析数据并渲染个人总结长图，返回 (image_base64, summary_text)"""
        slots = self._get_personal_slots_order()
        time_range = "今天"

        try:
            analysis_res = await self._service.analyze_user_summary(
                user_messages, user_name, time_range
            )
        except Exception as e:
            self.logger.warning(f"个人总结分析异常: {e}")
            analysis_res = None

        if not analysis_res:
            return None, None

        image_base64 = await self._renderer.render_user_summary(
            user_messages=user_messages,
            user_name=user_name,
            user_id=user_id,
            analysis_data=analysis_res,
            slots=slots,
            target_date=target_date,
        )
        return image_base64, analysis_res.get("summary")

    # ---------- 记忆注入 (实验性) ----------

    async def _inject_memory(self, stream_id: str, content: str, source_tag: str) -> None:
        """把总结内容注入麦麦记忆上下文，支持群聊/个人两类"""
        try:
            await self.ctx.call_capability(
                "maisaka.context.append",
                stream_id=stream_id,
                content=content,
                tag=source_tag,
            )
            self.logger.info(f"已将总结注入麦麦记忆: stream={stream_id}, tag={source_tag}")
        except Exception as e:
            self.logger.debug(f"注入记忆未成功（宿主可能不支持或该功能未开启）: {e}")

    # ---------- 权限与过滤 ----------

    def _is_group_allowed(self, group_id: str) -> bool:
        """根据命令权限配置判断该群是否允许执行命令"""
        mode = self.config.command_permission.mode
        target_chats = [str(c) for c in self.config.command_permission.target_chats if c]
        gid = str(group_id)
        if mode == "白名单":
            # 白名单：列表为空则全部禁用；否则仅列表内允许
            return bool(target_chats) and gid in target_chats
        # 黑名单：列表内禁用，其余允许
        return gid not in target_chats

    # ==================== 后台总结任务（命令秒回，重活后台跑） ====================

    async def _run_group_summary_in_background(
        self, stream_id: str, group_id: str, messages: List[dict],
        time_range: str, target_date: datetime, guard_key: str,
    ) -> None:
        """后台执行群聊总结：分析→渲染→发送。不受宿主对命令处理的 60 秒硬超时限制。"""
        try:
            summary = await self._service.analyze_group_summary(messages, len(messages))
            if not summary:
                self.ctx.logger.error(f"群 {group_id} 群聊总结文本生成失败")
                await self.ctx.send.text("⚠️ 群聊总结生成失败（模型分析未返回有效内容），请稍后重试。", stream_id)
                return
            image_base64 = await self._build_group_summary_image(
                messages, summary, time_range, target_date
            )
            if not image_base64:
                self.ctx.logger.error(f"群 {group_id} 的群聊总结图片渲染失败")
                await self.ctx.send.text("⚠️ 群聊总结图片渲染失败，请检查渲染环境或稍后重试。", stream_id)
                return
            await self.ctx.send.image(image_base64, stream_id)
            if self.config.advanced.inject_memory:
                await self._inject_memory(
                    stream_id, f"【{time_range}群聊总结】{summary}", "plugin:daily_analysis:group"
                )
        except Exception as e:
            self.ctx.logger.error(f"后台群聊总结异常 (群 {group_id}): {e}", exc_info=True)
        finally:
            self._generating.discard(guard_key)

    async def _run_user_summary_in_background(
        self, stream_id: str, user_messages: List[dict], query_user_name: str,
        query_user_id: str, time_range: str, target_date: datetime, guard_key: str,
    ) -> None:
        """后台执行个人总结：分析→渲染→发送。不受宿主对命令处理的 60 秒硬超时限制。"""
        try:
            image_base64, user_summary_text = await self._build_user_summary_image(
                user_messages, query_user_name, query_user_id, target_date
            )
            if not image_base64:
                self.ctx.logger.error(f"用户 {query_user_id} 的个人总结图片渲染失败")
                await self.ctx.send.text("⚠️ 个人总结图片渲染失败，请检查渲染环境或稍后重试。", stream_id)
                return
            await self.ctx.send.image(image_base64, stream_id)
            if self.config.advanced.inject_memory and user_summary_text:
                note = f"【关于 {query_user_name}（QQ{query_user_id}）{time_range}的个人总结】{user_summary_text}"
                await self._inject_memory(
                    stream_id, note, f"plugin:daily_analysis:user:{query_user_id}"
                )
        except Exception as e:
            self.ctx.logger.error(f"后台个人总结异常 (用户 {query_user_id}): {e}", exc_info=True)
        finally:
            self._generating.discard(guard_key)

    # ==================== 命令：群聊总结 ====================

    @Command("summary", description="生成群聊总结", pattern=r"^/summary(?:\s+(?P<args>.*))?$")
    async def cmd_summary(
        self,
        event: Any,
        args: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str], int]:
        """处理 /summary [今天|昨天] 命令"""
        if not self.config.plugin.enabled:
            return False, None, 0

        stream_id = str(getattr(event, "stream_id", "") or "")
        group_id = str(getattr(event, "group_id", "") or "")
        user_id = str(getattr(event, "user_id", "") or "")

        # 仅群聊可用
        if not group_id:
            return True, "该命令仅在群聊中可用", 1

        # 群权限检查
        if not self._is_group_allowed(group_id):
            return True, None, 0

        # 管理员限制：非空时仅列表内可用
        admin_users = [str(u) for u in self.config.command_permission.admin_users if u]
        if admin_users and user_id not in admin_users:
            self.logger.info(f"用户 {user_id} 未在管理员列表中，拒绝执行 /summary")
            return True, f"你没有权限使用 /summary 命令（仅限管理员：{admin_users}）", 1

        # 解析时间范围
        time_arg = (args or "").strip()
        now = datetime.now()
        if time_arg in ("昨天", "yesterday", "yt"):
            time_range = "昨天"
            target_date = now - timedelta(days=1)
            start_dt = datetime.combine(target_date.date(), datetime.min.time())
            end_dt = datetime.combine(target_date.date(), datetime.max.time())
        else:
            time_range = "今天"
            target_date = now
            start_dt = datetime.combine(now.date(), datetime.min.time())
            end_dt = now

        # 防重复触发守护
        guard_key = f"group:{stream_id}:{target_date.strftime('%Y-%m-%d')}"
        if guard_key in self._generating:
            return True, f"正在为群生成{time_range}的总结，请耐心等待完成...", 1

        # 查询消息
        messages = await self._fetch_messages_in_range(stream_id, start_dt, end_dt)
        min_msg = max(5, int(self.config.auto_summary.min_messages or 10))
        if len(messages) < min_msg:
            return True, f"该群{time_range}仅有 {len(messages)} 条有效消息（少于 {min_msg} 条），暂无法生成总结", 1

        # 立即秒回确认提示，把重任务交给后台
        self._generating.add(guard_key)
        try:
            await self.ctx.send.text(f"⏳ 正在分析{time_range}的聊天记录，请稍候...", stream_id)
        except Exception:
            self._generating.discard(guard_key)
            raise

        task = asyncio.create_task(
            self._run_group_summary_in_background(
                stream_id, group_id, messages, time_range, target_date, guard_key
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return True, "已开始生成群聊总结", 1

    # ==================== 命令：个人总结 ====================

    @Command("mysummary", description="生成个人总结", pattern=r"^/mysummary(?:\s+(?P<args>.*))?$")
    async def cmd_mysummary(
        self,
        event: Any,
        args: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str], int]:
        """处理 /mysummary [@某人|QQ号] [今天|昨天] 命令"""
        if not self.config.plugin.enabled or not self.config.user_summary.enabled:
            return False, None, 0

        stream_id = str(getattr(event, "stream_id", "") or "")
        group_id = str(getattr(event, "group_id", "") or "")
        caller_id = str(getattr(event, "user_id", "") or "")

        if not group_id:
            return True, "该命令仅在群聊中可用", 1

        if not self._is_group_allowed(group_id):
            return True, None, 0

        # 解析参数：[@某人|QQ号] [今天|昨天]
        raw_args = (args or "").strip().split()
        target_uid: Optional[str] = None
        target_name: Optional[str] = None
        time_arg: str = "今天"

        # 检查是否 @ 了人（从 raw segments 解析）
        at_uid = None
        raw_msg = getattr(event, "raw_message", None) or getattr(event, "message", None)
        if isinstance(raw_msg, list):
            for seg in raw_msg:
                if isinstance(seg, dict) and seg.get("type") == "at":
                    data = seg.get("data", {})
                    at_uid = str(data.get("qq") or data.get("user_id") or "")
                    break

        for token in raw_args:
            if token in ("昨天", "yesterday", "yt"):
                time_arg = "昨天"
            elif token in ("今天", "today"):
                time_arg = "今天"
            elif token.isdigit():
                target_uid = token
            elif token.startswith("@") and len(token) > 1:
                target_name = token[1:]

        if at_uid:
            target_uid = at_uid

        # 默认查自己
        if not target_uid and not target_name:
            target_uid = caller_id

        # 权限校验：查看他人时的黑/白名单检查（所有人始终能看自己）
        is_viewing_self = target_uid == caller_id
        if not is_viewing_self:
            mode = self.config.user_summary.view_others_mode
            allowed_users = [str(u) for u in self.config.user_summary.allowed_users if u]
            if mode == "白名单":
                # 白名单模式：列表为空时放行；有值时仅列表内可看他人
                if allowed_users and caller_id not in allowed_users:
                    return True, "你没有查看他人总结的权限", 1
            else:
                # 黑名单模式：列表内禁止看他人
                if caller_id in allowed_users:
                    return True, "你已被禁止查看他人总结", 1

        # 时间范围
        now = datetime.now()
        if time_arg == "昨天":
            time_range = "昨天"
            target_date = now - timedelta(days=1)
            start_dt = datetime.combine(target_date.date(), datetime.min.time())
            end_dt = datetime.combine(target_date.date(), datetime.max.time())
        else:
            time_range = "今天"
            target_date = now
            start_dt = datetime.combine(now.date(), datetime.min.time())
            end_dt = now

        # 查询该群所有消息
        messages = await self._fetch_messages_in_range(stream_id, start_dt, end_dt)

        # 过滤目标用户的消息
        user_messages = []
        resolved_name = target_name or ""
        resolved_uid = target_uid or ""

        for m in messages:
            uid = m.get("user_id", "")
            nick = m.get("user_nickname", "")
            card = m.get("user_cardname", "")
            match = False
            if resolved_uid and uid == resolved_uid:
                match = True
            elif target_name and target_name in (card, nick):
                match = True
                if not resolved_uid and uid:
                    resolved_uid = uid

            if match:
                user_messages.append(m)
                if not resolved_name:
                    resolved_name = card or nick or uid

        if not resolved_name:
            resolved_name = resolved_uid or "群友"

        if len(user_messages) < 3:
            who = "你" if is_viewing_self else f"用户 {resolved_name}"
            return True, f"{who}{time_range}仅有 {len(user_messages)} 条有效发言（少于 3 条），暂无法生成个人画像", 1

        # 防重复触发守护
        guard_key = f"user:{stream_id}:{resolved_uid}:{target_date.strftime('%Y-%m-%d')}"
        if guard_key in self._generating:
            who = "你" if is_viewing_self else f"{resolved_name}"
            return True, f"正在为 {who} 生成{time_range}的个人总结，请稍候...", 1

        # 立即秒回确认提示，把重任务交给后台
        self._generating.add(guard_key)
        try:
            who = "你" if is_viewing_self else f"用户 {resolved_name}"
            await self.ctx.send.text(f"⏳ 正在分析 {who} {time_range}的发言记录，请稍候...", stream_id)
        except Exception:
            self._generating.discard(guard_key)
            raise

        task = asyncio.create_task(
            self._run_user_summary_in_background(
                stream_id, user_messages, resolved_name, resolved_uid,
                time_range, target_date, guard_key
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return True, "已开始生成个人总结", 1

    # ==================== 每日定时自动总结调度 ====================

    def _start_scheduler(self) -> None:
        """启动每日定时自动总结任务"""
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
        if not self.config.plugin.enabled or not self.config.auto_summary.enabled:
            self.logger.debug("自动总结未启用，跳过调度器启动")
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def _scheduler_loop(self) -> None:
        """定时调度器主循环：每天在设定时间触发自动总结"""
        import zoneinfo

        while True:
            try:
                tz_name = self.config.auto_summary.timezone or "Asia/Shanghai"
                try:
                    tz = zoneinfo.ZoneInfo(tz_name)
                except Exception:
                    self.logger.warning(f"未知时区 '{tz_name}'，回退到 Asia/Shanghai")
                    tz = zoneinfo.ZoneInfo("Asia/Shanghai")

                target_time_str = self.config.auto_summary.time or "23:00"
                try:
                    hour, minute = [int(p) for p in target_time_str.split(":", 1)]
                except Exception:
                    self.logger.warning(f"时间格式非法 '{target_time_str}'，回退到 23:00")
                    hour, minute = 23, 0

                now = datetime.now(tz)
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)

                wait_sec = (next_run - now).total_seconds()
                hours, rem = divmod(int(wait_sec), 3600)
                mins = rem // 60
                self.logger.info(
                    f"下次自动总结: {next_run.strftime('%Y-%m-%d %H:%M:%S')} (等待 {hours}小时{mins}分钟)"
                )
                await asyncio.sleep(wait_sec)

                # 时间到，执行每日自动总结
                await self._run_daily_summary(tz)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"自动总结调度器循环异常: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _run_daily_summary(self, tz: Any) -> None:
        """执行每日自动总结（遍历目标群）"""
        self.logger.info("开始执行每日自动群聊总结...")
        now = datetime.now(tz)
        target_date = now
        time_range = "今天"
        start_dt = datetime.combine(now.date(), datetime.min.time())
        end_dt = now

        # 获取目标聊天流
        configured = [str(c) for c in self.config.auto_summary.target_chats if c]
        active_streams: List[Tuple[str, str]] = []  # (stream_id, group_id)

        try:
            res = await self.ctx.chat.get_group_streams()
            streams = res.get("streams") if isinstance(res, dict) else res
            if isinstance(streams, list):
                for item in streams:
                    if isinstance(item, dict):
                        sid = str(item.get("stream_id") or "")
                        gid = str(item.get("group_id") or "")
                        if sid and gid:
                            if not configured or gid in configured:
                                active_streams.append((sid, gid))
        except Exception as e:
            self.logger.error(f"获取群聊流列表失败: {e}")
            return

        if not active_streams:
            self.logger.info("未找到符合条件的群聊流，本次自动总结结束")
            return

        min_msg = max(5, int(self.config.auto_summary.min_messages or 10))
        self.logger.info(f"共有 {len(active_streams)} 个群聊待总结（最少消息数 {min_msg}）")

        for stream_id, group_id in active_streams:
            try:
                # 给单个群的处理设置超时保护
                await asyncio.wait_for(
                    self._summary_one_group_auto(
                        stream_id, group_id, start_dt, end_dt, time_range, target_date, min_msg
                    ),
                    timeout=_AUTO_SUMMARY_PER_GROUP_TIMEOUT,
                )
            except asyncio.TimeoutError:
                self.logger.error(f"群 {group_id} 自动总结超时(>{_AUTO_SUMMARY_PER_GROUP_TIMEOUT}s)，跳过该群")
            except Exception as e:
                self.logger.error(f"群 {group_id} 自动总结执行异常: {e}", exc_info=True)

        self.logger.info("每日自动群聊总结全部执行完成")

    async def _summary_one_group_auto(
        self,
        stream_id: str,
        group_id: str,
        start_dt: datetime,
        end_dt: datetime,
        time_range: str,
        target_date: datetime,
        min_msg: int,
    ) -> None:
        """为单个群生成自动总结长图并发送"""
        messages = await self._fetch_messages_in_range(stream_id, start_dt, end_dt)
        if len(messages) < min_msg:
            self.logger.info(f"群 {group_id} 今日仅有 {len(messages)} 条消息，少于 {min_msg}，跳过")
            return

        summary = await self._service.analyze_group_summary(messages, len(messages))
        if not summary:
            self.logger.error(f"群 {group_id} 自动总结文本生成失败")
            return

        image_base64 = await self._build_group_summary_image(
            messages, summary, time_range, target_date
        )
        if not image_base64:
            self.logger.error(f"群 {group_id} 自动总结图片渲染失败")
            return

        await self.ctx.send.image(image_base64, stream_id)
        if self.config.advanced.inject_memory:
            await self._inject_memory(
                stream_id, f"【每日自动总结】{summary}", "plugin:daily_analysis:auto"
            )
        self.logger.info(f"群 {group_id} 自动总结已成功发送")


# 插件工厂入口（供 SDK 加载）
create_plugin = DailyAnalysisPlugin
