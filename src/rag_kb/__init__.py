"""rag_kb 根包 bootstrap。

LangGraph 的 ``STRICT_MSGPACK_ENABLED`` 在其模块被 import 时读取一次环境变量并冻结。
任何 API/Worker 导入链都会先经过本包，因此这里必须在最早期无条件打开该开关，
否则 ``rag_kb.agent`` 对 LangGraph 的预导入会以非严格模式固化序列化边界。
"""

import os

# LangGraph 在导入时冻结该开关；必须先于任何 langgraph 导入设置。
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
