# API 服务启动脚本(推荐方式,替代裸 uvicorn 命令)
#
# 为什么需要它:中文 Windows 的默认 ANSI 编码是 GBK,Docling 加载布局模型时
# 第三方库以 open() 默认编码读取 UTF-8 文件会报
# "'gbk' codec can't decode byte ..."(2026-08-07 实测)。
# PYTHONUTF8=1 启用解释器级 UTF-8 模式,须在解释器启动前设置,故固化为启动脚本。
#
# 用法(任意目录):  .\backend\scripts\start_api.ps1
# 脚本通过 $PSScriptRoot 定位仓库根并切换工作目录,保持相对配置语义不变;
# try/finally 恢复调用方目录,exit 透传子进程退出码。

$repoRoot = (Get-Item (Join-Path $PSScriptRoot ".." "..")).FullName
Push-Location $repoRoot
try {
    $env:PYTHONUTF8 = "1"
    conda run -n rag-kb --no-capture-output python -m uvicorn rag_kb.main:app --port 8000
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
