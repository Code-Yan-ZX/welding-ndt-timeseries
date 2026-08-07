#!/usr/bin/env bash
# 多模态路由代理管理脚本
# 用法:
#   bash scripts/multimodal_proxy.sh seed     # 从 settings.json 生成配置 (text 路由自动填好)
#   bash scripts/multimodal_proxy.sh check    # 校验配置
#   bash scripts/multimodal_proxy.sh start    # 后台启动代理 (nohup, 日志 ~/.claude/multimodal_proxy.log)
#   bash scripts/multimodal_proxy.sh stop     # 停止代理
#   bash scripts/multimodal_proxy.sh status   # 查看代理与健康检查
#   bash scripts/multimodal_proxy.sh enable   # 备份 settings.json 并把 Claude Code 指向本地代理 (需重启 Claude Code 生效)
#   bash scripts/multimodal_proxy.sh disable  # 从备份还原 settings.json (回到直连火山方舟)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY="$REPO/scripts/multimodal_proxy.js"
HOME_CLAUDE="${HOME}/.claude"
SETTINGS="${HOME_CLAUDE}/settings.json"
BAK="${HOME_CLAUDE}/settings.json.bak-direct"
PIDF="${HOME_CLAUDE}/multimodal_proxy.pid"
LOG="${HOME_CLAUDE}/multimodal_proxy.log"
PORT="${MULTIMODAL_PROXY_PORT:-8787}"

need_node() { command -v node >/dev/null || { echo "缺少 node"; exit 1; }; }

case "${1:-}" in
  seed)
    need_node
    node "$PROXY" --seed "${@:2}"
    ;;
  check)
    need_node
    node "$PROXY" --check
    ;;
  start)
    need_node
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "代理已在运行 (pid $(cat "$PIDF"))"; exit 0
    fi
    nohup node "$PROXY" >"$LOG" 2>&1 &
    echo $! >"$PIDF"
    sleep 1
    if curl -sf "http://127.0.0.1:${PORT}/_router_health" >/dev/null 2>&1; then
      echo "代理已启动 (pid $(cat "$PIDF"), port ${PORT})"
      echo "日志: tail -f $LOG"
    else
      echo "代理可能未就绪, 查看日志: $LOG"; tail -n 20 "$LOG" || true
    fi
    ;;
  stop)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      kill "$(cat "$PIDF")" && echo "已停止代理 (pid $(cat "$PIDF"))"
      rm -f "$PIDF"
    else
      echo "代理未运行"; rm -f "$PIDF"
    fi
    ;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "进程: 运行中 (pid $(cat "$PIDF"))"
    else
      echo "进程: 未运行"
    fi
    if curl -sf "http://127.0.0.1:${PORT}/_router_health" 2>/dev/null; then
      echo ""
    else
      echo "健康检查: 不可达"
    fi
    ;;
  enable)
    # 备份并改写 settings.json -> 指向本地代理
    [ -f "$SETTINGS" ] || { echo "找不到 $SETTINGS"; exit 1; }
    if [ ! -f "$BAK" ]; then cp "$SETTINGS" "$BAK"; echo "已备份原配置 -> $BAK"; fi
    node -e '
      const fs=require("fs");
      const p=process.argv[1], bak=process.argv[2], port=process.argv[3];
      const s=JSON.parse(fs.readFileSync(p,"utf8"));
      s.env=s.env||{};
      s.env.ANTHROPIC_BASE_URL="http://127.0.0.1:"+port;
      s.env.ANTHROPIC_AUTH_TOKEN="local-proxy";
      s.env.ANTHROPIC_MODEL=s.env.ANTHROPIC_MODEL||"glm-5.2[1m]";
      s.model=s.env.ANTHROPIC_MODEL;
      fs.writeFileSync(p, JSON.stringify(s,null,2)+"\n","utf8");
      console.log("已将 settings.json 指向本地代理:");
      console.log("  ANTHROPIC_BASE_URL = "+s.env.ANTHROPIC_BASE_URL);
      console.log("  ANTHROPIC_MODEL    = "+s.env.ANTHROPIC_MODEL);
      console.log("  (原配置备份在 "+bak+")");
      console.log(">>> 请重启 Claude Code 生效 <<<");
    ' "$SETTINGS" "$BAK" "$PORT"
    ;;
  disable)
    if [ -f "$BAK" ]; then
      cp "$BAK" "$SETTINGS"
      echo "已从备份还原 settings.json (回到直连火山方舟): $BAK"
      echo ">>> 请重启 Claude Code 生效 <<<"
    else
      echo "找不到备份 $BAK, 无法自动还原"
    fi
    ;;
  *)
    echo "用法: bash scripts/multimodal_proxy.sh {seed|check|start|stop|status|enable|disable}"
    exit 1
    ;;
esac
