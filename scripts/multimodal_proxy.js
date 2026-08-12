#!/usr/bin/env node
/**
 * 本地多模态路由代理 (Anthropic 协议透传)
 * ---------------------------------------------------------------
 * Claude Code 整个对话只认一个主模型 / 一个 base_url。本代理插在
 * Claude Code 与上游之间, 按「请求体里是否含图片/文档块」自动分流:
 *
 *   含 image/document 块  ->  image 路由 (你的多模态模型)
 *   纯文字                ->  text  路由 (火山方舟 deepseek-v4-flash[1m], 原样透传)
 *
 * 两侧都是 Anthropic /v1/messages 协议, 因此无需协议翻译, 只做:
 *   1. 解析 body 检测图片块
 *   2. 选上游 + 重写 model 字段 + 套上对应凭证
 *   3. 流式 (SSE) 原样回传
 *
 * 配置: ~/.claude/multimodal_router.json  (在 ~/.claude 下, 不进仓库)
 *   首次使用: node scripts/multimodal_proxy.js --seed
 *
 * 用法:
 *   node scripts/multimodal_proxy.js            # 启动代理
 *   node scripts/multimodal_proxy.js --seed     # 从 settings.json 生成配置
 *   node scripts/multimodal_proxy.js --check    # 校验配置后退出
 */
'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const HOME = process.env.HOME || process.env.USERPROFILE;
const CONFIG_PATH = process.env.MULTIMODAL_ROUTER_CONFIG ||
  path.join(HOME, '.claude', 'multimodal_router.json');
const SETTINGS_PATH = path.join(HOME, '.claude', 'settings.json');

function log(...a) {
  console.log(`[${new Date().toISOString()}]`, ...a);
}

/* ----------------------------- 配置 ----------------------------- */

function readSettings() {
  if (!fs.existsSync(SETTINGS_PATH)) return {};
  try { return JSON.parse(fs.readFileSync(SETTINGS_PATH, 'utf8')); }
  catch { return {}; }
}

function seedConfig(force) {
  if (fs.existsSync(CONFIG_PATH) && !force) {
    console.log(`配置已存在: ${CONFIG_PATH} (用 --seed --force 覆盖)`);
    return;
  }
  const s = readSettings();
  const env = s.env || {};
  const token = env.ANTHROPIC_AUTH_TOKEN || env.ANTHROPIC_API_KEY || '';
  const authHeader = env.ANTHROPIC_AUTH_TOKEN ? 'bearer'
                   : env.ANTHROPIC_API_KEY  ? 'x-api-key' : 'bearer';
  const base = env.ANTHROPIC_BASE_URL || 'https://ark.cn-beijing.volces.com/api/coding';
  const model = env.ANTHROPIC_MODEL || s.model || 'deepseek-v4-flash[1m]';
  const cfg = {
    _comment: '本地多模态路由代理配置。仅存于 ~/.claude/ 下, 不进仓库。image 路由用于含图片/文档的请求, text 路由用于纯文字。',
    port: 8787,
    text: {
      base_url: base,
      auth_token: token,
      auth_header: authHeader,
      model: model
    },
    image: {
      base_url: '<填你的多模态模型 base_url, 例 https://host.com/v1 或 https://host.com>',
      auth_token: '<填你的多模态 apikey>',
      auth_header: 'both',
      model: '<填你的多模态模型名>'
    }
  };
  fs.mkdirSync(path.dirname(CONFIG_PATH), { recursive: true });
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2), 'utf8');
  const mask = (t) => t ? `${t.slice(0, 6)}...(${t.length}字符)` : '(空)';
  console.log(`已生成配置: ${CONFIG_PATH}`);
  console.log(`  text  -> ${cfg.text.base_url} | model=${cfg.text.model} | token=${mask(cfg.text.auth_token)} | auth=${cfg.text.auth_header}`);
  console.log(`  image -> 待填写 (base_url / auth_token / model)`);
  console.log(`\n请编辑该文件填好 image 路由的 base_url / auth_token / model, 再启动代理。`);
}

function loadConfig() {
  if (!fs.existsSync(CONFIG_PATH)) {
    console.error(`配置不存在: ${CONFIG_PATH}\n请先运行: node scripts/multimodal_proxy.js --seed`);
    process.exit(1);
  }
  const cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
  for (const r of ['text', 'image']) {
    const x = cfg[r];
    if (!x || !x.base_url || !x.auth_token || !x.model) {
      console.error(`路由 [${r}] 配置不完整 (需要 base_url / auth_token / model)。请编辑 ${CONFIG_PATH}`);
      process.exit(1);
    }
    const placeholder = String(x.base_url).startsWith('<') || String(x.auth_token).startsWith('<') || String(x.model).startsWith('<');
    if (placeholder) {
      if (r === 'text') {
        console.error(`路由 [text] 仍是占位符, text 路由必须配置真实值: ${CONFIG_PATH}`);
        process.exit(1);
      }
      // image 路由允许暂未配置: 代理照常启动 (text 可用), 含图片的请求会返回明确错误
      x._placeholder = true;
      console.error(`[警告] image 路由仍是占位符, 含图片的请求将返回 502; 填好 ${CONFIG_PATH} 后重启代理即可`);
      continue;
    }
    x.auth_header = x.auth_header || 'both';
    try { x._parsed = new URL(x.base_url); }
    catch { console.error(`路由 [${r}] base_url 无法解析: ${x.base_url}`); process.exit(1); }
  }
  cfg.port = cfg.port || 8787;
  return cfg;
}

/* --------------------------- 图片检测 --------------------------- */

// 递归查找 Anthropic 图片/文档内容块。Anthropic 块形如
// {type:'image', source:{type:'base64',...}} / {type:'document',...}
function containsMedia(obj) {
  if (obj == null) return false;
  if (Array.isArray(obj)) return obj.some(containsMedia);
  if (typeof obj === 'object') {
    const t = obj.type;
    if (t === 'image' || t === 'image_url' || t === 'document') return true;
    for (const v of Object.values(obj)) if (containsMedia(v)) return true;
  }
  return false;
}

/* --------------------------- 请求转发 --------------------------- */

const HOP_BY_HOP = new Set([
  'host', 'connection', 'content-length', 'authorization', 'x-api-key',
  'transfer-encoding', 'keep-alive', 'proxy-connection', 'proxy-authorization'
]);

function buildReqHeaders(reqHeaders, upstream, targetHost, bodyLen) {
  const h = {};
  for (const [k, v] of Object.entries(reqHeaders)) {
    if (HOP_BY_HOP.has(k.toLowerCase())) continue;
    h[k] = v;
  }
  const ah = upstream.auth_header || 'both';
  if (ah === 'bearer') h['authorization'] = 'Bearer ' + upstream.auth_token;
  else if (ah === 'x-api-key') h['x-api-key'] = upstream.auth_token;
  else { h['authorization'] = 'Bearer ' + upstream.auth_token; h['x-api-key'] = upstream.auth_token; }
  h['host'] = targetHost;
  h['content-length'] = String(bodyLen);
  return h;
}

const agents = {};
function agentFor(proto) {
  if (!agents[proto]) {
    const Cls = proto === 'https:' ? https.Agent : http.Agent;
    agents[proto] = new Cls({ keepAlive: true, maxSockets: 64 });
  }
  return agents[proto];
}

function forward(req, res, cfg) {
  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('error', (e) => { if (!res.headersSent) res.writeHead(400); res.end(String(e)); });
  req.on('end', () => {
    const raw = Buffer.concat(chunks);
    let parsed = null;
    try { parsed = chunks.length ? JSON.parse(raw.toString('utf8')) : null; } catch {}

    let routeName, upstream, outBody, detected = '';
    if (parsed && containsMedia(parsed)) {
      routeName = 'image'; upstream = cfg.image; detected = ' [media]';
      if (upstream._placeholder) {
        res.writeHead(502, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ type: 'error', error: { type: 'image_route_not_configured',
          message: '请求含图片, 但 image 路由尚未配置。请编辑 ~/.claude/multimodal_router.json 填好 base_url/auth_token/model 后重启代理。' } }));
        log(`image [media] ${req.method} ${req.url} -> 502 (image 路由未配置)`);
        return;
      }
      parsed.model = upstream.model;
      outBody = Buffer.from(JSON.stringify(parsed));
    } else {
      routeName = 'text'; upstream = cfg.text;
      // text 路由: Claude Code 发来的 model 已是 deepseek-v4-flash[1m], 直接原样透传, 零改写风险
      outBody = raw;
      if (parsed && parsed.model && parsed.model !== upstream.model) {
        parsed.model = upstream.model;
        outBody = Buffer.from(JSON.stringify(parsed));
      }
    }

    const base = upstream._parsed;
    const targetPath = base.pathname.replace(/\/$/, '') + req.url;
    const targetUrl = new URL(targetPath, base.origin);
    const mod = targetUrl.protocol === 'https:' ? https : http;
    const headers = buildReqHeaders(req.headers, upstream, targetUrl.host, Buffer.byteLength(outBody));
    const t0 = Date.now();

    const upReq = mod.request(targetUrl, { method: req.method, headers, agent: agentFor(targetUrl.protocol) }, (upRes) => {
      const respHeaders = {};
      for (const [k, v] of Object.entries(upRes.headers)) {
        if (HOP_BY_HOP.has(k.toLowerCase())) continue;
        respHeaders[k] = v;
      }
      res.writeHead(upRes.statusCode, respHeaders);
      upRes.pipe(res);
      upRes.on('end', () =>
        log(`${routeName}${detected} ${req.method} ${req.url} -> ${upstream.model} [${upRes.statusCode}] ${Date.now() - t0}ms`));
    });
    upReq.on('error', (e) => {
      log(`UPSTREAM ERROR [${routeName}]: ${e.message}`);
      if (!res.headersSent) res.writeHead(502, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ type: 'error', error: { type: 'upstream_error', message: String(e.message || e) } }));
    });
    upReq.write(outBody);
    upReq.end();
  });
}

/* ----------------------------- 主 ----------------------------- */

function main() {
  const argv = process.argv.slice(2);
  if (argv.includes('--seed')) {
    seedConfig(argv.includes('--force'));
    return;
  }

  const cfg = loadConfig();
  if (argv.includes('--check')) {
    const mask = (t) => `${String(t).slice(0, 6)}...(${String(t).length}字符)`;
    console.log('配置校验通过:');
    console.log(`  port  = ${cfg.port}`);
    console.log(`  text  -> ${cfg.text.base_url} | model=${cfg.text.model} | token=${mask(cfg.text.auth_token)} | auth=${cfg.text.auth_header}`);
    const ph = cfg.image._placeholder ? ' [未配置-占位符]' : '';
    console.log(`  image -> ${cfg.image.base_url}${ph}`);
    if (!ph) console.log(`         model=${cfg.image.model} | token=${mask(cfg.image.auth_token)} | auth=${cfg.image.auth_header}`);
    return;
  }

  const server = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/_router_health') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true, text: cfg.text.model, image: cfg.image.model }));
      return;
    }
    forward(req, res, cfg);
  });

  server.on('error', (e) => { console.error('启动失败:', e.message); process.exit(1); });
  server.listen(cfg.port, '127.0.0.1', () => {
    log(`多模态路由代理已启动: http://127.0.0.1:${cfg.port}`);
    log(`  text  -> ${cfg.text.base_url} (${cfg.text.model})`);
    log(`  image -> ${cfg.image.base_url} (${cfg.image.model})`);
    log(`  健康检查: curl http://127.0.0.1:${cfg.port}/_router_health`);
  });

  for (const sig of ['SIGINT', 'SIGTERM']) {
    process.on(sig, () => { log('收到退出信号, 关闭代理'); server.close(() => process.exit(0)); });
  }
}

main();
