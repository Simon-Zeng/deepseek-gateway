# DeepSeek Gateway - AI Model Proxy

## Overview

An AI model proxy gateway that routes OpenAI/Anthropic API requests to DeepSeek, converting both request and response formats transparently. Designed to work with Codex, Claude Code, and Xcode Agent out of the box. Domain: `llm.gorobotic.cn`

## What Was Built

### API Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI Chat | Near pass-through to DeepSeek with content array flattening and tool call passthrough |
| `POST /v1/responses` | OpenAI Responses | Maps `reasoning_content` to reasoning output items; supports `reasoning.effort` override |
| `POST /v1/messages` | Anthropic Messages | Maps `reasoning_content` to thinking blocks; bidirectional tool_use ↔ tool_calls conversion |
| `GET /v1/models` | OpenAI | Proxies DeepSeek's model list directly |
| `GET /health` | — | Health check |

### Key Features

1. **Three-protocol support**: OpenAI Chat, OpenAI Responses, Anthropic Messages
2. **Streaming-first**: Full SSE streaming with protocol-aware conversion and state machine (IDLE → REASONING → CONTENT → DONE)
3. **Reasoning/Thinking**: DeepSeek V4 Pro `reasoning_content` → OpenAI reasoning items / Anthropic thinking blocks
4. **Configurable model mapping**: Regex-based rules map model names → `deepseek-v4-flash` or `deepseek-v4-pro`
5. **Reasoning effort override**: When `reasoning.effort` ≥ "high" or `thinking.budget_tokens` ≥ 10000, forces `deepseek-v4-pro` regardless of mapping
6. **Unified auth**: Single `Depends(verify_api_key)` dependency across all routers — gateway key mode or key forwarding mode
7. **Xcode Agent compatibility**:
   - Content arrays (`[{"type": "text", ...}]`) → flattened to strings for DeepSeek
   - `image_url` parts → discarded (DeepSeek is text-only)
   - `tool_calls` / `tool_call_id` / `name` → preserved through OpenAI Chat pipeline
   - Anthropic `tool_use` blocks ↔ DeepSeek `tool_calls` bidirectional conversion
   - Anthropic `tool_result` blocks → DeepSeek `tool` role messages
   - Anthropic tool definitions (`input_schema`) → OpenAI format (`parameters`)
   - Anthropic `tool_choice` → DeepSeek `tool_choice` mapping
   - Streaming `tool_calls` → Anthropic `tool_use` content blocks with `input_json_delta`
8. **Models endpoint**: Proxies DeepSeek's `/v1/models` directly, returning only models DeepSeek actually provides

### Model Mapping

| Client Model | Target Model | Notes |
|---|---|---|
| gpt-4o, gpt-4 | deepseek-v4-flash | Standard GPT models |
| gpt-3.5-turbo, gpt-4-turbo | deepseek-v4-pro | Turbo models → Pro |
| o1, o1-mini, o3-mini | deepseek-v4-pro | Reasoning models |
| claude-opus-4, claude-3-opus | deepseek-v4-pro | Opus models → Pro |
| claude-3-5-sonnet, claude-haiku, claude-sonnet-4 | deepseek-v4-flash | Other Claude → Flash |
| deepseek-v4-flash, deepseek-v4-pro | Pass-through | Direct DeepSeek access |
| Any reasoning.effort ≥ "high" | deepseek-v4-pro | Override regardless of model |
| Any thinking.budget_tokens ≥ 10000 | deepseek-v4-pro | Anthropic thinking budget override |

### Architecture

```
Client Request → Router [Depends(verify_api_key)] → Converter (→ DeepSeek format) → DeepSeek Client
                                                                              ↓
Client Response ← Streamer/Converter ← DeepSeek Response ← ← ← ← ← ← ← ← ← ← ←
```

**Auth flow**:
- All routers use `Depends(verify_api_key)` from `app/dependencies.py`
- Accepts `Authorization: Bearer <key>` or `x-api-key` header
- If `GATEWAY_API_KEY` is set: validates client key against it, returns server's `DEEPSEEK_API_KEY`
- If `GATEWAY_API_KEY` is not set: forwards client's key directly to DeepSeek

### Tool Call Conversion (Anthropic ↔ DeepSeek)

**Request direction** (Anthropic → DeepSeek):
```
Anthropic tool_use blocks              DeepSeek tool_calls field
─────────────────────────              ──────────────────────────
content: [                              tool_calls: [
  {type: "tool_use",                     {id: "...", type: "function",
   id: "...", name: "...",                function: {name: "...",
   input: {...}}                           arguments: "{...}"}}
]                                       ]
```

**Response direction** (DeepSeek → Anthropic):
```
DeepSeek tool_calls                    Anthropic tool_use content blocks
───────────────────                    ──────────────────────────────────
tool_calls: [                          content: [
  {id: "...", type: "function",         {type: "tool_use", id: "...",
   function: {name: "...",               name: "...", input: {...}}
   arguments: "{...}"}}                ]
]

Streaming:
  delta.tool_calls → content_block_start (tool_use) + input_json_delta
```

**Tool definitions conversion**:
```
Anthropic: {name, description, input_schema}  →  OpenAI: {type: "function", function: {name, description, parameters}}
```

**Tool choice mapping**:
```
Anthropic: {type: "auto"}    → DeepSeek: "auto"
Anthropic: {type: "any"}     → DeepSeek: "required"
Anthropic: {type: "tool"}    → DeepSeek: {type: "function", function: {name: "..."}}
```

### Project Structure

```
deepseek-gateway/
├── app/
│   ├── main.py                      # FastAPI app, middleware, lifespan
│   ├── config.py                    # Pydantic Settings (env + YAML)
│   ├── dependencies.py              # Auth dependency injection (verify_api_key)
│   ├── routers/
│   │   ├── openai_chat.py           # /v1/chat/completions
│   │   ├── openai_responses.py      # /v1/responses
│   │   ├── anthropic_messages.py     # /v1/messages
│   │   └── models.py                # /v1/models (proxies DeepSeek)
│   ├── converters/
│   │   ├── openai_chat.py           # Content flattening, tool_calls passthrough
│   │   ├── openai_responses.py      # Reasoning effort → model mapping
│   │   └── anthropic.py             # tool_use ↔ tool_calls, tool_result → tool, tool defs conversion
│   ├── streamers/
│   │   ├── openai_chat.py           # SSE streaming for OpenAI Chat
│   │   ├── openai_responses.py      # SSE streaming for OpenAI Responses
│   │   └── anthropic.py             # State machine streaming (AnthropicStreamState + pending_tool_calls)
│   ├── models/                      # Pydantic schemas
│   ├── services/                    # DeepSeek client, model mapper
│   └── utils/                        # SSE, errors, logging
├── config/
│   ├── model_mapping.yaml           # Model name mapping rules + reasoning effort override
│   └── gateway.yaml                 # Server/deepseek/logging config
├── nginx/conf.d/                    # Nginx HTTPS config for llm.gorobotic.cn
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Deployment on Alibaba Cloud ECS

### Prerequisites
- Alibaba Cloud ECS instance (推荐: ecs.c7.xlarge 或以上, CentOS/Ubuntu)
- 域名 `llm.gorobotic.cn` 已在阿里云解析至 ECS 公网 IP
- 安全组已开放 80 和 443 端口

### Step 1: Install Docker

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable docker && sudo systemctl start docker

# CentOS
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-compose-plugin
sudo systemctl enable docker && sudo systemctl start docker
```

### Step 2: Deploy the Gateway

```bash
# Clone/copy the project to ECS
cd /opt
sudo git clone <repo-url> deepseek-gateway
cd deepseek-gateway

# Create .env from template
cp .env.example .env

# Edit .env and set your DeepSeek API key
sudo nano .env
```

Set these in `.env`:
```env
DEEPSEEK_API_KEY=sk-your-actual-deepseek-api-key
GATEWAY_API_KEY=sk-your-gateway-key-for-clients
```

### Step 3: SSL Certificate with certbot

```bash
# Install certbot
sudo apt-get install -y certbot    # Ubuntu
# sudo yum install -y certbot      # CentOS

# Create webroot directory for certbot
sudo mkdir -p /opt/deepseek-gateway/certbot-webroot

# Obtain certificate (replace with your email)
sudo certbot certonly \
  --webroot \
  --webroot-path /opt/deepseek-gateway/certbot-webroot \
  -d llm.gorobotic.cn \
  --email your@email.com \
  --agree-tos \
  --no-eff-email

# Copy certificates to nginx/ssl
sudo mkdir -p /opt/deepseek-gateway/nginx/ssl
sudo cp /etc/letsencrypt/live/llm.gorobotic.cn/fullchain.pem /opt/deepseek-gateway/nginx/ssl/
sudo cp /etc/letsencrypt/live/llm.gorobotic.cn/privkey.pem /opt/deepseek-gateway/nginx/ssl/
sudo chmod 644 /opt/deepseek-gateway/nginx/ssl/fullchain.pem
sudo chmod 600 /opt/deepseek-gateway/nginx/ssl/privkey.pem
```

### Step 4: Start Services

```bash
cd /opt/deepseek-gateway
sudo docker compose up -d --build
```

### Step 5: Verify

```bash
# Health check
curl http://localhost:8000/health

# Test through Nginx
curl https://llm.gorobotic.cn/health

# Test model listing (proxies DeepSeek's models)
curl https://llm.gorobotic.cn/v1/models \
  -H "Authorization: Bearer YOUR_GATEWAY_KEY"
```

### Step 6: Auto-renew SSL

```bash
# Add to crontab
sudo crontab -e

# Add this line (renew twice daily, reload nginx on success)
0 2,14 * * * certbot renew --quiet --deploy-hook "cp /etc/letsencrypt/live/llm.gorobotic.cn/*.pem /opt/deepseek-gateway/nginx/ssl/ && docker exec deepseek-gateway-nginx-1 nginx -s reload"
```

### Alibaba Cloud ECS Security Group

确保安全组入方向规则包含：

| 优先级 | 协议 | 端口 | 源地址 | 说明 |
|--------|------|------|--------|------|
| 1 | TCP | 22 | 你的IP | SSH 管理 |
| 2 | TCP | 80 | 0.0.0.0/0 | HTTP (certbot + 重定向) |
| 3 | TCP | 443 | 0.0.0.0/0 | HTTPS (API 访问) |
| 4 | TCP | 8000 | 127.0.0.1/32 | 仅本地 (Docker 内部，不对外暴露) |

### DNS Configuration (阿里云解析)

在阿里云域名控制台添加 A 记录：

| 记录类型 | 主机记录 | 记录值 | TTL |
|----------|---------|--------|-----|
| A | llm | 你的ECS公网IP | 600 |

---

## Usage Examples

**OpenAI Chat (Codex/any OpenAI-compatible client):**
```bash
curl https://llm.gorobotic.cn/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

**OpenAI Chat with tool calling:**
```bash
curl https://llm.gorobotic.cn/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
  }'
```

**OpenAI Responses (with reasoning effort override):**
```bash
curl https://llm.gorobotic.cn/v1/responses \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "input": "Solve this problem", "reasoning": {"effort": "high"}}'
```

**Anthropic (Claude Code):**
```bash
curl https://llm.gorobotic.cn/v1/messages \
  -H "x-api-key: YOUR_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1024}'
```

**Anthropic with tool use:**
```bash
curl https://llm.gorobotic.cn/v1/messages \
  -H "x-api-key: YOUR_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
    "messages": [{"role": "user", "content": "What is the weather in Beijing?"}]
  }'
```

**Client Environment Variables:**

For Claude Code:
```bash
export ANTHROPIC_BASE_URL=https://llm.gorobotic.cn
export ANTHROPIC_API_KEY=YOUR_KEY
```

For Codex/OpenAI tools:
```bash
export OPENAI_BASE_URL=https://llm.gorobotic.cn/v1
export OPENAI_API_KEY=YOUR_KEY
```

For Xcode Agent (uses OpenAI Chat API with content arrays and tool calls):
```bash
# Configure in Xcode's model settings
# Base URL: https://llm.gorobotic.cn/v1
# API Key: YOUR_KEY
# Model: gpt-4o (or any model name from the mapping table)
```

---

## Local Development & Testing

### Method 1: Direct uvicorn

```bash
cd /Users/simon.zeng/Documents/Code/deepseek-gateway
pip3 install -r requirements.txt

# Set env vars
export DEEPSEEK_API_KEY=sk-your-deepseek-key
# Optionally set GATEWAY_API_KEY for gateway auth mode

# Run
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

### Method 2: Docker Compose

```bash
cd /Users/simon.zeng/Documents/Code/deepseek-gateway

# Create .env
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY

docker compose up --build
```

### Test Endpoints

```bash
# Health
curl http://localhost:8000/health

# List models (proxies DeepSeek)
curl http://localhost:8000/v1/models -H "Authorization: Bearer YOUR_KEY"

# OpenAI Chat
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'

# Anthropic
curl http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-3-5-sonnet-20241022","messages":[{"role":"user","content":"Hello"}],"max_tokens":1024}'
```

---

## Version History

### v1 — Initial Build
- Three-protocol gateway: OpenAI Chat, OpenAI Responses, Anthropic Messages
- SSE streaming with state machine
- Configurable model mapping via YAML
- Hybrid auth (gateway key / key forwarding)

### v2 — Domain & Model Updates
- Domain: `model.gorobotic.cn` → `llm.gorobotic.cn`
- Model names: `deepseek-chat` → `deepseek-v4-flash`, `deepseek-reasoner` → `deepseek-v4-pro`
- Mapping: OpenAI turbo → Pro, Anthropic opus → Pro
- Reasoning effort override: `effort ≥ "high"` → force Pro
- Alibaba Cloud ECS deployment config

### v3 — Models Endpoint & Auth Refactor
- `/v1/models` now proxies DeepSeek's model list directly (instead of returning gateway-defined models)
- Auth unified: all routers use `Depends(verify_api_key)` from `app/dependencies.py`
- Removed duplicated `_resolve_api_key()` from individual routers

### v4 — Xcode Agent Compatibility
- **Content type fixes**:
  - Content arrays (`[{"type": "text", "text": "..."}]`) → flattened to strings
  - `image_url` parts → discarded (DeepSeek is text-only)
  - `tool` role messages → content flattened to string
- **Tool calls support (OpenAI Chat)**:
  - `tool_calls`, `tool_call_id`, `name` fields preserved in request conversion
  - `tools` and `tool_choice` definitions passed through to DeepSeek
- **Tool use support (Anthropic)**:
  - Request: `tool_use` blocks → `tool_calls` field, `tool_result` blocks → `tool` role messages
  - Request: Anthropic tool definitions (`input_schema`) → OpenAI format (`parameters`)
  - Request: Anthropic `tool_choice` → DeepSeek `tool_choice` mapping
  - Response: `tool_calls` → `tool_use` content blocks
  - Response: `finish_reason: "tool_calls"` → `stop_reason: "tool_use"`
- **Streaming tool calls (Anthropic)**:
  - `pending_tool_calls` dict in `AnthropicStreamState` for accumulating incremental tool call deltas
  - `content_block_start` (tool_use) + `input_json_delta` → streaming tool call output
- **Anthropic thinking budget**: `thinking.budget_tokens ≥ 10000` triggers Pro model override (like `reasoning.effort`)

---

## Maintenance

### View Logs
```bash
# Gateway logs
sudo docker compose logs -f gateway

# Nginx access logs
sudo tail -f /opt/deepseek-gateway/nginx/logs/access.log

# Nginx error logs
sudo tail -f /opt/deepseek-gateway/nginx/logs/error.log
```

### Restart
```bash
cd /opt/deepseek-gateway
sudo docker compose restart
```

### Update
```bash
cd /opt/deepseek-gateway
sudo git pull
sudo docker compose up -d --build
```
