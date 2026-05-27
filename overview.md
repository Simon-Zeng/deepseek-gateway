# DeepSeek Gateway - AI Model Proxy

## Overview

An AI model proxy gateway that routes OpenAI/Anthropic API requests to DeepSeek, converting both request and response formats transparently. Domain: `llm.gorobotic.cn`

## What Was Built

### API Endpoints
- **POST /v1/chat/completions** — OpenAI Chat Completions API (near pass-through to DeepSeek)
- **POST /v1/responses** — OpenAI Responses API (maps `reasoning_content` to reasoning output items)
- **POST /v1/messages** — Anthropic Messages API (maps `reasoning_content` to thinking blocks)
- **GET /v1/models** — List available models
- **GET /health** — Health check

### Key Features
1. **Three-protocol support**: OpenAI Chat, OpenAI Responses, Anthropic Messages
2. **Streaming-first**: Full SSE streaming with protocol-aware conversion
3. **Reasoning/Thinking**: DeepSeek V4 Pro `reasoning_content` → OpenAI reasoning items / Anthropic thinking blocks
4. **Configurable model mapping**: Regex-based rules map model names → deepseek-v4-flash or deepseek-v4-pro
5. **Reasoning effort override**: When `reasoning.effort` is "high" or above, forces deepseek-v4-pro regardless of mapping
6. **Hybrid auth**: Gateway API key or client key forwarding
7. **State machine streaming**: Clean phase tracking (IDLE → REASONING → CONTENT → DONE)

### Model Mapping

| Client Model | Target Model | Notes |
|---|---|---|
| gpt-4o, gpt-4 | deepseek-v4-flash | Standard GPT models |
| gpt-3.5-turbo, gpt-4-turbo | deepseek-v4-pro | Turbo models → Pro |
| o1, o1-mini, o3-mini | deepseek-v4-pro | Reasoning models |
| claude-opus-4, claude-3-opus | deepseek-v4-pro | Opus models → Pro |
| claude-3-5-sonnet, claude-haiku, claude-sonnet-4 | deepseek-v4-flash | Other Claude → Flash |
| deepseek-v4-flash, deepseek-v4-pro | Pass-through | Direct DeepSeek access |
| Any reasoning.effort="high" | deepseek-v4-pro | Override regardless of model |

### Architecture
```
Client Request → Router → Converter (→ DeepSeek format) → DeepSeek Client
                                                              ↓
Client Response ← Streamer/Converter ← DeepSeek Response ← ← ←
```

### Project Structure
```
deepseek-gateway/
├── app/
│   ├── main.py              # FastAPI app, middleware, lifespan
│   ├── config.py             # Pydantic Settings (env + YAML)
│   ├── dependencies.py       # Auth dependency injection
│   ├── routers/               # API endpoints
│   │   ├── openai_chat.py     # /v1/chat/completions
│   │   ├── openai_responses.py # /v1/responses
│   │   ├── anthropic_messages.py # /v1/messages
│   │   └── models.py          # /v1/models
│   ├── converters/            # Request/Response format conversion
│   │   ├── openai_chat.py
│   │   ├── openai_responses.py
│   │   └── anthropic.py
│   ├── streamers/             # SSE streaming conversion
│   │   ├── openai_chat.py
│   │   ├── openai_responses.py
│   │   └── anthropic.py
│   ├── models/                # Pydantic schemas
│   ├── services/              # DeepSeek client, model mapper
│   └── utils/                  # SSE, errors, logging
├── config/
│   ├── model_mapping.yaml     # Model name mapping rules
│   └── gateway.yaml           # Server/deepseek/logging config
├── nginx/conf.d/              # Nginx HTTPS config for llm.gorobotic.cn
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

# Test model listing
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
| 4 | TCP | 8000 | 127.0.0.1/32 | 仅本地 (Docker 内部) |

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
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

**OpenAI Responses (with reasoning effort override):**
```bash
curl https://llm.gorobotic.cn/v1/responses \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model": "gpt-4o", "input": "Solve this problem", "reasoning": {"effort": "high"}}'
```

**Anthropic (Claude Code):**
```bash
curl https://llm.gorobotic.cn/v1/messages \
  -H "x-api-key: YOUR_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1024}'
```

**For Claude Code, set these environment variables:**
```bash
export ANTHROPIC_BASE_URL=https://llm.gorobotic.cn
export ANTHROPIC_API_KEY=YOUR_KEY
```

**For Codex/OpenAI tools:**
```bash
export OPENAI_BASE_URL=https://llm.gorobotic.cn/v1
export OPENAI_API_KEY=YOUR_KEY
```

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
