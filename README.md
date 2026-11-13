# vertex2api

将 Vertex AI 的 Gemini 4 接入 OpenAI 兼容的 `chat.completions` 服务。支持普通文本对话、SSE 流式输出和用量统计。

## Setup

参考 [.env.example](.env.example) 设置环境变量。填写启用了 Vertex AI API 的项目 ID 和该项目的 Vertex API Key，另设一个独立的 `API_KEY` 给客户端使用。`VERTEX_LOCATION` 默认为 `global`，`PORT` 默认为 `8080`。

```powershell
go run .
```

容器部署时可以用仓库里的 Dockerfile 构建镜像，并通过运行环境注入同样的变量。

## Usage

模型使用以下 ID。

```text
gemini-4-hydrogen
gemini-4-carbon
gemini-4-argon
```

查看模型。

```powershell
curl.exe http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $env:API_KEY"
```

调用聊天。

```powershell
$body = @{ model = 'gemini-4-hydrogen'; messages = @(@{ role = 'user'; content = 'hello' }) } | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post `
  -Headers @{ Authorization = "Bearer $env:API_KEY" } -ContentType 'application/json' -Body $body
```

OpenAI 客户端的 Base URL 填 `http://127.0.0.1:8080/v1`。请求只发送到配置的 Vertex 项目，该项目的 Cloud Billing 设置决定费用和赠金抵扣。
