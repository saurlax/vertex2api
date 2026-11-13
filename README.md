# vertex2api

用 Vertex AI API Key 调用 Gemini。提供 OpenAI Chat Completions 兼容入口和 Vertex 原生转发入口，支持 SSE。

## 配置

| 环境变量 | 用途 |
| --- | --- |
| VERTEX_PROJECT_ID | GCP 项目 ID，必填 |
| VERTEX_API_KEY | 用于调用该项目 Vertex AI 的 API Key，必填 |
| API_KEY | 客户端连接 vertex2api 的独立 Bearer Token，必填 |
| VERTEX_LOCATION | Vertex 区域，默认 global |
| PORT | 监听端口，默认 8080 |

设置环境变量后运行 go run .，或使用仓库里的 Dockerfile 构建镜像。

## 接口

- POST /v1/chat/completions — OpenAI 文本对话，支持流式和 JSON 输出。
- GET /v1/models — 返回推荐模型，实际模型可用性由 Vertex 决定。
- POST /vertex/{version}/models/{model}:{method} — 原样转发 Vertex 请求体与响应，支持 v1 和 v1beta1。例如 /vertex/v1/models/gemini-4-hydrogen:streamGenerateContent?alt=sse。工具调用、多模态、思考签名等使用这个入口。

OpenAI 客户端 Base URL 填 http://127.0.0.1:8080/v1。以上接口通过 Authorization: Bearer <API_KEY> 验证。上游使用启动时配置的项目、区域和 Vertex Key，超时为 5 分钟。
