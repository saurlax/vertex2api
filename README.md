# vertex2api

A small Go proxy that uses a Vertex AI API key to serve OpenAI-compatible chat completions and forward native Gemini requests.

## Setup

Set these environment variables, then run `go run .` with Go 1.25 or later. You can also build an image using the [Dockerfile](Dockerfile) and pass the same variables at runtime.

| Variable | Value |
| --- | --- |
| `VERTEX_PROJECT_ID` | Google Cloud project with Vertex AI enabled. Required. |
| `VERTEX_API_KEY` | Vertex AI API key for that project. Required. |
| `API_KEY` | Separate bearer token for clients connecting to the proxy. Required. |
| `VERTEX_LOCATION` | Vertex AI location. Defaults to `global`. |
| `PORT` | Listen port. Defaults to `8080`. |

See [`.env.example`](.env.example) for the variable names and defaults. The program reads its process environment.

## Usage

Set your OpenAI client's base URL to `http://localhost:8080/v1` and use `API_KEY` as its API key.

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-4-hydrogen","messages":[{"role":"user","content":"Hello"}]}'
```

| Endpoint | Behavior |
| --- | --- |
| `POST /v1/chat/completions` | Text chat, structured JSON output, function tools, and SSE streaming. Supports one choice per request (`n=1`). |
| `GET /v1/models` | Returns a list of suggested model IDs. Vertex AI determines which models you can call. |
| `POST /vertex/{version}/models/{model}:{method}` | Forwards the native Vertex AI request body and response. Supports `v1` and `v1beta1`. |

When a chat completion returns `tool_calls`, send the assistant message back with the complete tool calls, including any `extra_content` thought signatures. Follow it with one `role: "tool"` message per call, using the matching `tool_call_id`.

Use the native endpoint for Vertex built-in tools and multimodal inputs. For example, `POST /vertex/v1/models/gemini-4-hydrogen:streamGenerateContent?alt=sse` streams a native Vertex AI response. Send the same bearer token as for chat completions. The proxy uses the configured project and location with a five-minute upstream timeout.
