package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestToolCallsRoundTrip(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(map[bool]string{false: "json", true: "stream"}[streaming], func(t *testing.T) {
			turn := 0
			vertex := newVertex("trial-project", "global", "vertex-key")
			vertex.Client.Transport = transport(func(r *http.Request) (*http.Response, error) {
				turn++
				var body vertexRequest
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if !strings.HasPrefix(r.URL.Path, "/v1beta1/") || len(body.Tools) != 1 ||
					body.Tools[0].Declarations[0].Name != "lookup" || body.ToolConfig.Calling.Mode != "ANY" {
					t.Fatalf("incorrect Vertex tool request: %+v", body)
				}
				result := `{"candidates":[{"content":{"parts":[{"functionCall":{"id":"native-id","name":"lookup","args":{"q":"Paris"}},"thoughtSignature":"opaque"}]},"finishReason":"STOP"}]}`
				contentType := "application/json"
				if turn == 2 {
					if len(body.Contents) != 3 || len(body.Contents[1].Parts) != 2 || len(body.Contents[2].Parts) != 2 {
						t.Fatalf("invalid parallel call history: %+v", body.Contents)
					}
					call, second := body.Contents[1].Parts[0], body.Contents[1].Parts[1]
					firstResult, secondResult := body.Contents[2].Parts[0].FunctionResponse, body.Contents[2].Parts[1].FunctionResponse
					if call.ThoughtSignature != "opaque" || call.Text != nil || call.FunctionCall.ID != "native-id" ||
						second.FunctionCall.ID != "" || firstResult.ID != "native-id" || secondResult.ID != "" ||
						string(firstResult.Response) != `{"status":"ok"}` || string(secondResult.Response) != `{"output":"London"}` {
						t.Fatalf("incorrect tool replay: %+v", body.Contents)
					}
					result = `{"candidates":[{"content":{"parts":[{"text":"Done"}]},"finishReason":"STOP"}]}`
				}
				if strings.HasSuffix(r.URL.Path, ":streamGenerateContent") {
					contentType = "text/event-stream"
					result = strings.Replace(result, `,"finishReason":"STOP"`, "", 1)
					result = "data: " + result + "\n\n" +
						"data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"q\":\"London\"}}}]},\"finishReason\":\"STOP\"}]}\n\n"
				} else if turn == 1 {
					result = strings.Replace(result, `"opaque"}`, `"opaque"},{"functionCall":{"name":"lookup","args":{"q":"London"}}}`, 1)
				}
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {contentType}}, Body: io.NopCloser(strings.NewReader(result))}, nil
			})
			router := newRouter(config{APIKey: "gateway-key"}, vertex)
			input := chatRequest{
				Model: "gemini-4-hydrogen", Stream: streaming, ToolChoice: json.RawMessage(`"required"`),
				Messages: []chatMessage{{Role: "user", Content: json.RawMessage(`"Look up Paris and London"`)}},
				Tools:    []chatTool{{Type: "function", Function: functionDeclaration{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}}},
			}
			send := func() string {
				payload, err := json.Marshal(input)
				if err != nil {
					t.Fatal(err)
				}
				req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(payload))
				req.Header.Set("Authorization", "Bearer gateway-key")
				w := httptest.NewRecorder()
				router.ServeHTTP(w, req)
				if w.Code != 200 {
					t.Fatalf("unexpected response %d %s", w.Code, w.Body.String())
				}
				return w.Body.String()
			}
			first := send()
			var calls []chatToolCall
			if streaming {
				if !strings.HasSuffix(first, "data: [DONE]\n\n") || !strings.Contains(first, `"finish_reason":"tool_calls"`) {
					t.Fatalf("incomplete SSE: %s", first)
				}
				for _, line := range strings.Split(first, "\n") {
					payload, ok := strings.CutPrefix(line, "data: ")
					if !ok || payload == "[DONE]" {
						continue
					}
					var chunk completion
					if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
						t.Fatal(err)
					}
					calls = append(calls, chunk.Choices[0].Delta.ToolCalls...)
				}
			} else {
				var reply completion
				if err := json.Unmarshal([]byte(first), &reply); err != nil {
					t.Fatal(err)
				}
				calls = reply.Choices[0].Message.ToolCalls
				if *reply.Choices[0].Finish != "tool_calls" {
					t.Fatalf("incorrect finish reason: %s", first)
				}
			}
			if len(calls) != 2 || calls[0].Extra == nil || calls[0].Extra.Google.Signature != "opaque" ||
				calls[0].ID != "native-id" || calls[1].ID == "" {
				t.Fatalf("incorrect tool calls: %s", first)
			}
			if streaming && (calls[0].Index == nil || *calls[0].Index != 0 || calls[1].Index == nil || *calls[1].Index != 1) {
				t.Fatalf("incorrect streaming tool indices: %s", first)
			}
			input.Stream = false
			input.Messages = append(input.Messages,
				chatMessage{Role: "assistant", Content: json.RawMessage("null"), ToolCalls: calls},
				chatMessage{Role: "tool", ToolCallID: calls[1].ID, Content: json.RawMessage(`"London"`)},
				chatMessage{Role: "tool", ToolCallID: calls[0].ID, Content: json.RawMessage(`"{\"status\":\"ok\"}"`)},
			)
			if second := send(); !strings.Contains(second, `"content":"Done"`) || turn != 2 {
				t.Fatalf("incorrect follow-up: %s", second)
			}
		})
	}
}

type transport func(*http.Request) (*http.Response, error)

func (f transport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestChat(t *testing.T) {
	for _, model := range append(append([]string{}, advertisedModels...), "gemini-future-preview") {
		for _, streaming := range []bool{false, true} {
			t.Run(model+map[bool]string{true: "/stream", false: "/json"}[streaming], func(t *testing.T) {
				project := "trial-project"
				vertex := newVertex(project, "global", "vertex-key")
				vertex.Client.Transport = transport(func(r *http.Request) (*http.Response, error) {
					action := "generateContent"
					if streaming {
						action = "streamGenerateContent?alt=sse"
					}
					want := "https://aiplatform.googleapis.com/v1/projects/" + project + "/locations/global/publishers/google/models/" + model + ":" + action
					if r.URL.String() != want || r.Header.Get("x-goog-api-key") != "vertex-key" || r.Header.Get("Authorization") != "" {
						t.Fatalf("incorrect Vertex URL or authentication")
					}
					var body vertexRequest
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Fatal(err)
					}
					if body.Contents[1].Role != "model" || *body.System.Parts[0].Text != "Be brief" || *body.Config.MaxOutputTokens != 32 {
						t.Fatalf("unexpected Vertex body: %+v", body)
					}
					result := `{"candidates":[{"content":{"parts":[{"thought":true,"text":"hidden"},{"text":"OK"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"thoughtsTokenCount":3,"totalTokenCount":6}}`
					contentType := "application/json"
					if streaming {
						contentType = "text/event-stream"
						result = "data: " + result + "\r\n\r\n"
					}
					return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {contentType}}, Body: io.NopCloser(strings.NewReader(result))}, nil
				})
				router := newRouter(config{APIKey: "gateway-key"}, vertex)
				body := `{"model":"google/` + model + `","messages":[{"role":"system","content":"Be brief"},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"},{"role":"user","content":"Say OK"}],"max_tokens":32,"stream_options":{"include_usage":true}`
				if streaming {
					body += `,"stream":true`
				}
				req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body+"}"))
				req.Header.Set("Authorization", "Bearer gateway-key")
				w := httptest.NewRecorder()
				router.ServeHTTP(w, req)
				got := w.Body.String()
				if w.Code != 200 || !strings.Contains(got, `"content":"OK"`) || !strings.Contains(got, `"completion_tokens":4`) ||
					strings.Contains(got, "hidden") || streaming != strings.Contains(got, "data: [DONE]") {
					t.Fatalf("unexpected response %d %s", w.Code, got)
				}
			})
		}
	}
}

func TestAuthenticationAndInvalidModelPath(t *testing.T) {
	router := newRouter(config{APIKey: "gateway-key"}, vertexClient{})
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != 401 {
		t.Fatalf("unauthenticated response: %d", w.Code)
	}
	req = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"../other"}`))
	req.Header.Set("Authorization", "Bearer gateway-key")
	w = httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != 400 {
		t.Fatalf("invalid model response: %d", w.Code)
	}
}

func TestNativePassthrough(t *testing.T) {
	const payload = `{"contents":[{"parts":[{"text":"test","thoughtSignature":"opaque"}]}],"tools":[{"googleSearch":{}}]}`
	const upstreamBody = `data: {"futureField":{"value":1}}` + "\n\n"
	upstream := newVertex("trial-project", "global", "vertex-key")
	upstream.Client.Transport = transport(func(r *http.Request) (*http.Response, error) {
		want := "https://aiplatform.googleapis.com/v1beta1/projects/trial-project/locations/global/publishers/google/models/gemini-future-preview:streamGenerateContent?alt=sse"
		if r.URL.String() != want || r.Header.Get("x-goog-api-key") != "vertex-key" ||
			r.Header.Get("Authorization") != "" || r.Header.Get("x-goog-user-project") != "" {
			t.Fatal("incorrect upstream URL or credentials")
		}
		body, err := io.ReadAll(r.Body)
		if err != nil || string(body) != payload {
			t.Fatal("native body was changed")
		}
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(upstreamBody))}, nil
	})
	router := newRouter(config{APIKey: "gateway-key"}, upstream)
	req := httptest.NewRequest("POST", "/vertex/v1beta1/models/gemini-future-preview:streamGenerateContent?alt=sse&key=override", strings.NewReader(payload))
	req.Header.Set("Authorization", "Bearer gateway-key")
	req.Header.Set("x-goog-api-key", "override")
	req.Header.Set("x-goog-user-project", "other-project")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != 200 || w.Header().Get("Content-Type") != "text/event-stream" || w.Body.String() != upstreamBody {
		t.Fatalf("unexpected native response %d %s", w.Code, w.Body.String())
	}
}
