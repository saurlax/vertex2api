package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/providers"
)

type transport func(*http.Request) (*http.Response, error)

func (f transport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestChat(t *testing.T) {
	for _, model := range providers.Models {
		for _, streaming := range []bool{false, true} {
			t.Run(model+map[bool]string{true: "/stream", false: "/json"}[streaming], func(t *testing.T) {
				project := "trial-project"
				vertex := providers.NewVertex(project, "global", "vertex-key")
				vertex.Client.Transport = transport(func(r *http.Request) (*http.Response, error) {
					action := "generateContent"
					if streaming {
						action = "streamGenerateContent?alt=sse"
					}
					want := "https://aiplatform.googleapis.com/v1/projects/" + project + "/locations/global/publishers/google/models/" + model + ":" + action
					if r.URL.String() != want || r.Header.Get("x-goog-api-key") != "vertex-key" || r.Header.Get("Authorization") != "" {
						t.Fatalf("incorrect Vertex URL or authentication")
					}
					var body providers.Request
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Fatal(err)
					}
					if body.Contents[1].Role != "model" || body.System.Parts[0].Text != "Be brief" || *body.Config.MaxOutputTokens != 32 {
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
				router := newRouter(Config{APIKey: "gateway-key"}, vertex)
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

func TestAuthenticationAndModelAllowlist(t *testing.T) {
	router := newRouter(Config{APIKey: "gateway-key"}, providers.Vertex{})
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != 401 {
		t.Fatalf("unauthenticated response: %d", w.Code)
	}
	req = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"claude"}`))
	req.Header.Set("Authorization", "Bearer gateway-key")
	w = httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != 404 {
		t.Fatalf("disallowed model response: %d", w.Code)
	}
}
