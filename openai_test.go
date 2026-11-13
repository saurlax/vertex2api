package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
