package api

import (
	"crypto/subtle"
	"encoding/json"
	"net/http"

	"vertex2api/providers"
)

func Router(cfg Config) http.Handler {
	return newRouter(cfg, providers.NewVertex(cfg.Project, cfg.Location, cfg.VertexKey))
}

func newRouter(cfg Config, vertex providers.Vertex) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		models := make([]map[string]any, 0, len(providers.Models))
		for _, model := range providers.Models {
			models = append(models, map[string]any{"id": model, "object": "model", "created": 0, "owned_by": "google"})
		}
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": models})
	})
	mux.HandleFunc("POST /v1/chat/completions", handler{vertex: vertex}.chat)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" && subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+cfg.APIKey)) != 1 {
			writeError(w, http.StatusUnauthorized, "invalid gateway credential")
			return
		}
		mux.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func errorBody(status int, message string) map[string]any {
	kind := "invalid_request_error"
	if status >= 500 {
		kind = "api_error"
	}
	return map[string]any{"error": map[string]string{"message": message, "type": kind}}
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorBody(status, message))
}
