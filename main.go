package main

import (
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}
	server := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           newRouter(cfg, newVertex(cfg.Project, cfg.Location, cfg.VertexKey)),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}
	log.Printf("vertex2api listening on %s", server.Addr)
	log.Fatal(server.ListenAndServe())
}

var resourceID = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

type config struct {
	Port, Project, Location, VertexKey, APIKey string
}

func loadConfig() (config, error) {
	cfg := config{
		Port:      os.Getenv("PORT"),
		Project:   strings.TrimSpace(os.Getenv("VERTEX_PROJECT_ID")),
		Location:  strings.TrimSpace(os.Getenv("VERTEX_LOCATION")),
		VertexKey: strings.TrimSpace(os.Getenv("VERTEX_API_KEY")),
		APIKey:    strings.TrimSpace(os.Getenv("API_KEY")),
	}
	if cfg.Port == "" {
		cfg.Port = "8080"
	}
	if cfg.Location == "" {
		cfg.Location = "global"
	}
	if !resourceID.MatchString(cfg.Project) || !resourceID.MatchString(cfg.Location) {
		return config{}, fmt.Errorf("set a valid VERTEX_PROJECT_ID and VERTEX_LOCATION")
	}
	if cfg.VertexKey == "" || cfg.APIKey == "" || cfg.VertexKey == cfg.APIKey {
		return config{}, fmt.Errorf("set distinct VERTEX_API_KEY and API_KEY values")
	}
	return cfg, nil
}

func newRouter(cfg config, upstream vertexClient) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		models := make([]map[string]any, 0, len(advertisedModels))
		for _, model := range advertisedModels {
			models = append(models, map[string]any{"id": model, "object": "model", "created": 0, "owned_by": "google"})
		}
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": models})
	})
	mux.HandleFunc("POST /v1/chat/completions", handler{vertex: upstream}.chat)
	mux.HandleFunc("POST /vertex/{version}/models/{action}", upstream.native)
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
