package api

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

var resourceID = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

type Config struct {
	Port, Project, Location, VertexKey, APIKey string
}

func LoadConfig() (Config, error) {
	cfg := Config{
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
		return Config{}, fmt.Errorf("set a valid VERTEX_PROJECT_ID and VERTEX_LOCATION")
	}
	if cfg.VertexKey == "" || cfg.APIKey == "" || cfg.VertexKey == cfg.APIKey {
		return Config{}, fmt.Errorf("set distinct VERTEX_API_KEY and API_KEY values")
	}
	return cfg, nil
}
