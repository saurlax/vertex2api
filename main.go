package main

import (
	"log"
	"net/http"
	"time"

	"vertex2api/api"
)

func main() {
	cfg, err := api.LoadConfig()
	if err != nil {
		log.Fatal(err)
	}
	server := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           api.Router(cfg),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}
	log.Printf("vertex2api listening on %s", server.Addr)
	log.Fatal(server.ListenAndServe())
}
