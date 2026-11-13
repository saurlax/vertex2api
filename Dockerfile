FROM golang:1.26-alpine AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /vertex2api .

FROM alpine:3.23
COPY --from=build /vertex2api /usr/local/bin/vertex2api
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/vertex2api"]
