# Build context: webapp/frontend/  (docker build -f Dockerfile .)
# Produces a minimal nginx image serving the compiled React SPA.
# Set VITE_API_URL at build time to point at the backend origin:
#   docker build --build-arg VITE_API_URL=https://api.courtvision.app .

FROM node:20-alpine AS builder

ARG VITE_API_URL=""
ENV VITE_API_URL=${VITE_API_URL}

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# ── Serve with nginx ──────────────────────────────────────────────────────────
FROM nginx:1.27-alpine

# SPA routing: all 404s fall back to index.html so client-side routes work.
COPY --from=builder /app/dist /usr/share/nginx/html
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
    location /assets/ {\n\
        expires 1y;\n\
        add_header Cache-Control "public, immutable";\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
