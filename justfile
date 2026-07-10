searxng_port := env("SEARXNG_PORT", "18888")
searxng_host := env("SEARXNG_HOST", "127.0.0.1")
searxng_dir := env("SEARXNG_DIR", if os() == "macos" { "~/Library/Application Support/fedotmas/searxng" } else { "~/.local/share/fedotmas/searxng" })

# List discovered benchmarks and systems (only systems with installed deps appear).
available:
    uv run --no-sync python -c "from benchlib.benchmarks import discover; from benchlib.adapters import discover_adapters; print('benchmarks:', list(discover())); print('systems:', discover_adapters())"

# Run experiments (cross-OS); pass any CLI flag. See all with: just run --help
run *ARGS:
    uv run --no-sync python -m benchlib.cli {{ARGS}}

# Build a benchmark's questions.jsonl via its builder (seal_*: uv sync --group benchmarks first).
download NAME:
    uv run --no-sync python -c "from benchlib.benchmarks import load_spec, get_builder; spec = load_spec('{{NAME}}'); get_builder('{{NAME}}').download(spec)"

# SearXNG (shared web-search backend for both systems; point SEARXNG_URL in
# .env at http://localhost:{{ searxng_port }})

searxng-install:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    mkdir -p "$dir"

    if [ -f "$dir/docker-compose.yml" ]; then
        docker compose -f "$dir/docker-compose.yml" down --remove-orphans 2>/dev/null || true
    fi

    docker rm -f searxng-core searxng-valkey searxng 2>/dev/null || true

    if ! touch "$dir/.fedotmas-write-test" 2>/dev/null; then
        echo "Fixing ownership for $dir"
        sudo chown -R "$USER:$USER" "$dir"
    else
        rm -f "$dir/.fedotmas-write-test"
    fi

    mkdir -p "$dir/core-config"

    secret_file="$dir/.searxng_secret"

    if [ ! -f "$secret_file" ]; then
        if command -v openssl >/dev/null 2>&1; then
            openssl rand -hex 32 > "$secret_file"
        elif command -v python3 >/dev/null 2>&1; then
            python3 -c 'import secrets; print(secrets.token_hex(32))' > "$secret_file"
        else
            date +%s | sha256sum | awk '{print $1}' > "$secret_file"
        fi
    fi

    secret="$(cat "$secret_file")"

    {
        printf '%s\n' 'use_default_settings: true'
        printf '%s\n' ''
        printf '%s\n' 'general:'
        printf '%s\n' '  debug: false'
        printf '%s\n' '  instance_name: "FEDOT.MAS SearXNG"'
        printf '%s\n' ''
        printf '%s\n' 'search:'
        printf '%s\n' '  safe_search: 0'
        printf '%s\n' '  autocomplete: ""'
        printf '%s\n' '  formats:'
        printf '%s\n' '    - html'
        printf '%s\n' '    - json'
        printf '%s\n' ''
        printf '%s\n' 'server:'
        printf '%s\n' "  secret_key: \"$secret\""
        printf '%s\n' '  limiter: false'
        printf '%s\n' '  image_proxy: false'
        printf '%s\n' '  method: "GET"'
        printf '%s\n' ''
        printf '%s\n' 'engines:'
        printf '%s\n' '  - name: wikidata'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: ahmia'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: torch'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: google'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: brave'
        printf '%s\n' '    disabled: true'
    } > "$dir/core-config/settings.yml"

    rm -f "$dir/core-config/limiter.toml"

    {
        printf '%s\n' 'services:'
        printf '%s\n' '  searxng:'
        printf '%s\n' '    image: searxng/searxng:latest'
        printf '%s\n' '    container_name: searxng-core'
        printf '%s\n' '    restart: unless-stopped'
        printf '%s\n' '    ports:'
        printf '%s\n' '      - "{{ searxng_host }}:{{ searxng_port }}:8080"'
        printf '%s\n' '    volumes:'
        printf '%s\n' '      - ./core-config:/etc/searxng'
        printf '%s\n' '      - core-data:/var/cache/searxng'
        printf '%s\n' ''
        printf '%s\n' 'volumes:'
        printf '%s\n' '  core-data:'
    } > "$dir/docker-compose.yml"

    echo "SearXNG installed at $dir"
    echo "JSON API will be available at: http://localhost:{{ searxng_port }}/search?q=test&format=json"
    echo "Set SEARXNG_URL=http://localhost:{{ searxng_port }} in .env"

searxng-start:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ ! -f "$dir/docker-compose.yml" ] || [ ! -f "$dir/core-config/settings.yml" ]; then
        just searxng-install
    fi

    cd "$dir"

    docker compose up -d --force-recreate

    echo "SearXNG running at http://localhost:{{ searxng_port }}"
    echo "Checking JSON API..."

    for i in $(seq 1 30); do
        if curl -fsS "http://localhost:{{ searxng_port }}/search?q=test&format=json" | python3 -m json.tool >/dev/null 2>&1; then
            echo "SearXNG JSON API OK"
            exit 0
        fi
        sleep 1
    done

    echo "ERROR: SearXNG started, but JSON API check failed."
    echo ""
    echo "Recent logs:"
    docker compose logs --tail 160 searxng || docker logs searxng-core --tail 160
    exit 1

searxng-restart:
    #!/usr/bin/env bash
    set -euo pipefail

    just searxng-stop
    just searxng-start

searxng-reinstall:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ -d "$dir" ] && [ -f "$dir/docker-compose.yml" ]; then
        docker compose -f "$dir/docker-compose.yml" down --remove-orphans 2>/dev/null || true
    fi

    docker rm -f searxng-core searxng-valkey searxng 2>/dev/null || true

    if [ -n "$dir" ] && [ "$dir" != "/" ] && [ "$dir" != "$HOME" ]; then
        if [ -d "$dir" ]; then
            sudo chown -R "$USER:$USER" "$dir" 2>/dev/null || true
            chmod -R u+rwX "$dir" 2>/dev/null || true
        fi
        rm -rf "$dir"
    else
        echo "ERROR: unsafe searxng_dir: $dir"
        exit 1
    fi

    just searxng-install
    just searxng-start

searxng-stop:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ ! -d "$dir" ] || [ ! -f "$dir/docker-compose.yml" ]; then
        docker rm -f searxng-core searxng-valkey searxng 2>/dev/null || true
        echo "SearXNG is not installed at $dir"
        exit 0
    fi

    cd "$dir"
    docker compose down --remove-orphans

searxng-status:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ ! -f "$dir/docker-compose.yml" ]; then
        echo "SearXNG is not installed at $dir"
        exit 1
    fi

    cd "$dir"
    docker compose ps

searxng-logs:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ -f "$dir/docker-compose.yml" ]; then
        cd "$dir"
        docker compose logs -f --tail 200 searxng
    else
        docker logs -f --tail 200 searxng-core
    fi

searxng-check:
    #!/usr/bin/env bash
    set -euo pipefail

    url="http://localhost:{{ searxng_port }}/search?q=test&format=json"

    echo "Checking $url"

    if command -v python3 >/dev/null 2>&1; then
        response="$(curl -fsS "$url")"
        echo "$response" | python3 -m json.tool >/dev/null
    else
        curl -fsS "$url" >/dev/null
    fi

    echo "SearXNG JSON API OK"
