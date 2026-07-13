searxng_port := env("SEARXNG_PORT", "18888")
searxng_host := env("SEARXNG_HOST", "127.0.0.1")
searxng_dir := env("SEARXNG_DIR", if os() == "macos" { "~/Library/Application Support/fedotmas/searxng" } else { "~/.local/share/fedotmas/searxng" })
searxng_container := env("SEARXNG_CONTAINER", "searxng-core")
searxng_volume := env("SEARXNG_VOLUME", "searxng-cache")
podman_machine := env("PODMAN_MACHINE", "podman-machine-default")

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
#
# Runs as a single rootless podman container -- no compose, since `podman
# compose` only shells out to an external provider (podman-compose /
# docker-compose) that may not be installed.

# Ensure the podman VM is up (macOS/Windows only; on Linux podman runs natively).
[private]
podman-machine:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ "$(uname -s)" != "Darwin" ]; then
        exit 0
    fi

    if ! command -v podman >/dev/null 2>&1; then
        echo "ERROR: podman is not installed (brew install podman)"
        exit 1
    fi

    if podman machine inspect {{ podman_machine }} >/dev/null 2>&1; then
        if [ "$(podman machine inspect {{ podman_machine }} --format '{{{{.State}}')" != "running" ]; then
            echo "Starting podman machine '{{ podman_machine }}'..."
            podman machine start {{ podman_machine }}
        fi
    else
        echo "Initializing podman machine '{{ podman_machine }}'..."
        podman machine init --now {{ podman_machine }}
    fi

searxng-install: podman-machine
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    mkdir -p "$dir"

    podman rm -f {{ searxng_container }} 2>/dev/null || true

    # Rootless podman writes into bind mounts as a subuid-mapped user, so a
    # previous run can leave the config dir unwritable by $USER.
    if ! touch "$dir/.fedotmas-write-test" 2>/dev/null; then
        echo "Fixing ownership for $dir"
        podman unshare chown -R 0:0 "$dir" 2>/dev/null || sudo chown -R "$USER:$USER" "$dir"
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
        printf '%s\n' '# SearXNG defaults to a 3s engine timeout. From inside the podman VM the'
        printf '%s\n' '# scraping engines routinely need longer, and a timed-out engine returns'
        printf '%s\n' '# nothing -- a benchmark question then silently researches an empty result'
        printf '%s\n' '# set. Give them room.'
        printf '%s\n' 'outgoing:'
        printf '%s\n' '  request_timeout: 10.0'
        printf '%s\n' '  max_request_timeout: 15.0'
        printf '%s\n' '  pool_connections: 100'
        printf '%s\n' '  pool_maxsize: 20'
        printf '%s\n' ''
        printf '%s\n' '# The stock general-category engines (duckduckgo, google cse, startpage) all'
        printf '%s\n' '# throttle or CAPTCHA a benchmark-rate client, and a search that returns'
        printf '%s\n' '# nothing is indistinguishable from a question with no answer. Widen the pool'
        printf '%s\n' '# with engines that tolerate volume, so a blocked engine is survivable.'
        printf '%s\n' 'engines:'
        printf '%s\n' '  - name: mojeek'
        printf '%s\n' '    disabled: false'
        printf '%s\n' '  - name: qwant'
        printf '%s\n' '    disabled: false'
        printf '%s\n' '  - name: wikipedia'
        printf '%s\n' '    disabled: false'
        printf '%s\n' '  - name: wikidata'
        printf '%s\n' '    disabled: false'
        printf '%s\n' '  - name: brave'
        printf '%s\n' '    disabled: false'
        printf '%s\n' '  - name: bing'
        printf '%s\n' '    disabled: false'
        printf '%s\n' '  - name: ahmia'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: torch'
        printf '%s\n' '    disabled: true'
    } > "$dir/core-config/settings.yml"

    rm -f "$dir/core-config/limiter.toml"

    echo "SearXNG installed at $dir"
    echo "JSON API will be available at: http://localhost:{{ searxng_port }}/search?q=test&format=json"
    echo "Set SEARXNG_URL=http://localhost:{{ searxng_port }} in .env"

searxng-start: podman-machine
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ ! -f "$dir/core-config/settings.yml" ]; then
        just searxng-install
    fi

    podman rm -f {{ searxng_container }} 2>/dev/null || true
    podman volume create {{ searxng_volume }} >/dev/null 2>&1 || true

    podman run -d \
        --name {{ searxng_container }} \
        --restart unless-stopped \
        -p "{{ searxng_host }}:{{ searxng_port }}:8080" \
        -v "$dir/core-config:/etc/searxng:Z" \
        -v "{{ searxng_volume }}:/var/cache/searxng" \
        docker.io/searxng/searxng:latest

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
    podman logs --tail 160 {{ searxng_container }}
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

    podman rm -f {{ searxng_container }} 2>/dev/null || true
    podman volume rm -f {{ searxng_volume }} 2>/dev/null || true

    if [ -n "$dir" ] && [ "$dir" != "/" ] && [ "$dir" != "$HOME" ]; then
        if [ -d "$dir" ]; then
            podman unshare chown -R 0:0 "$dir" 2>/dev/null || sudo chown -R "$USER:$USER" "$dir" 2>/dev/null || true
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

    podman rm -f {{ searxng_container }} 2>/dev/null || true
    echo "SearXNG stopped"

searxng-status:
    podman ps -a --filter "name={{ searxng_container }}"

searxng-logs:
    podman logs -f --tail 200 {{ searxng_container }}

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
