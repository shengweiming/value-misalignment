# GitHub Push Troubleshooting Behind a VPN

If `git push` fails with something like:

```text
Failed to connect to github.com port 443: Couldn't connect to server
```

but GitHub works in the browser, check whether the issue is Git bypassing the VPN proxy.

## 1. Check whether GitHub is reachable

```bash
curl -I https://github.com
```

If this returns `HTTP/2 200` (or otherwise connects), the network/VPN itself is working.

## 2. Confirm Git is failing at the connection stage

```bash
GIT_CURL_VERBOSE=1 git ls-remote https://github.com/<USER>/<REPO>.git
```

If you see something like:

```text
Trying <IP>:443...
connect ... failed: Operation timed out
```

Git is probably attempting a direct connection instead of using the VPN's proxy.

## 3. Find the macOS proxy used by the VPN

```bash
scutil --proxy
```

Look for:

```text
HTTPEnable : 1
HTTPPort : <PORT>
HTTPProxy : 127.0.0.1

HTTPSEnable : 1
HTTPSPort : <PORT>
HTTPSProxy : 127.0.0.1
```

For example, if the VPN uses `127.0.0.1:9098`:

```bash
git config --global http.proxy http://127.0.0.1:9098
git config --global https.proxy http://127.0.0.1:9098
```

Then retry:

```bash
git push origin main
```

## 4. Remove the proxy afterward

If the VPN's proxy port may change next time, remove the Git-specific configuration once finished:

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### Diagnosis from the original incident

- `curl https://github.com` worked.
- `git` resolved `github.com` correctly but timed out connecting directly to port 443.
- macOS showed the VPN's HTTP/HTTPS proxy at `127.0.0.1:9098`.
- Explicitly configuring Git to use that proxy fixed `git push`.