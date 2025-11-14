# Quick Start: Docker Image Comparison Test

## TL;DR - Run the Test

```bash
# 1. Make sure you have Docker running
docker --version

# 2. Run the comparison script
./test-docker-comparison.sh

# 3. Wait 10-15 minutes for both builds to complete

# 4. Review the results
```

Expected output:
```
Current Image:   ~4.0 GB
Optimized Image: ~600 MB
Size Reduction:  85%

✅ RECOMMENDATION: Use optimized Dockerfile
```

---

## Why This Matters

**Problem**: Your 4GB Docker pull is caused by this line in `cognee-mcp/Dockerfile:55`:
```dockerfile
COPY --from=uv /usr/local /usr/local  # ❌ Copies 3GB of build tools
```

**Solution**: The optimized Dockerfile only copies what's needed:
```dockerfile
COPY --from=builder /build/.venv ./.venv  # ✅ Only 400MB of dependencies
```

---

## What the Test Does

1. **Builds both Dockerfiles** side-by-side
2. **Compares sizes** (expecting 75-85% reduction)
3. **Tests functionality** (both should work identically)
4. **Checks security** (optimized uses non-root user)
5. **Measures build time** (should be similar)

---

## If You Don't Have Time for Full Test

### Quick Manual Comparison

```bash
# Check current image size on Docker Hub
docker pull lvarming/cognee-mcp:latest
docker images lvarming/cognee-mcp:latest

# Build just the optimized version
docker build -t cognee-mcp:optimized -f cognee-mcp/Dockerfile.optimized .

# Compare
docker images | grep cognee-mcp
```

---

## What to Look For in Results

### ✅ Success Indicators

- **Size reduction > 70%** → Huge win
- **Both images pass import test** → Functionally identical
- **Optimized runs as non-root** → Security improvement
- **No errors during build** → Ready for production

### ⚠️ Warning Signs

- **Build failures** → Need to investigate
- **Import errors** → Dependency issue
- **Size reduction < 50%** → Not getting expected benefit

### ❌ Failure Cases

- **Optimized image doesn't start** → Configuration issue
- **Missing dependencies** → Build process problem
- **Size actually larger** → Something very wrong

---

## After the Test: Next Steps

### If Test Passes (Expected)

1. **Update GitHub Actions** to use optimized Dockerfile:
   ```yaml
   # .github/workflows/cognee-mcp-docker.yml, line 109:
   file: ./cognee-mcp/Dockerfile.optimized
   ```

2. **Commit and push** to trigger new build:
   ```bash
   git add .github/workflows/cognee-mcp-docker.yml
   git commit -m "feat: switch to optimized MCP Dockerfile (85% size reduction)"
   git push
   ```

3. **Monitor the GitHub Actions** build
4. **Pull new image** and verify size:
   ```bash
   docker pull lvarming/cognee-mcp:latest
   docker images lvarming/cognee-mcp:latest
   # Should show ~600MB instead of 4GB
   ```

### If Test Reveals Issues

1. **Review build logs**: `build-current.log` and `build-optimized.log`
2. **Check error messages** in test output
3. **Try manual build** with verbose output:
   ```bash
   docker build -f cognee-mcp/Dockerfile.optimized --progress=plain . 2>&1 | tee debug.log
   ```
4. **Report findings** and we can troubleshoot

---

## Troubleshooting

### Issue: "docker: command not found"

**Solution**: Install Docker
```bash
# macOS
brew install --cask docker

# Linux
curl -fsSL https://get.docker.com | sh

# Windows
# Download from https://docker.com
```

### Issue: "Permission denied"

**Solution**: Make script executable
```bash
chmod +x test-docker-comparison.sh
```

### Issue: "No space left on device"

**Solution**: Free up disk space
```bash
# Remove old Docker images
docker system prune -a

# Check space
df -h
```

### Issue: Build fails with "not found" errors

**Solution**: Make sure you're in the repo root directory
```bash
# Both Dockerfiles build from repo root
cd /path/to/cognee

# Then build
docker build -f cognee-mcp/Dockerfile .
docker build -f cognee-mcp/Dockerfile.optimized .
```

---

## Understanding the Results

### Size Breakdown

**Current Image (~4GB)**:
```
/usr/local/           3.2 GB  ❌ Build tools (gcc, clang, uv)
/app/.venv/           0.5 GB  ✅ Python dependencies
/app/src/             0.02 GB ✅ Application code
/usr/lib/             0.3 GB  ❌ System libraries
```

**Optimized Image (~600MB)**:
```
/app/.venv/           0.5 GB  ✅ Python dependencies
/app/src/             0.02 GB ✅ Application code
/usr/lib/ (minimal)   0.08 GB ✅ Runtime libs only
```

### Security Improvements

| Aspect | Current | Optimized |
|--------|---------|-----------|
| User | root (UID 0) | cognee (UID 1000) |
| Build Tools | Included | Removed |
| Shell | /bin/bash | /sbin/nologin |
| Attack Surface | Large | Minimal |

---

## Expected Timeline

- **Setup**: 1 minute (make script executable)
- **Current build**: 5-8 minutes
- **Optimized build**: 5-8 minutes
- **Testing**: 2 minutes
- **Total**: ~15-20 minutes

**Recommendation**: Run during coffee break ☕

---

## Manual Verification Commands

After the test completes, you can manually verify:

```bash
# 1. Check both images exist
docker images | grep cognee-mcp

# 2. Verify Python works in both
docker run --rm cognee-mcp:current python --version
docker run --rm cognee-mcp:optimized python --version

# 3. Check file ownership (should be cognee:cognee for optimized)
docker run --rm cognee-mcp:optimized ls -la /app/

# 4. Verify virtual environment
docker run --rm cognee-mcp:optimized python -c "import sys; print(sys.prefix)"

# 5. Test MCP module import
docker run --rm cognee-mcp:optimized python -c "import cognee_mcp; print('✅')"
```

---

## Questions?

- **How long will this take?** ~15-20 minutes total
- **Will it break anything?** No, just builds test images
- **Do I need to stop running containers?** No
- **Can I cancel mid-test?** Yes, just Ctrl+C
- **What if test fails?** Share the logs and we'll debug

---

## One-Liner Test (If You're in a Hurry)

```bash
docker build -f cognee-mcp/Dockerfile -t test:current . && \
docker build -f cognee-mcp/Dockerfile.optimized -t test:optimized . && \
docker images | grep test && \
docker rmi test:current test:optimized
```

This will:
1. Build both images
2. Show sizes
3. Clean up

Should take ~10-15 minutes.
