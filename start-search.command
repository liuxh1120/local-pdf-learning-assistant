#!/bin/zsh
set -e
cd "${0:A:h}"

pause_before_exit() {
  if [[ "${PDF_ASSISTANT_NO_PAUSE:-0}" != "1" ]]; then
    read "?按回车键关闭此窗口..."
  fi
}

if ! command -v docker >/dev/null 2>&1; then
  echo "未安装 Docker Desktop。请先从 https://www.docker.com/products/docker-desktop/ 安装。"
  pause_before_exit
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "正在启动 Docker Desktop..."
  open -a Docker
  for _ in {1..60}; do
    if docker info >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop 尚未就绪。请等待其菜单栏图标显示运行后，再双击本文件。"
  pause_before_exit
  exit 1
fi

echo "正在启动 SearXNG；首次使用需要下载镜像..."
GHCR_IMAGE="ghcr.io/searxng/searxng:latest"
DOCKERHUB_IMAGE="docker.io/searxng/searxng:latest"

if docker image inspect "$GHCR_IMAGE" >/dev/null 2>&1; then
  export SEARXNG_IMAGE="$GHCR_IMAGE"
elif docker image inspect "$DOCKERHUB_IMAGE" >/dev/null 2>&1; then
  export SEARXNG_IMAGE="$DOCKERHUB_IMAGE"
elif docker pull "$GHCR_IMAGE"; then
  export SEARXNG_IMAGE="$GHCR_IMAGE"
elif docker pull "$DOCKERHUB_IMAGE"; then
  export SEARXNG_IMAGE="$DOCKERHUB_IMAGE"
else
  echo "无法从 GHCR 或 Docker Hub 下载镜像。"
  echo "请检查代理后重试；当前推荐镜像：$GHCR_IMAGE"
  pause_before_exit
  exit 1
fi

docker compose -f docker-compose.search.yml up -d

echo "正在检查服务..."
for _ in {1..30}; do
  if curl -fsS --max-time 3 \
    "http://127.0.0.1:8080/search?q=status&format=json" >/dev/null 2>&1; then
    echo "SearXNG 已启动：http://127.0.0.1:8080"
    pause_before_exit
    exit 0
  fi
  sleep 2
done

echo "容器已经启动，但搜索接口暂时不可用。请运行："
echo "docker compose -f docker-compose.search.yml logs"
pause_before_exit
exit 1
