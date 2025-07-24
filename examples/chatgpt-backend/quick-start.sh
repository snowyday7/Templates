#!/bin/bash

# ChatGPT Backend 快速启动脚本
# 使用方法: ./quick-start.sh [dev|prod|stop|clean]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_message "错误: Docker未安装，请先安装Docker" "$RED"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_message "错误: Docker Compose未安装，请先安装Docker Compose" "$RED"
        exit 1
    fi
}

# 检查环境变量文件
check_env_file() {
    if [ ! -f ".env" ]; then
        print_message "创建环境配置文件..." "$YELLOW"
        cp .env.example .env
        print_message "请编辑 .env 文件，设置必要的配置（特别是 OPENAI_API_KEY）" "$YELLOW"
        print_message "然后重新运行此脚本" "$YELLOW"
        exit 1
    fi
}

# 开发环境启动
start_dev() {
    print_message "启动开发环境..." "$GREEN"
    check_docker
    check_env_file
    
    # 构建并启动服务
    docker-compose -f docker-compose.dev.yml up --build -d
    
    print_message "等待服务启动..." "$YELLOW"
    sleep 10
    
    # 显示服务状态
    docker-compose -f docker-compose.dev.yml ps
    
    print_message "开发环境启动完成！" "$GREEN"
    print_message "API文档: http://localhost:8000/docs" "$BLUE"
    print_message "数据库管理: http://localhost:8080" "$BLUE"
    print_message "Redis管理: http://localhost:8081" "$BLUE"
    print_message "查看日志: docker-compose -f docker-compose.dev.yml logs -f app" "$BLUE"
}

# 生产环境启动
start_prod() {
    print_message "启动生产环境..." "$GREEN"
    check_docker
    check_env_file
    
    # 构建并启动服务
    docker-compose up --build -d
    
    print_message "等待服务启动..." "$YELLOW"
    sleep 15
    
    # 显示服务状态
    docker-compose ps
    
    print_message "生产环境启动完成！" "$GREEN"
    print_message "API文档: http://localhost:8000/docs" "$BLUE"
    print_message "健康检查: http://localhost:8000/health" "$BLUE"
    print_message "查看日志: docker-compose logs -f app" "$BLUE"
}

# 停止服务
stop_services() {
    print_message "停止服务..." "$YELLOW"
    
    if [ -f "docker-compose.dev.yml" ]; then
        docker-compose -f docker-compose.dev.yml down
    fi
    
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
    fi
    
    print_message "服务已停止" "$GREEN"
}

# 清理资源
clean_all() {
    print_message "清理所有资源..." "$YELLOW"
    
    # 停止服务
    stop_services
    
    # 删除容器和网络
    docker-compose -f docker-compose.dev.yml down --volumes --remove-orphans 2>/dev/null || true
    docker-compose down --volumes --remove-orphans 2>/dev/null || true
    
    # 删除镜像
    docker rmi $(docker images "chatgpt-backend*" -q) 2>/dev/null || true
    
    # 清理未使用的资源
    docker system prune -f
    
    print_message "清理完成" "$GREEN"
}

# 显示帮助信息
show_help() {
    echo "ChatGPT Backend 快速启动脚本"
    echo ""
    echo "使用方法:"
    echo "  ./quick-start.sh dev     - 启动开发环境"
    echo "  ./quick-start.sh prod    - 启动生产环境"
    echo "  ./quick-start.sh stop    - 停止所有服务"
    echo "  ./quick-start.sh clean   - 清理所有资源"
    echo "  ./quick-start.sh help    - 显示此帮助信息"
    echo ""
    echo "开发环境包含:"
    echo "  - API服务 (端口 8000)"
    echo "  - PostgreSQL数据库 (端口 5433)"
    echo "  - Redis缓存 (端口 6380)"
    echo "  - Adminer数据库管理 (端口 8080)"
    echo "  - Redis Commander (端口 8081)"
    echo ""
    echo "生产环境包含:"
    echo "  - API服务 (端口 8000)"
    echo "  - PostgreSQL数据库 (端口 5432)"
    echo "  - Redis缓存 (端口 6379)"
    echo "  - 可选: Nginx反向代理"
    echo "  - 可选: Prometheus + Grafana监控"
}

# 主逻辑
case "${1:-help}" in
    "dev")
        start_dev
        ;;
    "prod")
        start_prod
        ;;
    "stop")
        stop_services
        ;;
    "clean")
        clean_all
        ;;
    "help")
        show_help
        ;;
    *)
        print_message "未知命令: $1" "$RED"
        show_help
        exit 1
        ;;
esac