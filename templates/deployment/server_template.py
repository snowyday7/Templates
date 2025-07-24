"""服务器配置模板

提供完整的服务器配置功能，包括：
- Nginx配置生成
- Apache配置生成
- SSL/TLS配置
- 负载均衡配置
- 反向代理配置
- 静态文件服务
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ServerType(str, Enum):
    """服务器类型"""
    NGINX = "nginx"
    APACHE = "apache"
    CADDY = "caddy"


class LoadBalanceMethod(str, Enum):
    """负载均衡方法"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONN = "least_conn"
    IP_HASH = "ip_hash"
    WEIGHTED = "weighted"


class SSLProvider(str, Enum):
    """SSL提供商"""
    LETS_ENCRYPT = "lets_encrypt"
    SELF_SIGNED = "self_signed"
    CUSTOM = "custom"


class UpstreamServer(BaseModel):
    """上游服务器配置"""
    host: str
    port: int
    weight: int = 1
    max_fails: int = 3
    fail_timeout: str = "30s"
    backup: bool = False
    down: bool = False
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @property
    def address(self) -> str:
        """获取服务器地址"""
        return f"{self.host}:{self.port}"


class SSLConfig(BaseModel):
    """SSL配置"""
    enabled: bool = True
    provider: SSLProvider = SSLProvider.LETS_ENCRYPT
    
    # 证书文件路径
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    
    # Let's Encrypt配置
    email: Optional[str] = None
    
    # SSL设置
    protocols: List[str] = Field(default_factory=lambda: ["TLSv1.2", "TLSv1.3"])
    ciphers: Optional[str] = None
    prefer_server_ciphers: bool = True
    
    # HSTS配置
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000
    hsts_include_subdomains: bool = True
    
    # OCSP配置
    ocsp_stapling: bool = True
    
    @validator('email')
    def validate_email(cls, v, values):
        if values.get('provider') == SSLProvider.LETS_ENCRYPT and not v:
            raise ValueError('Email is required for Let\'s Encrypt')
        return v


class CacheConfig(BaseModel):
    """缓存配置"""
    enabled: bool = True
    
    # 静态文件缓存
    static_cache_duration: str = "1y"
    static_file_extensions: List[str] = Field(
        default_factory=lambda: ["css", "js", "png", "jpg", "jpeg", "gif", "ico", "svg"]
    )
    
    # 动态内容缓存
    dynamic_cache_enabled: bool = False
    dynamic_cache_duration: str = "1h"
    
    # 缓存控制头
    cache_control_headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "Cache-Control": "public, max-age=31536000",
            "Expires": "1y"
        }
    )


class SecurityConfig(BaseModel):
    """安全配置"""
    # 基本安全头
    security_headers: bool = True
    
    # XSS保护
    xss_protection: bool = True
    
    # 内容类型嗅探保护
    content_type_nosniff: bool = True
    
    # 点击劫持保护
    frame_options: str = "DENY"
    
    # CSP配置
    csp_enabled: bool = True
    csp_policy: str = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    
    # 隐藏服务器信息
    hide_server_tokens: bool = True
    
    # 限流配置
    rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: str = "1m"
    
    # IP白名单/黑名单
    ip_whitelist: List[str] = Field(default_factory=list)
    ip_blacklist: List[str] = Field(default_factory=list)
    
    # 基本认证
    basic_auth_enabled: bool = False
    basic_auth_file: Optional[str] = None


class CompressionConfig(BaseModel):
    """压缩配置"""
    enabled: bool = True
    
    # 压缩类型
    mime_types: List[str] = Field(
        default_factory=lambda: [
            "text/plain",
            "text/css",
            "text/xml",
            "text/javascript",
            "application/javascript",
            "application/json",
            "application/xml",
            "application/rss+xml",
            "application/atom+xml",
            "image/svg+xml"
        ]
    )
    
    # 压缩级别
    compression_level: int = 6
    
    # 最小压缩文件大小
    min_length: int = 1000
    
    @validator('compression_level')
    def validate_compression_level(cls, v):
        if not 1 <= v <= 9:
            raise ValueError('Compression level must be between 1 and 9')
        return v


class VirtualHostConfig(BaseModel):
    """虚拟主机配置"""
    server_name: str
    document_root: Optional[str] = None
    
    # 监听配置
    listen_port: int = 80
    listen_ssl_port: int = 443
    
    # SSL配置
    ssl: Optional[SSLConfig] = None
    
    # 上游服务器
    upstream_servers: List[UpstreamServer] = Field(default_factory=list)
    upstream_name: Optional[str] = None
    load_balance_method: LoadBalanceMethod = LoadBalanceMethod.ROUND_ROBIN
    
    # 代理配置
    proxy_enabled: bool = False
    proxy_pass: Optional[str] = None
    proxy_headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "Host": "$host",
            "X-Real-IP": "$remote_addr",
            "X-Forwarded-For": "$proxy_add_x_forwarded_for",
            "X-Forwarded-Proto": "$scheme"
        }
    )
    
    # 静态文件配置
    static_locations: List[Dict[str, str]] = Field(default_factory=list)
    
    # 重定向配置
    redirects: List[Dict[str, str]] = Field(default_factory=list)
    
    # 自定义配置
    custom_config: List[str] = Field(default_factory=list)
    
    # 日志配置
    access_log: str = "/var/log/nginx/access.log"
    error_log: str = "/var/log/nginx/error.log"
    
    @validator('listen_port')
    def validate_listen_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Listen port must be between 1 and 65535')
        return v
    
    @validator('listen_ssl_port')
    def validate_ssl_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('SSL port must be between 1 and 65535')
        return v


class ServerConfig(BaseModel):
    """服务器配置"""
    server_type: ServerType = ServerType.NGINX
    
    # 全局配置
    worker_processes: str = "auto"
    worker_connections: int = 1024
    
    # 虚拟主机
    virtual_hosts: List[VirtualHostConfig]
    
    # 全局安全配置
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # 全局缓存配置
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    # 全局压缩配置
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    
    # 日志格式
    log_format: str = '$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent"'
    
    # 自定义配置
    custom_global_config: List[str] = Field(default_factory=list)


class ServerGenerator:
    """服务器配置生成器"""
    
    def __init__(self, output_dir: str = "server-configs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_nginx_config(self, config: ServerConfig) -> str:
        """生成Nginx配置"""
        nginx_config = f"""# Nginx配置文件
# 自动生成，请勿手动修改

user nginx;
worker_processes {config.worker_processes};
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {{
    worker_connections {config.worker_connections};
    use epoll;
    multi_accept on;
}}

http {{
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # 日志格式
    log_format main '{config.log_format}';
    
    # 基本设置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
"""
        
        # 安全配置
        if config.security.hide_server_tokens:
            nginx_config += "    server_tokens off;\n"
        
        # 压缩配置
        if config.compression.enabled:
            nginx_config += f"""
    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length {config.compression.min_length};
    gzip_comp_level {config.compression.compression_level};
    gzip_types
        {' '.join(config.compression.mime_types)};
"""
        
        # 限流配置
        if config.security.rate_limiting:
            nginx_config += f"""
    # 限流配置
    limit_req_zone $binary_remote_addr zone=api:10m rate={config.security.rate_limit_requests}r/{config.security.rate_limit_window};
"""
        
        # 自定义全局配置
        if config.custom_global_config:
            nginx_config += "\n    # 自定义全局配置\n"
            for custom_line in config.custom_global_config:
                nginx_config += f"    {custom_line}\n"
        
        # 上游服务器配置
        for vhost in config.virtual_hosts:
            if vhost.upstream_servers and vhost.upstream_name:
                nginx_config += f"""
    # 上游服务器: {vhost.upstream_name}
    upstream {vhost.upstream_name} {{
"""
                
                if vhost.load_balance_method != LoadBalanceMethod.ROUND_ROBIN:
                    nginx_config += f"        {vhost.load_balance_method.value};\n"
                
                for upstream in vhost.upstream_servers:
                    server_line = f"        server {upstream.address}"
                    if upstream.weight != 1:
                        server_line += f" weight={upstream.weight}"
                    if upstream.max_fails != 3:
                        server_line += f" max_fails={upstream.max_fails}"
                    if upstream.fail_timeout != "30s":
                        server_line += f" fail_timeout={upstream.fail_timeout}"
                    if upstream.backup:
                        server_line += " backup"
                    if upstream.down:
                        server_line += " down"
                    server_line += ";\n"
                    nginx_config += server_line
                
                nginx_config += "    }\n"
        
        # 虚拟主机配置
        for vhost in config.virtual_hosts:
            nginx_config += self._generate_nginx_vhost(vhost, config)
        
        nginx_config += "}\n"
        return nginx_config
    
    def _generate_nginx_vhost(self, vhost: VirtualHostConfig, global_config: ServerConfig) -> str:
        """生成Nginx虚拟主机配置"""
        vhost_config = f"""
    # 虚拟主机: {vhost.server_name}
    server {{
        listen {vhost.listen_port};
        server_name {vhost.server_name};
        
        access_log {vhost.access_log} main;
        error_log {vhost.error_log};
"""
        
        # SSL配置
        if vhost.ssl and vhost.ssl.enabled:
            vhost_config += f"""
        listen {vhost.listen_ssl_port} ssl http2;
        
        # SSL证书
        ssl_certificate {vhost.ssl.cert_file or '/etc/ssl/certs/server.crt'};
        ssl_certificate_key {vhost.ssl.key_file or '/etc/ssl/private/server.key'};
        
        # SSL设置
        ssl_protocols {' '.join(vhost.ssl.protocols)};
        ssl_prefer_server_ciphers {('on' if vhost.ssl.prefer_server_ciphers else 'off')};
"""
            
            if vhost.ssl.ciphers:
                vhost_config += f"        ssl_ciphers {vhost.ssl.ciphers};\n"
            
            if vhost.ssl.hsts_enabled:
                hsts_header = f"max-age={vhost.ssl.hsts_max_age}"
                if vhost.ssl.hsts_include_subdomains:
                    hsts_header += "; includeSubDomains"
                vhost_config += f"        add_header Strict-Transport-Security \"{hsts_header}\" always;\n"
            
            if vhost.ssl.ocsp_stapling:
                vhost_config += "        ssl_stapling on;\n        ssl_stapling_verify on;\n"
        
        # 安全头
        if global_config.security.security_headers:
            if global_config.security.xss_protection:
                vhost_config += "        add_header X-XSS-Protection \"1; mode=block\" always;\n"
            
            if global_config.security.content_type_nosniff:
                vhost_config += "        add_header X-Content-Type-Options \"nosniff\" always;\n"
            
            if global_config.security.frame_options:
                vhost_config += f"        add_header X-Frame-Options \"{global_config.security.frame_options}\" always;\n"
            
            if global_config.security.csp_enabled:
                vhost_config += f"        add_header Content-Security-Policy \"{global_config.security.csp_policy}\" always;\n"
        
        # 限流
        if global_config.security.rate_limiting:
            vhost_config += "        limit_req zone=api burst=20 nodelay;\n"
        
        # IP访问控制
        for ip in global_config.security.ip_blacklist:
            vhost_config += f"        deny {ip};\n"
        
        for ip in global_config.security.ip_whitelist:
            vhost_config += f"        allow {ip};\n"
        
        if global_config.security.ip_whitelist:
            vhost_config += "        deny all;\n"
        
        # 基本认证
        if global_config.security.basic_auth_enabled and global_config.security.basic_auth_file:
            vhost_config += f"        auth_basic \"Restricted\";\n"
            vhost_config += f"        auth_basic_user_file {global_config.security.basic_auth_file};\n"
        
        # 静态文件位置
        for location in vhost.static_locations:
            vhost_config += f"""
        location {location['path']} {{
            root {location['root']};
            expires {global_config.cache.static_cache_duration};
            add_header Cache-Control \"public, immutable\";
        }}
"""
        
        # 代理配置
        if vhost.proxy_enabled:
            proxy_pass = vhost.proxy_pass
            if vhost.upstream_name:
                proxy_pass = f"http://{vhost.upstream_name}"
            
            vhost_config += f"""
        location / {{
            proxy_pass {proxy_pass};
"""
            
            for header_name, header_value in vhost.proxy_headers.items():
                vhost_config += f"            proxy_set_header {header_name} {header_value};\n"
            
            vhost_config += "        }\n"
        
        # 文档根目录
        elif vhost.document_root:
            vhost_config += f"""
        location / {{
            root {vhost.document_root};
            index index.html index.htm;
            try_files $uri $uri/ =404;
        }}
"""
        
        # 重定向
        for redirect in vhost.redirects:
            vhost_config += f"        rewrite {redirect['from']} {redirect['to']} {redirect.get('type', 'permanent')};\n"
        
        # 自定义配置
        if vhost.custom_config:
            vhost_config += "\n        # 自定义配置\n"
            for custom_line in vhost.custom_config:
                vhost_config += f"        {custom_line}\n"
        
        vhost_config += "    }\n"
        
        # HTTP到HTTPS重定向
        if vhost.ssl and vhost.ssl.enabled:
            vhost_config += f"""
    # HTTP到HTTPS重定向
    server {{
        listen {vhost.listen_port};
        server_name {vhost.server_name};
        return 301 https://$server_name$request_uri;
    }}
"""
        
        return vhost_config
    
    def generate_apache_config(self, config: ServerConfig) -> str:
        """生成Apache配置"""
        apache_config = """# Apache配置文件
# 自动生成，请勿手动修改

# 基本设置
ServerRoot "/etc/apache2"
PidFile /var/run/apache2.pid
Timeout 300
KeepAlive On
MaxKeepAliveRequests 100
KeepAliveTimeout 5

# 模块加载
LoadModule rewrite_module modules/mod_rewrite.so
LoadModule ssl_module modules/mod_ssl.so
LoadModule headers_module modules/mod_headers.so
"""
        
        if config.compression.enabled:
            apache_config += "LoadModule deflate_module modules/mod_deflate.so\n"
        
        # 压缩配置
        if config.compression.enabled:
            apache_config += f"""
# 压缩配置
<IfModule mod_deflate.c>
    SetOutputFilter DEFLATE
    SetEnvIfNoCase Request_URI \\
        \\.(?:gif|jpe?g|png)$ no-gzip dont-vary
    SetEnvIfNoCase Request_URI \\
        \\.(?:exe|t?gz|zip|bz2|sit|rar)$ no-gzip dont-vary
    
    AddOutputFilterByType DEFLATE {' '.join(config.compression.mime_types)}
</IfModule>
"""
        
        # 虚拟主机配置
        for vhost in config.virtual_hosts:
            apache_config += self._generate_apache_vhost(vhost, config)
        
        return apache_config
    
    def _generate_apache_vhost(self, vhost: VirtualHostConfig, global_config: ServerConfig) -> str:
        """生成Apache虚拟主机配置"""
        vhost_config = f"""
# 虚拟主机: {vhost.server_name}
<VirtualHost *:{vhost.listen_port}>
    ServerName {vhost.server_name}
    
    ErrorLog {vhost.error_log}
    CustomLog {vhost.access_log} combined
"""
        
        # 文档根目录
        if vhost.document_root:
            vhost_config += f"    DocumentRoot {vhost.document_root}\n"
        
        # 代理配置
        if vhost.proxy_enabled and vhost.proxy_pass:
            vhost_config += f"""
    ProxyPreserveHost On
    ProxyPass / {vhost.proxy_pass}/
    ProxyPassReverse / {vhost.proxy_pass}/
"""
        
        # 安全头
        if global_config.security.security_headers:
            vhost_config += "\n    # 安全头\n"
            if global_config.security.xss_protection:
                vhost_config += "    Header always set X-XSS-Protection \"1; mode=block\"\n"
            
            if global_config.security.content_type_nosniff:
                vhost_config += "    Header always set X-Content-Type-Options \"nosniff\"\n"
            
            if global_config.security.frame_options:
                vhost_config += f"    Header always set X-Frame-Options \"{global_config.security.frame_options}\"\n"
        
        # SSL配置
        if vhost.ssl and vhost.ssl.enabled:
            vhost_config += f"""
    
    # SSL配置
    SSLEngine on
    SSLCertificateFile {vhost.ssl.cert_file or '/etc/ssl/certs/server.crt'}
    SSLCertificateKeyFile {vhost.ssl.key_file or '/etc/ssl/private/server.key'}
    SSLProtocol {' '.join(vhost.ssl.protocols)}
"""
            
            if vhost.ssl.hsts_enabled:
                hsts_header = f"max-age={vhost.ssl.hsts_max_age}"
                if vhost.ssl.hsts_include_subdomains:
                    hsts_header += "; includeSubDomains"
                vhost_config += f"    Header always set Strict-Transport-Security \"{hsts_header}\"\n"
        
        vhost_config += "</VirtualHost>\n"
        return vhost_config
    
    def generate_docker_compose_with_nginx(self, app_name: str, 
                                          app_image: str,
                                          domain: str,
                                          ssl_email: Optional[str] = None) -> str:
        """生成包含Nginx的Docker Compose配置"""
        compose_config = f"""version: '3.8'

services:
  {app_name}:
    image: {app_image}
    restart: unless-stopped
    environment:
      - NODE_ENV=production
    networks:
      - app-network
  
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - {app_name}
    networks:
      - app-network
"""
        
        if ssl_email:
            compose_config += f"""
  
  certbot:
    image: certbot/certbot
    volumes:
      - ./ssl:/etc/letsencrypt
      - ./certbot-var:/var/lib/letsencrypt
    command: certonly --webroot --webroot-path=/var/www/certbot --email {ssl_email} --agree-tos --no-eff-email -d {domain}
"""
        
        compose_config += """

networks:
  app-network:
    driver: bridge
"""
        
        return compose_config
    
    def save_config(self, config: ServerConfig, filename: Optional[str] = None) -> None:
        """保存服务器配置"""
        if not filename:
            filename = f"{config.server_type.value}.conf"
        
        if config.server_type == ServerType.NGINX:
            content = self.generate_nginx_config(config)
        elif config.server_type == ServerType.APACHE:
            content = self.generate_apache_config(config)
        else:
            raise ValueError(f"Unsupported server type: {config.server_type}")
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def save_docker_compose(self, app_name: str, app_image: str, 
                           domain: str, ssl_email: Optional[str] = None,
                           filename: str = "docker-compose.yml") -> None:
        """保存Docker Compose配置"""
        content = self.generate_docker_compose_with_nginx(
            app_name, app_image, domain, ssl_email
        )
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


# 全局生成器
server_generator = ServerGenerator()


# 便捷函数
def generate_nginx_config(config: ServerConfig) -> str:
    """生成Nginx配置"""
    return server_generator.generate_nginx_config(config)


def generate_apache_config(config: ServerConfig) -> str:
    """生成Apache配置"""
    return server_generator.generate_apache_config(config)


def create_reverse_proxy_config(app_name: str, domain: str, 
                               upstream_servers: List[UpstreamServer],
                               ssl_enabled: bool = True,
                               ssl_email: Optional[str] = None) -> ServerConfig:
    """创建反向代理配置"""
    ssl_config = None
    if ssl_enabled:
        ssl_config = SSLConfig(
            enabled=True,
            provider=SSLProvider.LETS_ENCRYPT if ssl_email else SSLProvider.SELF_SIGNED,
            email=ssl_email
        )
    
    vhost = VirtualHostConfig(
        server_name=domain,
        upstream_servers=upstream_servers,
        upstream_name=f"{app_name}-upstream",
        proxy_enabled=True,
        ssl=ssl_config
    )
    
    return ServerConfig(
        server_type=ServerType.NGINX,
        virtual_hosts=[vhost]
    )


# 使用示例
if __name__ == "__main__":
    # 创建生成器
    generator = ServerGenerator()
    
    # 创建上游服务器
    upstream_servers = [
        UpstreamServer(host="127.0.0.1", port=8001),
        UpstreamServer(host="127.0.0.1", port=8002),
        UpstreamServer(host="127.0.0.1", port=8003)
    ]
    
    # 创建反向代理配置
    config = create_reverse_proxy_config(
        app_name="my-python-app",
        domain="api.example.com",
        upstream_servers=upstream_servers,
        ssl_enabled=True,
        ssl_email="admin@example.com"
    )
    
    # 保存Nginx配置
    generator.save_config(config, "nginx.conf")
    
    # 生成Docker Compose配置
    generator.save_docker_compose(
        app_name="my-python-app",
        app_image="my-python-app:latest",
        domain="api.example.com",
        ssl_email="admin@example.com"
    )
    
    print("Server configurations generated successfully!")