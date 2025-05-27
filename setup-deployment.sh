#!/bin/bash

# Deployment Server Setup Script for Nicomatic Chatbot
# Run this script on your deployment servers (staging/production)

set -e

# Configuration
APP_NAME="nicomatic-chatbot"
APP_DIR="/opt/$APP_NAME"
APP_USER="nicomatic"
DOCKER_COMPOSE_VERSION="2.21.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
        exit 1
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    apt-get update
    apt-get upgrade -y
    apt-get install -y \
        curl \
        wget \
        git \
        unzip \
        htop \
        nano \
        ufw \
        fail2ban \
        logrotate \
        cron \
        rsync \
        nginx-utils \
        openssl
}

# Install Docker
install_docker() {
    log "Installing Docker..."
    
    # Remove old versions
    apt-get remove -y docker docker-engine docker.io containerd runc || true
    
    # Install dependencies
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker GPG key
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Add Docker repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Start and enable Docker
    systemctl start docker
    systemctl enable docker
    
    # Verify installation
    docker --version
    docker compose version
}

# Install NVIDIA Docker (for GPU support)
install_nvidia_docker() {
    log "Installing NVIDIA Docker runtime..."
    
    # Check if NVIDIA GPU is available
    if ! nvidia-smi > /dev/null 2>&1; then
        warning "NVIDIA GPU not detected. Skipping NVIDIA Docker installation."
        return
    fi
    
    # Install NVIDIA Docker
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update
    apt-get install -y nvidia-docker2
    
    # Restart Docker
    systemctl restart docker
    
    log "NVIDIA Docker installed successfully"
}

# Create application user
create_app_user() {
    log "Creating application user: $APP_USER"
    
    if ! id "$APP_USER" &>/dev/null; then
        useradd -r -s /bin/bash -m -d /home/$APP_USER $APP_USER
        usermod -aG docker $APP_USER
        log "User $APP_USER created successfully"
    else
        info "User $APP_USER already exists"
        usermod -aG docker $APP_USER
    fi
}

# Setup application directory
setup_app_directory() {
    log "Setting up application directory: $APP_DIR"
    
    mkdir -p $APP_DIR
    mkdir -p $APP_DIR/logs
    mkdir -p $APP_DIR/backups
    mkdir -p $APP_DIR/ssl
    mkdir -p $APP_DIR/monitoring
    mkdir -p $APP_DIR/data/postgres
    mkdir -p $APP_DIR/data/ollama
    mkdir -p $APP_DIR/data/redis
    
    # Set ownership
    chown -R $APP_USER:$APP_USER $APP_DIR
    chmod -R 755 $APP_DIR
    
    log "Application directory setup completed"
}

# Setup firewall
setup_firewall() {
    log "Configuring firewall..."
    
    # Reset UFW to defaults
    ufw --force reset
    
    # Default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH
    ufw allow ssh
    
    # Allow HTTP and HTTPS
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # Allow internal communication (Docker networks)
    ufw allow from 172.16.0.0/12
    ufw allow from 10.0.0.0/8
    ufw allow from 192.168.0.0/16
    
    # Enable firewall
    ufw --force enable
    
    log "Firewall configured successfully"
}

# Setup fail2ban
setup_fail2ban() {
    log "Configuring fail2ban..."
    
    # Create jail.local configuration
    cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
backend = auto

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
findtime = 600
bantime = 7200
EOF
    
    # Restart fail2ban
    systemctl restart fail2ban
    systemctl enable fail2ban
    
    log "Fail2ban configured successfully"
}

# Generate SSL certificates (self-signed for testing)
generate_ssl_certs() {
    log "Generating SSL certificates..."
    
    SSL_DIR="$APP_DIR/ssl"
    
    # Generate private key
    openssl genrsa -out $SSL_DIR/key.pem 2048
    
    # Generate certificate
    openssl req -new -x509 -key $SSL_DIR/key.pem -out $SSL_DIR/cert.pem -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    
    # Set permissions
    chmod 600 $SSL_DIR/key.pem
    chmod 644 $SSL_DIR/cert.pem
    chown -R $APP_USER:$APP_USER $SSL_DIR
    
    warning "Self-signed certificates generated. Replace with proper certificates in production."
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat > /etc/logrotate.d/$APP_NAME << EOF
$APP_DIR/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 $APP_USER $APP_USER
    postrotate
        docker-compose -f $APP_DIR/docker-compose.yml restart nginx || true
    endscript
}

/var/log/nginx/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data adm
    prerotate
        if [ -d /etc/logrotate.d/httpd-prerotate ]; then \
            run-parts /etc/logrotate.d/httpd-prerotate; \
        fi
    endscript
    postrotate
        docker exec nicomatic-nginx nginx -s reload || true
    endscript
}
EOF
    
    log "Log rotation configured successfully"
}

# Setup backup script
setup_backup_script() {
    log "Setting up backup script..."
    
    cat > $APP_DIR/backup.sh << 'EOF'
#!/bin/bash

# Backup script for Nicomatic Chatbot
BACKUP_DIR="/opt/nicomatic-chatbot/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$DATE.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
docker exec nicomatic-postgres-prod pg_dump -U postgres alexis > $BACKUP_DIR/db_$DATE.sql

# Application data backup
tar -czf $BACKUP_FILE \
    --exclude='*/logs/*' \
    --exclude='*/backups/*' \
    /opt/nicomatic-chatbot/data

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
EOF
    
    chmod +x $APP_DIR/backup.sh
    chown $APP_USER:$APP_USER $APP_DIR/backup.sh
    
    # Add to crontab
    (crontab -u $APP_USER -l 2>/dev/null; echo "0 2 * * * $APP_DIR/backup.sh") | crontab -u $APP_USER -
    
    log "Backup script configured successfully"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    # Create prometheus configuration
    mkdir -p $APP_DIR/monitoring
    cat > $APP_DIR/monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nicomatic-app'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    scrape_interval: 30s
EOF
    
    chown -R $APP_USER:$APP_USER $APP_DIR/monitoring
    
    log "Monitoring configuration created"
}

# Setup systemd service for docker-compose
setup_systemd_service() {
    log "Setting up systemd service..."
    
    cat > /etc/systemd/system/$APP_NAME.service << EOF
[Unit]
Description=Nicomatic Chatbot Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0
User=$APP_USER
Group=$APP_USER

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable $APP_NAME
    
    log "Systemd service configured successfully"
}

# Main installation function
main() {
    log "Starting deployment server setup for $APP_NAME..."
    
    check_root
    update_system
    install_docker
    install_nvidia_docker
    create_app_user
    setup_app_directory
    setup_firewall
    setup_fail2ban
    generate_ssl_certs
    setup_log_rotation
    setup_backup_script
    setup_monitoring
    setup_systemd_service
    
    log "Deployment server setup completed successfully!"
    log "Next steps:"
    log "1. Replace self-signed SSL certificates with proper ones"
    log "2. Update firewall rules for your specific network"
    log "3. Configure domain names and DNS"
    log "4. Set up GitLab CI/CD variables"
    log "5. Deploy the application using GitLab CI/CD"
    
    info "Application directory: $APP_DIR"
    info "Application user: $APP_USER"
    info "SSL certificates: $APP_DIR/ssl/"
    info "Logs: $APP_DIR/logs/"
    info "Backups: $APP_DIR/backups/"
    
    warning "Remember to:"
    warning "- Update SSH keys for GitLab CI/CD access"
    warning "- Configure environment variables"
    warning "- Set up proper SSL certificates"
    warning "- Review and adjust firewall rules"
}

# Run main function
main "$@"
