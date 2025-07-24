#!/bin/bash

# Microservice Demo Kubernetes Deployment Script
# This script deploys the entire microservice architecture to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    print_success "kubectl is available"
}

# Function to check if cluster is accessible
check_cluster() {
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    print_success "Kubernetes cluster is accessible"
}

# Function to create namespace and basic resources
deploy_namespace() {
    print_status "Deploying namespace and basic resources..."
    kubectl apply -f namespace.yaml
    print_success "Namespace created successfully"
}

# Function to deploy databases
deploy_databases() {
    print_status "Deploying databases (PostgreSQL and Redis)..."
    kubectl apply -f database.yaml
    
    print_status "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n microservice-demo --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n microservice-demo --timeout=300s
    
    print_success "Databases deployed successfully"
}

# Function to deploy monitoring stack
deploy_monitoring() {
    print_status "Deploying monitoring stack (Prometheus, Grafana, Jaeger)..."
    kubectl apply -f monitoring.yaml
    
    print_status "Waiting for monitoring services to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n microservice-demo --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n microservice-demo --timeout=300s
    kubectl wait --for=condition=ready pod -l app=jaeger -n microservice-demo --timeout=300s
    
    print_success "Monitoring stack deployed successfully"
}

# Function to build and push Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not available. Skipping image build."
        print_warning "Please build and push images manually before deploying services."
        return
    fi
    
    # Build images for each service
    services=("api-gateway" "user-service" "product-service" "order-service" "notification-service")
    
    for service in "${services[@]}"; do
        print_status "Building $service image..."
        if [ -d "../services/$service" ]; then
            docker build -t "microservice-demo/$service:latest" "../services/$service"
            print_success "$service image built successfully"
        else
            print_warning "Service directory ../services/$service not found. Skipping."
        fi
    done
}

# Function to deploy API Gateway
deploy_api_gateway() {
    print_status "Deploying API Gateway..."
    kubectl apply -f api-gateway.yaml
    
    print_status "Waiting for API Gateway to be ready..."
    kubectl wait --for=condition=ready pod -l app=api-gateway -n microservice-demo --timeout=300s
    
    print_success "API Gateway deployed successfully"
}

# Function to deploy microservices
deploy_services() {
    print_status "Deploying microservices..."
    
    # Deploy services if their YAML files exist
    services=("user-service" "product-service" "order-service" "notification-service")
    
    for service in "${services[@]}"; do
        if [ -f "$service.yaml" ]; then
            print_status "Deploying $service..."
            kubectl apply -f "$service.yaml"
            kubectl wait --for=condition=ready pod -l "app=$service" -n microservice-demo --timeout=300s
            print_success "$service deployed successfully"
        else
            print_warning "$service.yaml not found. Skipping."
        fi
    done
}

# Function to show deployment status
show_status() {
    print_status "Deployment Status:"
    echo
    
    print_status "Pods:"
    kubectl get pods -n microservice-demo
    echo
    
    print_status "Services:"
    kubectl get services -n microservice-demo
    echo
    
    print_status "Ingresses:"
    kubectl get ingress -n microservice-demo
    echo
}

# Function to show access information
show_access_info() {
    print_status "Access Information:"
    echo
    
    # Get service URLs
    print_status "To access services, use port-forwarding:"
    echo
    
    echo "API Gateway:"
    echo "  kubectl port-forward svc/api-gateway 8080:8000 -n microservice-demo"
    echo "  Then access: http://localhost:8080"
    echo
    
    echo "Grafana Dashboard:"
    echo "  kubectl port-forward svc/grafana 3000:3000 -n microservice-demo"
    echo "  Then access: http://localhost:3000 (admin/admin123)"
    echo
    
    echo "Prometheus:"
    echo "  kubectl port-forward svc/prometheus 9090:9090 -n microservice-demo"
    echo "  Then access: http://localhost:9090"
    echo
    
    echo "Jaeger Tracing:"
    echo "  kubectl port-forward svc/jaeger 16686:16686 -n microservice-demo"
    echo "  Then access: http://localhost:16686"
    echo
}

# Function to clean up deployment
cleanup() {
    print_status "Cleaning up deployment..."
    
    # Delete all resources in the namespace
    kubectl delete namespace microservice-demo --ignore-not-found=true
    
    print_success "Cleanup completed"
}

# Main deployment function
deploy_all() {
    print_status "Starting microservice deployment..."
    echo
    
    check_kubectl
    check_cluster
    
    deploy_namespace
    deploy_databases
    deploy_monitoring
    
    # Uncomment the following lines when service YAML files are available
    # build_images
    # deploy_api_gateway
    # deploy_services
    
    show_status
    show_access_info
    
    print_success "Deployment completed successfully!"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy_all
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        show_status
        ;;
    "access")
        show_access_info
        ;;
    "namespace")
        deploy_namespace
        ;;
    "databases")
        deploy_databases
        ;;
    "monitoring")
        deploy_monitoring
        ;;
    "gateway")
        deploy_api_gateway
        ;;
    "services")
        deploy_services
        ;;
    "build")
        build_images
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  deploy     - Deploy all components (default)"
        echo "  cleanup    - Remove all deployed resources"
        echo "  status     - Show deployment status"
        echo "  access     - Show access information"
        echo "  namespace  - Deploy only namespace"
        echo "  databases  - Deploy only databases"
        echo "  monitoring - Deploy only monitoring stack"
        echo "  gateway    - Deploy only API gateway"
        echo "  services   - Deploy only microservices"
        echo "  build      - Build Docker images"
        echo "  help       - Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac