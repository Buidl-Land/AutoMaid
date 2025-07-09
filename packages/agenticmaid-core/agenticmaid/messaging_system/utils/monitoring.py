"""
Monitoring and metrics utilities for the messaging system.

This module provides monitoring capabilities and metrics collection
for the messaging system components.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ComponentMetrics:
    """Metrics for a single component."""
    component_id: str
    component_type: str
    status: str
    uptime: float = 0.0
    message_count: int = 0
    error_count: int = 0
    last_activity: Optional[datetime] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores metrics for messaging system components."""
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize the metrics collector.
        
        Args:
            max_history_size: Maximum number of metric points to keep in history
        """
        self.max_history_size = max_history_size
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self._component_metrics: Dict[str, ComponentMetrics] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        component_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            component_id: ID of the component (optional)
            tags: Additional tags for the metric
        """
        with self._lock:
            metric_key = f"{component_id}.{metric_name}" if component_id else metric_name
            
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                tags=tags or {}
            )
            
            self._metrics_history[metric_key].append(point)
    
    def update_component_metrics(self, component_id: str, metrics: ComponentMetrics) -> None:
        """Update metrics for a component."""
        with self._lock:
            self._component_metrics[component_id] = metrics
    
    def get_component_metrics(self, component_id: str) -> Optional[ComponentMetrics]:
        """Get metrics for a specific component."""
        with self._lock:
            return self._component_metrics.get(component_id)
    
    def get_all_component_metrics(self) -> Dict[str, ComponentMetrics]:
        """Get metrics for all components."""
        with self._lock:
            return self._component_metrics.copy()
    
    def get_metric_history(
        self,
        metric_name: str,
        component_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """
        Get metric history.
        
        Args:
            metric_name: Name of the metric
            component_id: ID of the component (optional)
            since: Only return points after this timestamp
            
        Returns:
            List of metric points
        """
        with self._lock:
            metric_key = f"{component_id}.{metric_name}" if component_id else metric_name
            history = list(self._metrics_history.get(metric_key, []))
            
            if since:
                history = [point for point in history if point.timestamp >= since]
            
            return history
    
    def get_metric_summary(
        self,
        metric_name: str,
        component_id: Optional[str] = None,
        window_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Get summary statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            component_id: ID of the component (optional)
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary with summary statistics
        """
        since = datetime.utcnow() - timedelta(minutes=window_minutes)
        history = self.get_metric_history(metric_name, component_id, since)
        
        if not history:
            return {}
        
        values = [point.value for point in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0.0
        }
    
    def clear_metrics(self, component_id: Optional[str] = None) -> None:
        """Clear metrics for a component or all metrics."""
        with self._lock:
            if component_id:
                # Clear metrics for specific component
                keys_to_remove = [key for key in self._metrics_history.keys() if key.startswith(f"{component_id}.")]
                for key in keys_to_remove:
                    del self._metrics_history[key]
                
                if component_id in self._component_metrics:
                    del self._component_metrics[component_id]
            else:
                # Clear all metrics
                self._metrics_history.clear()
                self._component_metrics.clear()


class MessagingSystemMonitor:
    """Monitor for the messaging system that tracks component health and performance."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the system monitor.
        
        Args:
            metrics_collector: Metrics collector instance (creates new if None)
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # Health check configuration
        self.health_check_interval = 60.0  # seconds
        self.health_check_timeout = 10.0   # seconds
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Component registry
        self._components: Dict[str, Any] = {}  # component_id -> component instance
        self._health_checkers: Dict[str, Callable[[], bool]] = {}
    
    def register_component(
        self,
        component_id: str,
        component: Any,
        health_checker: Optional[Callable[[], bool]] = None
    ) -> None:
        """
        Register a component for monitoring.
        
        Args:
            component_id: Unique identifier for the component
            component: Component instance
            health_checker: Optional custom health check function
        """
        self._components[component_id] = component
        
        if health_checker:
            self._health_checkers[component_id] = health_checker
        elif hasattr(component, 'health_check'):
            self._health_checkers[component_id] = component.health_check
        
        self.logger.info(f"Registered component for monitoring: {component_id}")
    
    def unregister_component(self, component_id: str) -> None:
        """Unregister a component from monitoring."""
        if component_id in self._components:
            del self._components[component_id]
        
        if component_id in self._health_checkers:
            del self._health_checkers[component_id]
        
        # Clear metrics for the component
        self.metrics.clear_metrics(component_id)
        
        self.logger.info(f"Unregistered component from monitoring: {component_id}")
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self._monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self._monitoring_active = True
        self._shutdown_event.clear()
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MessagingSystemMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Started messaging system monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        self._shutdown_event.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Stopped messaging system monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Monitoring loop started")
        
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                # Perform health checks
                self._perform_health_checks()
                
                # Collect component metrics
                self._collect_component_metrics()
                
                # Wait for next check
                self._shutdown_event.wait(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Brief pause before retrying
        
        self.logger.info("Monitoring loop stopped")
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all registered components."""
        for component_id, health_checker in self._health_checkers.items():
            try:
                start_time = time.time()
                is_healthy = health_checker()
                check_duration = time.time() - start_time
                
                # Record health check metrics
                self.metrics.record_metric(
                    "health_check_duration",
                    check_duration,
                    component_id,
                    {"status": "healthy" if is_healthy else "unhealthy"}
                )
                
                self.metrics.record_metric(
                    "health_status",
                    1.0 if is_healthy else 0.0,
                    component_id
                )
                
                if not is_healthy:
                    self.logger.warning(f"Health check failed for component: {component_id}")
                
            except Exception as e:
                self.logger.error(f"Health check error for component {component_id}: {e}")
                self.metrics.record_metric("health_status", 0.0, component_id, {"error": str(e)})
    
    def _collect_component_metrics(self) -> None:
        """Collect metrics from all registered components."""
        for component_id, component in self._components.items():
            try:
                # Get component info if available
                if hasattr(component, 'client_info'):
                    info = component.client_info
                    metrics = ComponentMetrics(
                        component_id=component_id,
                        component_type=info.client_type,
                        status=info.status.value,
                        message_count=info.message_count,
                        error_count=info.error_count,
                        last_activity=info.last_activity
                    )
                elif hasattr(component, 'trigger_info'):
                    info = component.trigger_info
                    metrics = ComponentMetrics(
                        component_id=component_id,
                        component_type=info.trigger_type,
                        status=info.status.value,
                        message_count=info.events_processed,
                        error_count=info.error_count,
                        last_activity=info.last_check
                    )
                else:
                    # Generic component
                    metrics = ComponentMetrics(
                        component_id=component_id,
                        component_type="unknown",
                        status="unknown"
                    )
                
                self.metrics.update_component_metrics(component_id, metrics)
                
                # Record individual metrics
                self.metrics.record_metric("message_count", metrics.message_count, component_id)
                self.metrics.record_metric("error_count", metrics.error_count, component_id)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics for component {component_id}: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        component_metrics = self.metrics.get_all_component_metrics()
        
        total_components = len(component_metrics)
        healthy_components = 0
        total_messages = 0
        total_errors = 0
        
        for metrics in component_metrics.values():
            if metrics.status in ["connected", "running"]:
                healthy_components += 1
            total_messages += metrics.message_count
            total_errors += metrics.error_count
        
        health_percentage = (healthy_components / total_components * 100) if total_components > 0 else 0
        
        return {
            "overall_health": "healthy" if health_percentage >= 80 else "degraded" if health_percentage >= 50 else "unhealthy",
            "health_percentage": health_percentage,
            "total_components": total_components,
            "healthy_components": healthy_components,
            "total_messages": total_messages,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_messages * 100) if total_messages > 0 else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
