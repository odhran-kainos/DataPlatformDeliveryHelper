[⬅️ Back to Main SDLC Page](00_data_platform_sdlc.md) | [⬅️ Back to Analytics & Consumption Part 1](07_analytics_consumption.md)

# Advanced Analytics & Consumption Topics
## Real-time Analytics, ML-powered Insights & Advanced Visualization

**Purpose:** This document covers advanced analytics and consumption topics including real-time analytics, machine learning-powered insights, advanced visualization techniques, and comprehensive real-world implementation scenarios.

---

## 🌊 Real-time Analytics Architecture

### 5.1 Stream Analytics Implementation

#### Apache Kafka + ksqlDB Real-time Analytics

```sql
-- ksqlDB Stream Processing for Real-time Analytics
-- Create streams from Kafka topics

-- Raw events stream
CREATE STREAM raw_events (
    user_id VARCHAR,
    event_type VARCHAR,
    timestamp BIGINT,
    properties MAP<VARCHAR, VARCHAR>
) WITH (
    KAFKA_TOPIC='raw-events',
    VALUE_FORMAT='JSON',
    TIMESTAMP='timestamp'
);

-- User profiles table
CREATE TABLE user_profiles (
    user_id VARCHAR PRIMARY KEY,
    segment VARCHAR,
    registration_date BIGINT,
    total_value DOUBLE
) WITH (
    KAFKA_TOPIC='user-profiles',
    VALUE_FORMAT='JSON'
);

-- Real-time event enrichment
CREATE STREAM enriched_events AS
SELECT 
    e.user_id,
    e.event_type,
    e.timestamp,
    e.properties,
    u.segment,
    u.total_value
FROM raw_events e
LEFT JOIN user_profiles u ON e.user_id = u.user_id
EMIT CHANGES;

-- Real-time aggregations - Events per minute by segment
CREATE TABLE events_per_minute_by_segment AS
SELECT 
    segment,
    COUNT(*) as event_count,
    WINDOWSTART as window_start,
    WINDOWEND as window_end
FROM enriched_events
WINDOW TUMBLING (SIZE 1 MINUTE)
GROUP BY segment
EMIT CHANGES;

-- Real-time anomaly detection - High activity users
CREATE STREAM high_activity_users AS
SELECT 
    user_id,
    COUNT(*) as event_count,
    WINDOWSTART as window_start
FROM enriched_events
WINDOW TUMBLING (SIZE 5 MINUTES)
GROUP BY user_id
HAVING COUNT(*) > 100
EMIT CHANGES;

-- Revenue tracking in real-time
CREATE STREAM revenue_events AS
SELECT 
    user_id,
    CAST(properties['amount'] AS DOUBLE) as amount,
    timestamp,
    segment
FROM enriched_events
WHERE event_type = 'purchase'
EMIT CHANGES;

CREATE TABLE revenue_per_minute AS
SELECT 
    segment,
    SUM(amount) as total_revenue,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_transaction_value,
    WINDOWSTART as window_start
FROM revenue_events
WINDOW TUMBLING (SIZE 1 MINUTE)
GROUP BY segment
EMIT CHANGES;
```

#### Real-time Analytics Dashboard Backend

```python
# realtime_analytics_backend.py
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
import asyncio
import websockets
import logging
from typing import Dict, List, Set
from datetime import datetime, timedelta
import redis
from dataclasses import dataclass, asdict
import threading

@dataclass
class RealTimeMetric:
    metric_name: str
    value: float
    timestamp: datetime
    dimensions: Dict[str, str]
    metadata: Dict[str, any] = None

class RealTimeAnalyticsEngine:
    def __init__(self, kafka_config: Dict, redis_config: Dict):
        self.kafka_config = kafka_config
        self.redis_client = redis.Redis(**redis_config)
        self.logger = logging.getLogger(__name__)
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Set[websockets.WebSocketServerProtocol] = set()
        
        # Metric aggregators
        self.metric_aggregators = {}
        
        # Consumer threads
        self.consumer_threads = []
        self.running = False
    
    def start(self):
        """Start the real-time analytics engine"""
        self.running = True
        
        # Start Kafka consumers for different metric streams
        metric_topics = [
            'events_per_minute_by_segment',
            'revenue_per_minute',
            'high_activity_users'
        ]
        
        for topic in metric_topics:
            thread = threading.Thread(
                target=self._consume_metrics,
                args=(topic,),
                daemon=True
            )
            thread.start()
            self.consumer_threads.append(thread)
        
        # Start WebSocket server
        websocket_thread = threading.Thread(
            target=self._start_websocket_server,
            daemon=True
        )
        websocket_thread.start()
        
        self.logger.info("Real-time analytics engine started")
    
    def stop(self):
        """Stop the analytics engine"""
        self.running = False
        self.logger.info("Real-time analytics engine stopped")
    
    def _consume_metrics(self, topic: str):
        """Consume metrics from Kafka topic"""
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id=f'analytics_engine_{topic}'
        )
        
        try:
            while self.running:
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        self._process_metric_message(topic, message.value)
                        
        except KafkaError as e:
            self.logger.error(f"Kafka consumer error for topic {topic}: {e}")
        finally:
            consumer.close()
    
    def _process_metric_message(self, topic: str, message: Dict):
        """Process incoming metric message"""
        
        try:
            # Create metric object
            metric = self._create_metric_from_message(topic, message)
            
            # Store in Redis for persistence
            self._store_metric(metric)
            
            # Send to WebSocket clients
            asyncio.run(self._broadcast_metric(metric))
            
            # Update aggregators
            self._update_aggregators(metric)
            
        except Exception as e:
            self.logger.error(f"Error processing metric message: {e}")
    
    def _create_metric_from_message(self, topic: str, message: Dict) -> RealTimeMetric:
        """Create metric object from Kafka message"""
        
        if topic == 'events_per_minute_by_segment':
            return RealTimeMetric(
                metric_name='events_per_minute',
                value=message['EVENT_COUNT'],
                timestamp=datetime.fromtimestamp(message['WINDOW_START'] / 1000),
                dimensions={'segment': message['SEGMENT']},
                metadata={'window_end': message['WINDOW_END']}
            )
        
        elif topic == 'revenue_per_minute':
            return RealTimeMetric(
                metric_name='revenue_per_minute',
                value=message['TOTAL_REVENUE'],
                timestamp=datetime.fromtimestamp(message['WINDOW_START'] / 1000),
                dimensions={'segment': message['SEGMENT']},
                metadata={
                    'transaction_count': message['TRANSACTION_COUNT'],
                    'avg_transaction_value': message['AVG_TRANSACTION_VALUE']
                }
            )
        
        elif topic == 'high_activity_users':
            return RealTimeMetric(
                metric_name='high_activity_alert',
                value=message['EVENT_COUNT'],
                timestamp=datetime.fromtimestamp(message['WINDOW_START'] / 1000),
                dimensions={'user_id': message['USER_ID']},
                metadata={'alert_type': 'high_activity'}
            )
        
        else:
            raise ValueError(f"Unknown topic: {topic}")
    
    def _store_metric(self, metric: RealTimeMetric):
        """Store metric in Redis"""
        
        # Create Redis key
        key_parts = [
            'realtime_metric',
            metric.metric_name,
            str(int(metric.timestamp.timestamp()))
        ]
        
        # Add dimensions to key
        for dim_key, dim_value in metric.dimensions.items():
            key_parts.append(f"{dim_key}:{dim_value}")
        
        redis_key = ':'.join(key_parts)
        
        # Store metric data
        metric_data = {
            'value': metric.value,
            'timestamp': metric.timestamp.isoformat(),
            'dimensions': json.dumps(metric.dimensions),
            'metadata': json.dumps(metric.metadata or {})
        }
        
        # Set with TTL (keep for 24 hours)
        self.redis_client.hmset(redis_key, metric_data)
        self.redis_client.expire(redis_key, 86400)
    
    async def _broadcast_metric(self, metric: RealTimeMetric):
        """Broadcast metric to WebSocket clients"""
        
        if not self.websocket_connections:
            return
        
        message = {
            'type': 'metric_update',
            'data': asdict(metric),
            'timestamp': metric.timestamp.isoformat()
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(websocket)
            except Exception as e:
                self.logger.error(f"Error sending WebSocket message: {e}")
                disconnected_clients.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected_clients
    
    def _update_aggregators(self, metric: RealTimeMetric):
        """Update metric aggregators for dashboard summaries"""
        
        aggregator_key = f"{metric.metric_name}:aggregator"
        
        if aggregator_key not in self.metric_aggregators:
            self.metric_aggregators[aggregator_key] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'last_update': metric.timestamp
            }
        
        aggregator = self.metric_aggregators[aggregator_key]
        aggregator['count'] += 1
        aggregator['sum'] += metric.value
        aggregator['min'] = min(aggregator['min'], metric.value)
        aggregator['max'] = max(aggregator['max'], metric.value)
        aggregator['last_update'] = metric.timestamp
        aggregator['avg'] = aggregator['sum'] / aggregator['count']
    
    def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        
        async def handle_client(websocket, path):
            """Handle WebSocket client connection"""
            self.websocket_connections.add(websocket)
            self.logger.info(f"New WebSocket client connected: {websocket.remote_address}")
            
            try:
                # Send current aggregator state to new client
                await self._send_current_state(websocket)
                
                # Keep connection alive
                async for message in websocket:
                    # Handle client messages if needed
                    client_message = json.loads(message)
                    await self._handle_client_message(websocket, client_message)
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.discard(websocket)
                self.logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        
        # Start WebSocket server
        start_server = websockets.serve(handle_client, "localhost", 8765)
        asyncio.new_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    
    async def _send_current_state(self, websocket):
        """Send current aggregator state to client"""
        
        state_message = {
            'type': 'current_state',
            'data': self.metric_aggregators,
            'timestamp': datetime.now().isoformat()
        }
        
        await websocket.send(json.dumps(state_message))
    
    async def _handle_client_message(self, websocket, message: Dict):
        """Handle messages from WebSocket clients"""
        
        message_type = message.get('type')
        
        if message_type == 'subscribe_metric':
            # Handle metric subscription
            metric_name = message.get('metric_name')
            self.logger.info(f"Client subscribed to metric: {metric_name}")
            
        elif message_type == 'get_historical_data':
            # Send historical data
            await self._send_historical_data(websocket, message)
    
    async def _send_historical_data(self, websocket, request: Dict):
        """Send historical data to client"""
        
        metric_name = request.get('metric_name')
        time_range = request.get('time_range', 3600)  # Default 1 hour
        
        # Query Redis for historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=time_range)
        
        historical_data = self._get_historical_metrics(
            metric_name, start_time, end_time
        )
        
        response = {
            'type': 'historical_data',
            'metric_name': metric_name,
            'data': historical_data,
            'timestamp': datetime.now().isoformat()
        }
        
        await websocket.send(json.dumps(response))
    
    def _get_historical_metrics(self, metric_name: str, 
                               start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get historical metrics from Redis"""
        
        # Search for metric keys in time range
        pattern = f"realtime_metric:{metric_name}:*"
        keys = self.redis_client.keys(pattern)
        
        historical_data = []
        
        for key in keys:
            # Extract timestamp from key
            key_parts = key.decode().split(':')
            if len(key_parts) >= 3:
                try:
                    key_timestamp = datetime.fromtimestamp(int(key_parts[2]))
                    
                    if start_time <= key_timestamp <= end_time:
                        # Get metric data
                        metric_data = self.redis_client.hgetall(key)
                        
                        if metric_data:
                            historical_data.append({
                                'timestamp': key_timestamp.isoformat(),
                                'value': float(metric_data[b'value']),
                                'dimensions': json.loads(metric_data[b'dimensions']),
                                'metadata': json.loads(metric_data[b'metadata'])
                            })
                
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error parsing metric key {key}: {e}")
        
        # Sort by timestamp
        historical_data.sort(key=lambda x: x['timestamp'])
        
        return historical_data

# Example usage
kafka_config = {
    'bootstrap_servers': ['localhost:9092']
}

redis_config = {
    'host': 'localhost',
    'port': 6379,
    'db': 2
}

# Start real-time analytics engine
analytics_engine = RealTimeAnalyticsEngine(kafka_config, redis_config)
analytics_engine.start()
```

### 5.2 Real-time Dashboard Frontend

#### React Real-time Dashboard

```javascript
// RealTimeDashboard.jsx
import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer
} from 'recharts';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';

const RealTimeDashboard = () => {
  const [metrics, setMetrics] = useState({});
  const [historicalData, setHistoricalData] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const wsRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8765');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
      
      // Request historical data for key metrics
      ws.send(JSON.stringify({
        type: 'get_historical_data',
        metric_name: 'events_per_minute',
        time_range: 3600 // Last hour
      }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
      
      // Attempt to reconnect after 5 seconds
      setTimeout(connectWebSocket, 5000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };
  };

  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'metric_update':
        handleMetricUpdate(message.data);
        break;
      case 'current_state':
        setMetrics(message.data);
        break;
      case 'historical_data':
        setHistoricalData(prev => ({
          ...prev,
          [message.metric_name]: message.data
        }));
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const handleMetricUpdate = (metric) => {
    // Update real-time metrics
    setMetrics(prev => ({
      ...prev,
      [`${metric.metric_name}:aggregator`]: {
        ...prev[`${metric.metric_name}:aggregator`],
        last_value: metric.value,
        last_update: metric.timestamp
      }
    }));

    // Handle alerts
    if (metric.metric_name === 'high_activity_alert') {
      const alert = {
        id: Date.now(),
        type: 'warning',
        message: `High activity detected for user ${metric.dimensions.user_id}: ${metric.value} events`,
        timestamp: metric.timestamp
      };
      
      setAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep last 10 alerts
    }

    // Update historical data
    setHistoricalData(prev => {
      const metricHistory = prev[metric.metric_name] || [];
      const updatedHistory = [
        ...metricHistory,
        {
          timestamp: metric.timestamp,
          value: metric.value,
          dimensions: metric.dimensions
        }
      ].slice(-100); // Keep last 100 data points

      return {
        ...prev,
        [metric.metric_name]: updatedHistory
      };
    });
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'bg-green-500';
      case 'disconnected': return 'bg-red-500';
      case 'error': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const renderMetricCard = (metricKey, title, format = 'number') => {
    const metric = metrics[metricKey];
    if (!metric) return null;

    const formatValue = (value) => {
      switch (format) {
        case 'currency':
          return `$${value?.toLocaleString() || 0}`;
        case 'percentage':
          return `${(value || 0).toFixed(1)}%`;
        default:
          return (value || 0).toLocaleString();
      }
    };

    return (
      <Card className="w-full">
        <CardHeader className="pb-2">
          <h3 className="text-sm font-medium text-gray-600">{title}</h3>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {formatValue(metric.last_value || metric.avg)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Last updated: {metric.last_update ? formatTimestamp(metric.last_update) : 'N/A'}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Min: {formatValue(metric.min)} | Max: {formatValue(metric.max)} | Avg: {formatValue(metric.avg)}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderLineChart = (metricName, title) => {
    const data = historicalData[metricName] || [];
    
    if (data.length === 0) {
      return (
        <Card className="w-full h-64">
          <CardHeader>
            <h3 className="text-lg font-semibold">{title}</h3>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-32">
            <p className="text-gray-500">No data available</p>
          </CardContent>
        </Card>
      );
    }

    const chartData = data.map(point => ({
      time: formatTimestamp(point.timestamp),
      value: point.value,
      ...point.dimensions
    }));

    return (
      <Card className="w-full">
        <CardHeader>
          <h3 className="text-lg font-semibold">{title}</h3>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#8884d8" 
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Real-Time Analytics Dashboard</h1>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()}`}></div>
          <span className="text-sm text-gray-600 capitalize">{connectionStatus}</span>
        </div>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-3">Recent Alerts</h2>
          <div className="space-y-2">
            {alerts.slice(0, 3).map(alert => (
              <Alert key={alert.id} className="border-yellow-200 bg-yellow-50">
                <AlertDescription>
                  <div className="flex justify-between items-center">
                    <span>{alert.message}</span>
                    <Badge variant="outline">{formatTimestamp(alert.timestamp)}</Badge>
                  </div>
                </AlertDescription>
              </Alert>
            ))}
          </div>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {renderMetricCard('events_per_minute:aggregator', 'Events/Minute')}
        {renderMetricCard('revenue_per_minute:aggregator', 'Revenue/Minute', 'currency')}
        {renderMetricCard('high_activity_alert:aggregator', 'High Activity Alerts')}
        <Card className="w-full">
          <CardHeader className="pb-2">
            <h3 className="text-sm font-medium text-gray-600">Active Users</h3>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Object.keys(historicalData).length || 0}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Tracked metrics: {Object.keys(metrics).length}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {renderLineChart('events_per_minute', 'Events Per Minute Trend')}
        {renderLineChart('revenue_per_minute', 'Revenue Per Minute Trend')}
      </div>

      {/* Additional Charts */}
      <div className="mt-6">
        <Card className="w-full">
          <CardHeader>
            <h3 className="text-lg font-semibold">System Status</h3>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {connectionStatus === 'connected' ? '✓' : '✗'}
                </div>
                <div className="text-sm text-gray-600">WebSocket</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {Object.keys(metrics).length}
                </div>
                <div className="text-sm text-gray-600">Active Metrics</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {alerts.length}
                </div>
                <div className="text-sm text-gray-600">Total Alerts</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {new Date().toLocaleTimeString()}
                </div>
                <div className="text-sm text-gray-600">Current Time</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default RealTimeDashboard;
```

---

## 🤖 ML-Powered Analytics Integration

### 6.1 Predictive Analytics Framework

#### TODO!