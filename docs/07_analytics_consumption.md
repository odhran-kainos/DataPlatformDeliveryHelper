[⬅️ Back to Main SDLC Page](00_data_platform_sdlc.md)

# Analytics & Consumption for Data Platform Projects
## LLM-Optimized Framework for Data Analytics and Consumption Architecture

**Purpose:** This document provides a comprehensive, actionable framework for designing and implementing analytics and data consumption solutions, specifically optimized for LLM-assisted analytics architecture and consumption workflows.

**📋 Document Structure:**
- **Part 1 (This Document):** Core analytics architecture, BI tools, and consumption patterns
- **Part 2:** [Advanced Analytics Topics](07b_analytics_consumption_advanced.md) - Real-time analytics, ML-powered insights, and advanced visualization

---

## 🤖 LLM Analytics Assistant Guide

### LLM Roles in Analytics & Consumption

**1. Analytics Architect**
- Design comprehensive analytics architecture and data consumption patterns
- Select optimal BI tools and visualization technologies
- Create scalable and performant analytics solutions

**2. Dashboard Developer**
- Build interactive dashboards and reporting solutions
- Implement self-service analytics capabilities
- Design user-friendly data exploration interfaces

**3. API Developer**
- Create robust data APIs and microservices
- Implement authentication and authorization systems
- Build real-time data streaming endpoints

**4. Performance Optimizer**
- Optimize query performance and caching strategies
- Implement data aggregation and materialization
- Design efficient data serving architectures

### LLM Prompt Templates

**Analytics Architecture Design:**
```
Design a complete analytics and consumption architecture for this use case:
- User types: [BUSINESS_USERS/DATA_ANALYSTS/EXECUTIVES/EXTERNAL_CONSUMERS]
- Data volume: [RECORDS_COUNT/DATA_SIZE]
- Query patterns: [OLAP/OLTP/MIXED/REAL_TIME]
- Latency requirements: [REAL_TIME/NEAR_REAL_TIME/BATCH]
- Visualization needs: [DASHBOARDS/REPORTS/EMBEDDED_ANALYTICS]
- Technology constraints: [CLOUD_PROVIDER/BUDGET/EXISTING_TOOLS]
Use the patterns from Section 3 and advanced techniques from Part 2.
```

**Dashboard Implementation:**
```
Create a comprehensive dashboard solution for this analytics requirement:
- Business domain: [SALES/MARKETING/OPERATIONS/FINANCE]
- Key metrics: [KPI_LIST]
- User personas: [EXECUTIVE/ANALYST/OPERATIONAL]
- Data sources: [SOURCE_SYSTEMS]
- Interactivity needs: [DRILL_DOWN/FILTERING/REAL_TIME_UPDATES]
- Delivery method: [WEB_APP/MOBILE/EMBEDDED]
Include data modeling, visualization design, and performance optimization.
```

**Data API Development:**
```
Build a data API service for this consumption pattern:
- API type: [REST/GRAPHQL/STREAMING]
- Data access patterns: [CRUD/ANALYTICS/AGGREGATIONS]
- Authentication: [JWT/OAUTH/API_KEYS]
- Rate limiting: [REQUESTS_PER_MINUTE]
- Response format: [JSON/PARQUET/CSV]
- Caching strategy: [REDIS/CDN/APPLICATION_LEVEL]
Provide complete implementation with error handling and monitoring.
```

---

## 📊 Analytics Architecture Decision Framework

### Decision Tree: Analytics Technology Selection

```mermaid
graph TD
    A[Start Analytics Design] --> B{User Type?}
    B -->|Business Users| C[Self-Service BI]
    B -->|Data Analysts| D[Advanced Analytics]
    B -->|Developers| E[API/Programmatic Access]
    
    C --> F{Data Complexity?}
    F -->|Simple| G[Tableau/Power BI]
    F -->|Complex| H[Looker/Sisense]
    
    D --> I{Analysis Type?}
    I -->|Exploratory| J[Jupyter/Databricks]
    I -->|Production| K[Superset/Metabase]
    
    E --> L{Access Pattern?}
    L -->|Real-time| M[GraphQL/WebSocket]
    L -->|Batch| N[REST API/Data Export]
```

### Analytics Technology Selection Matrix

| Use Case | User Type | Data Volume | Latency | Recommended Technology |
|----------|-----------|-------------|---------|----------------------|
| **Executive Dashboards** | Business | Medium | Minutes | Power BI, Tableau |
| **Operational Analytics** | Analysts | High | Seconds | Looker, Sisense |
| **Self-Service BI** | Mixed | Medium | Minutes | Superset, Metabase |
| **Real-time Monitoring** | Operations | High | Sub-second | Grafana, DataDog |
| **Embedded Analytics** | Customers | Variable | Seconds | Plotly Dash, D3.js |
| **Data Science** | Scientists | Very High | Variable | Jupyter, Databricks |

### Consumption Pattern Framework

```json
{
  "consumption_patterns": {
    "dashboard_consumption": {
      "description": "Interactive dashboards for business users",
      "use_cases": ["executive_reporting", "operational_monitoring", "kpi_tracking"],
      "advantages": ["visual_insights", "self_service", "real_time_updates"],
      "disadvantages": ["limited_flexibility", "performance_constraints", "licensing_costs"]
    },
    "api_consumption": {
      "description": "Programmatic access via REST/GraphQL APIs",
      "use_cases": ["application_integration", "mobile_apps", "third_party_systems"],
      "advantages": ["flexibility", "automation", "custom_integration"],
      "disadvantages": ["development_overhead", "maintenance_complexity", "security_concerns"]
    },
    "export_consumption": {
      "description": "Batch data exports and file downloads",
      "use_cases": ["regulatory_reporting", "data_migration", "offline_analysis"],
      "advantages": ["simple_implementation", "large_datasets", "offline_access"],
      "disadvantages": ["not_real_time", "manual_process", "version_control_issues"]
    }
  }
}
```

---

## 🏗️ Comprehensive Analytics Technology Toolkit

### 3.1 Business Intelligence Platforms

#### Apache Superset Dashboard Framework

```python
# superset_dashboard_framework.py
from typing import Dict, List, Optional
import logging
import json
import requests

class SupersetDashboardFramework:
    def __init__(self, superset_url: str, username: str, password: str):
        self.superset_url = superset_url
        self.username = username
        self.password = password
        self.access_token = None
        self.logger = logging.getLogger(__name__)
    
    def authenticate(self) -> str:
        """Authenticate with Superset API"""
        
        auth_url = f"{self.superset_url}/api/v1/security/login"
        auth_data = {
            "username": self.username,
            "password": self.password,
            "provider": "db",
            "refresh": True
        }
        
        response = requests.post(auth_url, json=auth_data)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data['access_token']
        
        self.logger.info("Successfully authenticated with Superset")
        return self.access_token
    
    def get_headers(self) -> Dict:
        """Get authentication headers"""
        if not self.access_token:
            self.authenticate()
        
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def create_database_connection(self, connection_config: Dict) -> Dict:
        """Create database connection in Superset"""
        
        url = f"{self.superset_url}/api/v1/database/"
        
        database_data = {
            "database_name": connection_config['name'],
            "sqlalchemy_uri": connection_config['uri'],
            "expose_in_sqllab": connection_config.get('expose_in_sqllab', True),
            "allow_ctas": connection_config.get('allow_ctas', True),
            "allow_cvas": connection_config.get('allow_cvas', True),
            "allow_dml": connection_config.get('allow_dml', False)
        }
        
        if 'extra' in connection_config:
            database_data['extra'] = json.dumps(connection_config['extra'])
        
        response = requests.post(url, headers=self.get_headers(), json=database_data)
        response.raise_for_status()
        
        database = response.json()
        self.logger.info(f"Created database connection: {connection_config['name']}")
        return database
    
    def create_dataset(self, dataset_config: Dict, database_id: int) -> Dict:
        """Create dataset in Superset"""
        
        url = f"{self.superset_url}/api/v1/dataset/"
        
        dataset_data = {
            "table_name": dataset_config['table_name'],
            "database": database_id,
            "schema": dataset_config.get('schema'),
            "sql": dataset_config.get('sql')
        }
        
        response = requests.post(url, headers=self.get_headers(), json=dataset_data)
        response.raise_for_status()
        
        dataset = response.json()
        self.logger.info(f"Created dataset: {dataset_config['table_name']}")
        return dataset
    
    def create_chart(self, chart_config: Dict, dataset_id: int) -> Dict:
        """Create chart in Superset"""
        
        url = f"{self.superset_url}/api/v1/chart/"
        
        chart_data = {
            "slice_name": chart_config['name'],
            "viz_type": chart_config['viz_type'],
            "datasource_id": dataset_id,
            "datasource_type": "table",
            "params": json.dumps(chart_config['params'])
        }
        
        response = requests.post(url, headers=self.get_headers(), json=chart_data)
        response.raise_for_status()
        
        chart = response.json()
        self.logger.info(f"Created chart: {chart_config['name']}")
        return chart
    
    def create_dashboard(self, dashboard_config: Dict, chart_ids: List[int]) -> Dict:
        """Create dashboard with charts"""
        
        url = f"{self.superset_url}/api/v1/dashboard/"
        
        dashboard_data = {
            "dashboard_title": dashboard_config['title'],
            "slug": dashboard_config.get('slug'),
            "position_json": json.dumps(dashboard_config.get('layout', {})),
            "css": dashboard_config.get('css', ''),
            "json_metadata": json.dumps(dashboard_config.get('metadata', {}))
        }
        
        response = requests.post(url, headers=self.get_headers(), json=dashboard_data)
        response.raise_for_status()
        
        dashboard = response.json()
        
        # Add charts to dashboard
        for chart_id in chart_ids:
            self._add_chart_to_dashboard(dashboard['id'], chart_id)
        
        self.logger.info(f"Created dashboard: {dashboard_config['title']}")
        return dashboard
    
    def _add_chart_to_dashboard(self, dashboard_id: int, chart_id: int) -> None:
        """Add chart to dashboard"""
        
        url = f"{self.superset_url}/api/v1/dashboard/{dashboard_id}/charts"
        
        response = requests.post(
            url, 
            headers=self.get_headers(), 
            json={"chart_ids": [chart_id]}
        )
        response.raise_for_status()

# Example usage configuration
analytics_solution_config = {
    "databases": [
        {
            "name": "data_warehouse",
            "uri": "postgresql://user:password@localhost:5432/warehouse",
            "expose_in_sqllab": True,
            "allow_ctas": True,
            "extra": {
                "metadata_params": {},
                "engine_params": {
                    "pool_size": 10,
                    "max_overflow": 20
                }
            }
        }
    ],
    "datasets": [
        {
            "table_name": "sales_summary",
            "schema": "analytics"
        }
    ],
    "charts": [
        {
            "name": "Revenue Trend",
            "viz_type": "line",
            "params": {
                "metrics": ["sum__revenue"],
                "groupby": ["date"],
                "time_grain_sqla": "P1D"
            }
        }
    ],
    "dashboards": [
        {
            "title": "Sales Executive Dashboard",
            "slug": "sales-executive",
            "layout": {
                "CHART-1": {
                    "type": "CHART",
                    "id": "CHART-1",
                    "children": [],
                    "meta": {"width": 6, "height": 4}
                }
            }
        }
    ]
}
```

### 3.2 Data API Development Framework

#### FastAPI Data Service Implementation

```python
# fastapi_data_service.py
from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from datetime import datetime
import redis
import json
import io

# Pydantic models for API
class QueryRequest(BaseModel):
    sql: str = Field(..., description="SQL query to execute")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Query parameters")
    limit: Optional[int] = Field(default=1000, description="Maximum number of rows")
    format: Optional[str] = Field(default="json", description="Response format: json, csv, parquet")

class MetricRequest(BaseModel):
    metric_name: str = Field(..., description="Name of the metric")
    dimensions: Optional[List[str]] = Field(default=[], description="Grouping dimensions")
    filters: Optional[Dict[str, Any]] = Field(default={}, description="Filter conditions")
    time_range: Optional[Dict[str, str]] = Field(default={}, description="Time range filter")
    aggregation: Optional[str] = Field(default="sum", description="Aggregation function")

class DataResponse(BaseModel):
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time_ms: float
    row_count: int

class DataAPIService:
    def __init__(self, database_config: Dict, redis_config: Dict):
        self.app = FastAPI(
            title="Data Platform API",
            description="Comprehensive data access and analytics API",
            version="1.0.0"
        )
        
        self.logger = logging.getLogger(__name__)
        self.database_config = database_config
        self.redis_client = redis.Redis(**redis_config)
        
        # Setup middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup security
        self.security = HTTPBearer()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.post("/query", response_model=DataResponse)
        async def execute_query(
            request: QueryRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._execute_query(request, credentials)
        
        @self.app.post("/metrics", response_model=DataResponse)
        async def get_metrics(
            request: MetricRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._get_metrics(request, credentials)
        
        @self.app.get("/datasets")
        async def list_datasets(
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._list_datasets(credentials)
    
    async def _authenticate_user(self, credentials: HTTPAuthorizationCredentials) -> Dict:
        """Authenticate user and return user info"""
        
        token = credentials.credentials
        
        # Check token in Redis cache first
        cached_user = self.redis_client.get(f"auth_token:{token}")
        if cached_user:
            return json.loads(cached_user)
        
        # Validate token (implement your authentication logic)
        if token == "valid_token":
            user_info = {
                "user_id": "user123",
                "username": "analyst",
                "roles": ["data_analyst", "dashboard_viewer"],
                "permissions": ["read_data", "execute_queries"]
            }
            
            # Cache user info for 1 hour
            self.redis_client.setex(
                f"auth_token:{token}",
                3600,
                json.dumps(user_info)
            )
            
            return user_info
        
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    async def _execute_query(self, request: QueryRequest, credentials: HTTPAuthorizationCredentials) -> DataResponse:
        """Execute SQL query"""
        
        start_time = datetime.now()
        
        # Authenticate user
        user_info = await self._authenticate_user(credentials)
        
        # Check permissions
        if "execute_queries" not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        try:
            # Check cache first
            cache_key = f"query_cache:{hash(request.sql + str(request.parameters))}"
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                self.logger.info("Returning cached query result")
                return DataResponse(**json.loads(cached_result))
            
            # Execute query (implement your database connection logic)
            import sqlalchemy
            
            engine = sqlalchemy.create_engine(self.database_config['connection_string'])
            
            df = pd.read_sql(
                request.sql,
                engine,
                params=request.parameters
            )
            
            # Apply limit
            if request.limit:
                df = df.head(request.limit)
            
            # Convert to desired format
            if request.format == "json":
                data = df.to_dict('records')
            elif request.format == "csv":
                data = df.to_csv(index=False)
            else:
                data = df.to_dict('records')  # Default to JSON
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = DataResponse(
                data=data,
                metadata={
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                    "query": request.sql,
                    "parameters": request.parameters
                },
                execution_time_ms=execution_time,
                row_count=len(df)
            )
            
            # Cache result for 5 minutes
            self.redis_client.setex(
                cache_key,
                300,
                response.json()
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
    
    async def _get_metrics(self, request: MetricRequest, credentials: HTTPAuthorizationCredentials) -> DataResponse:
        """Get predefined metrics"""
        
        # Authenticate user
        user_info = await self._authenticate_user(credentials)
        
        try:
            # Build metric query based on request
            metric_config = self._get_metric_config(request.metric_name)
            
            if not metric_config:
                raise HTTPException(status_code=404, detail=f"Metric '{request.metric_name}' not found")
            
            # Build SQL query from metric configuration
            sql_query = self._build_metric_query(metric_config, request)
            
            # Execute query
            query_request = QueryRequest(
                sql=sql_query,
                parameters={},
                limit=1000
            )
            
            return await self._execute_query(query_request, credentials)
            
        except Exception as e:
            self.logger.error(f"Metric calculation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Metric calculation failed: {str(e)}")
    
    def _get_metric_config(self, metric_name: str) -> Optional[Dict]:
        """Get metric configuration"""
        
        metric_configs = {
            "total_revenue": {
                "table": "sales_fact",
                "measure": "revenue",
                "aggregation": "sum",
                "dimensions": ["date", "region", "product"]
            },
            "customer_count": {
                "table": "customer_dim",
                "measure": "customer_id",
                "aggregation": "count_distinct",
                "dimensions": ["registration_date", "segment"]
            }
        }
        
        return metric_configs.get(metric_name)
    
    def _build_metric_query(self, metric_config: Dict, request: MetricRequest) -> str:
        """Build SQL query for metric"""
        
        table = metric_config['table']
        measure = metric_config['measure']
        aggregation = request.aggregation or metric_config['aggregation']
        
        # Build SELECT clause
        select_parts = []
        if request.dimensions:
            select_parts.extend(request.dimensions)
        
        # Add aggregation
        if aggregation == "sum":
            select_parts.append(f"SUM({measure}) as {request.metric_name}")
        elif aggregation == "count":
            select_parts.append(f"COUNT({measure}) as {request.metric_name}")
        elif aggregation == "count_distinct":
            select_parts.append(f"COUNT(DISTINCT {measure}) as {request.metric_name}")
        elif aggregation == "avg":
            select_parts.append(f"AVG({measure}) as {request.metric_name}")
        
        select_clause = ", ".join(select_parts)
        
        # Build WHERE clause
        where_conditions = []
        for key, value in request.filters.items():
            if isinstance(value, list):
                where_conditions.append(f"{key} IN ({','.join([f"'{v}'" for v in value])})")
            else:
                where_conditions.append(f"{key} = '{value}'")
        
        # Add time range filter
        if request.time_range:
            start_date = request.time_range.get('start_date')
            end_date = request.time_range.get('end_date')
            date_column = request.time_range.get('date_column', 'date')
            
            if start_date:
                where_conditions.append(f"{date_column} >= '{start_date}'")
            if end_date:
                where_conditions.append(f"{date_column} <= '{end_date}'")
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # Build GROUP BY clause
        group_by_clause = ", ".join(request.dimensions) if request.dimensions else ""
        
        # Construct final query
        query = f"SELECT {select_clause} FROM {table} WHERE {where_clause}"
        
        if group_by_clause:
            query += f" GROUP BY {group_by_clause}"
        
        return query
    
    async def _list_datasets(self, credentials: HTTPAuthorizationCredentials) -> Dict:
        """List available datasets"""
        
        # Authenticate user
        user_info = await self._authenticate_user(credentials)
        
        datasets = [
            {
                "name": "sales_fact",
                "description": "Sales transaction data",
                "schema": "analytics",
                "row_count": 1000000,
                "last_updated": "2024-01-15T10:30:00Z"
            },
            {
                "name": "customer_dim",
                "description": "Customer dimension data",
                "schema": "analytics", 
                "row_count": 50000,
                "last_updated": "2024-01-15T08:00:00Z"
            }
        ]
        
        return {"datasets": datasets}

# Example usage
database_config = {
    'connection_string': 'postgresql://user:password@localhost:5432/warehouse'
}

redis_config = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

api_service = DataAPIService(database_config, redis_config)
```

### 3.3 Visualization and Reporting Framework

#### Custom Dashboard Framework

```python
# custom_dashboard_framework.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

class CustomDashboardFramework:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.charts = {}
        self.data_sources = {}
    
    def register_data_source(self, name: str, data_source: Any) -> None:
        """Register a data source for the dashboard"""
        self.data_sources[name] = data_source
        self.logger.info(f"Registered data source: {name}")
    
    def create_kpi_card(self, title: str, value: float, 
                       previous_value: Optional[float] = None,
                       format_type: str = "number") -> Dict:
        """Create KPI card configuration"""
        
        change_percent = None
        if previous_value and previous_value != 0:
            change_percent = ((value - previous_value) / previous_value) * 100
        
        if format_type == "currency":
            formatted_value = f"${value:,.2f}"
        elif format_type == "percentage":
            formatted_value = f"{value:.1f}%"
        else:
            formatted_value = f"{value:,.0f}"
        
        return {
            "type": "kpi_card",
            "title": title,
            "value": formatted_value,
            "raw_value": value,
            "change_percent": change_percent,
            "trend": "up" if change_percent and change_percent > 0 else "down" if change_percent and change_percent < 0 else "neutral"
        }
    
    def create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str,
                         title: str, color_col: Optional[str] = None) -> go.Figure:
        """Create line chart"""
        
        if color_col:
            fig = px.line(data, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.line(data, x=x_col, y=y_col, title=title)
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            hovermode='x unified'
        )
        
        return fig
    
    def create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str,
                        title: str, orientation: str = "v") -> go.Figure:
        """Create bar chart"""
        
        if orientation == "h":
            fig = px.bar(data, x=y_col, y=x_col, orientation='h', title=title)
        else:
            fig = px.bar(data, x=x_col, y=y_col, title=title)
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
    
    def create_pie_chart(self, data: pd.DataFrame, values_col: str, 
                        names_col: str, title: str) -> go.Figure:
        """Create pie chart"""
        
        fig = px.pie(data, values=values_col, names=names_col, title=title)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, x_col: str, y_col: str,
                      values_col: str, title: str) -> go.Figure:
        """Create heatmap"""
        
        # Pivot data for heatmap
        pivot_data = data.pivot(index=y_col, columns=x_col, values=values_col)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
    
    def create_multi_chart_dashboard(self, dashboard_config: Dict) -> Dict:
        """Create multi-chart dashboard"""
        
        dashboard = {
            "title": dashboard_config['title'],
            "description": dashboard_config.get('description', ''),
            "charts": [],
            "kpis": [],
            "filters": dashboard_config.get('filters', []),
            "refresh_interval": dashboard_config.get('refresh_interval', 300)
        }
        
        # Create KPI cards
        for kpi_config in dashboard_config.get('kpis', []):
            kpi_data = self._get_kpi_data(kpi_config)
            kpi_card = self.create_kpi_card(
                title=kpi_config['title'],
                value=kpi_data['current_value'],
                previous_value=kpi_data.get('previous_value'),
                format_type=kpi_config.get('format_type', 'number')
            )
            dashboard['kpis'].append(kpi_card)
        
        # Create charts
        for chart_config in dashboard_config.get('charts', []):
            chart_data = self._get_chart_data(chart_config)
            
            if chart_config['type'] == 'line':
                chart = self.create_line_chart(
                    data=chart_data,
                    x_col=chart_config['x_column'],
                    y_col=chart_config['y_column'],
                    title=chart_config['title'],
                    color_col=chart_config.get('color_column')
                )
            elif chart_config['type'] == 'bar':
                chart = self.create_bar_chart(
                    data=chart_data,
                    x_col=chart_config['x_column'],
                    y_col=chart_config['y_column'],
                    title=chart_config['title'],
                    orientation=chart_config.get('orientation', 'v')
                )
            elif chart_config['type'] == 'pie':
                chart = self.create_pie_chart(
                    data=chart_data,
                    values_col=chart_config['values_column'],
                    names_col=chart_config['names_column'],
                    title=chart_config['title']
                )
            elif chart_config['type'] == 'heatmap':
                chart = self.create_heatmap(
                    data=chart_data,
                    x_col=chart_config['x_column'],
                    y_col=chart_config['y_column'],
                    values_col=chart_config['values_column'],
                    title=chart_config['title']
                )
            
            dashboard['charts'].append({
                'id': f"chart_{len(dashboard['charts'])}",
                'config': chart_config,
                'figure': chart
            })
        
        return dashboard
    
    def _get_kpi_data(self, kpi_config: Dict) -> Dict:
        """Get KPI data from data source"""
        
        data_source = self.data_sources.get(kpi_config['data_source'])
        if not data_source:
            raise ValueError(f"Data source {kpi_config['data_source']} not found")
        
        # This would typically execute a query or call an API
        # For demo purposes, return mock data
        return {
            'current_value': 125000.0,
            'previous_value': 120000.0
        }
    
    def _get_chart_data(self, chart_config: Dict) -> pd.DataFrame:
        """Get chart data from data source"""
        
        data_source = self.data_sources.get(chart_config['data_source'])
        if not data_source:
            raise ValueError(f"Data source {chart_config['data_source']} not found")
        
        # This would typically execute a query or call an API
        # For demo purposes, return mock data
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'revenue': [1000 + i * 100 for i in range(30)],
            'region': ['North', 'South'] * 15
        })

# Example dashboard configuration
dashboard_config = {
    "title": "Sales Performance Dashboard",
    "description": "Comprehensive sales analytics and KPI tracking",
    "refresh_interval": 300,
    "kpis": [
        {
            "title": "Total Revenue",
            "data_source": "sales_db",
            "format_type": "currency"
        },
        {
            "title": "Customer Count",
            "data_source": "customer_db",
            "format_type": "number"
        }
    ],
    "charts": [
        {
            "type": "line",
            "title": "Revenue Trend",
            "data_source": "sales_db",
            "x_column": "date",
            "y_column": "revenue"
        },
        {
            "type": "pie",
            "title": "Revenue by Region",
            "data_source": "sales_db",
            "values_column": "revenue",
            "names_column": "region"
        }
    ],
    "filters": [
        {
            "name": "date_range",
            "type": "date_range",
            "default": "last_30_days"
        },
        {
            "name": "region",
            "type": "multi_select",
            "options": ["North", "South", "East", "West"]
        }
    ]
}
```

---

## 🎯 Performance Optimization Strategies

### 4.1 Query Optimization Framework

#### SQL Query Performance Analyzer

```python
# query_performance_analyzer.py
import sqlparse
import time
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, text

class QueryPerformanceAnalyzer:
    def __init__(self, database_config: Dict):
        self.engine = create_engine(database_config['connection_string'])
        self.logger = logging.getLogger(__name__)
        self.query_cache = {}
    
    def analyze_query_performance(self, sql_query: str, parameters: Dict = None) -> Dict:
        """Analyze query performance and provide optimization suggestions"""
        
        start_time = time.time()
        
        # Parse SQL query
        parsed_query = sqlparse.parse(sql_query)[0]
        query_type = self._get_query_type(parsed_query)
        
        # Execute query with timing
        execution_metrics = self._execute_with_metrics(sql_query, parameters)
        
        # Analyze query structure
        structure_analysis = self._analyze_query_structure(parsed_query)
        
        # Get execution plan
        execution_plan = self._get_execution_plan(sql_query, parameters)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            parsed_query, execution_metrics, execution_plan
        )
        
        analysis_time = time.time() - start_time
        
        return {
            'query_type': query_type,
            'execution_metrics': execution_metrics,
            'structure_analysis': structure_analysis,
            'execution_plan': execution_plan,
            'optimization_suggestions': optimization_suggestions,
            'analysis_time_seconds': analysis_time
        }
    
    def _get_query_type(self, parsed_query) -> str:
        """Determine the type of SQL query"""
        
        first_token = str(parsed_query.tokens[0]).upper().strip()
        
        if first_token.startswith('SELECT'):
            return 'SELECT'
        elif first_token.startswith('INSERT'):
            return 'INSERT'
        elif first_token.startswith('UPDATE'):
            return 'UPDATE'
        elif first_token.startswith('DELETE'):
            return 'DELETE'
        else:
            return 'OTHER'
    
    def _execute_with_metrics(self, sql_query: str, parameters: Dict = None) -> Dict:
        """Execute query and collect performance metrics"""
        
        start_time = time.time()
        
        try:
            with self.engine.connect() as connection:
                if parameters:
                    result = connection.execute(text(sql_query), parameters)
                else:
                    result = connection.execute(text(sql_query))
                
                # Fetch results to measure full execution time
                rows = result.fetchall()
                row_count = len(rows)
                
                execution_time = time.time() - start_time
                
                return {
                    'execution_time_seconds': execution_time,
                    'row_count': row_count,
                    'rows_per_second': row_count / execution_time if execution_time > 0 else 0,
                    'status': 'success'
                }
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                'execution_time_seconds': execution_time,
                'row_count': 0,
                'rows_per_second': 0,
                'status': 'error',
                'error_message': str(e)
            }
    
    def _analyze_query_structure(self, parsed_query) -> Dict:
        """Analyze the structure of the SQL query"""
        
        analysis = {
            'has_joins': False,
            'join_count': 0,
            'has_subqueries': False,
            'subquery_count': 0,
            'has_aggregations': False,
            'has_order_by': False,
            'has_group_by': False,
            'has_where_clause': False,
            'estimated_complexity': 'low'
        }
        
        query_str = str(parsed_query).upper()
        
        # Check for joins
        join_keywords = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']
        for join_keyword in join_keywords:
            if join_keyword in query_str:
                analysis['has_joins'] = True
                analysis['join_count'] += query_str.count(join_keyword)
        
        # Check for subqueries
        if '(' in query_str and 'SELECT' in query_str:
            analysis['has_subqueries'] = True
            analysis['subquery_count'] = query_str.count('SELECT') - 1
        
        # Check for aggregations
        agg_functions = ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(']
        for agg_func in agg_functions:
            if agg_func in query_str:
                analysis['has_aggregations'] = True
                break
        
        # Check for other clauses
        analysis['has_order_by'] = 'ORDER BY' in query_str
        analysis['has_group_by'] = 'GROUP BY' in query_str
        analysis['has_where_clause'] = 'WHERE' in query_str
        
        # Estimate complexity
        complexity_score = 0
        complexity_score += analysis['join_count'] * 2
        complexity_score += analysis['subquery_count'] * 3
        complexity_score += 1 if analysis['has_aggregations'] else 0
        complexity_score += 1 if analysis['has_group_by'] else 0
        
        if complexity_score <= 2:
            analysis['estimated_complexity'] = 'low'
        elif complexity_score <= 5:
            analysis['estimated_complexity'] = 'medium'
        else:
            analysis['estimated_complexity'] = 'high'
        
        return analysis
    
    def _get_execution_plan(self, sql_query: str, parameters: Dict = None) -> Dict:
        """Get query execution plan"""
        
        try:
            with self.engine.connect() as connection:
                # Get execution plan (PostgreSQL example)
                plan_query = f"EXPLAIN (FORMAT JSON, ANALYZE) {sql_query}"
                
                if parameters:
                    result = connection.execute(text(plan_query), parameters)
                else:
                    result = connection.execute(text(plan_query))
                
                plan_data = result.fetchone()[0]
                
                return {
                    'plan_available': True,
                    'plan_data': plan_data,
                    'total_cost': plan_data[0]['Plan']['Total Cost'] if plan_data else 0,
                    'execution_time': plan_data[0]['Execution Time'] if plan_data else 0
                }
        
        except Exception as e:
            return {
                'plan_available': False,
                'error': str(e),
                'total_cost': 0,
                'execution_time': 0
            }
    
    def _generate_optimization_suggestions(self, parsed_query, execution_metrics: Dict, 
                                         execution_plan: Dict) -> List[str]:
        """Generate optimization suggestions based on analysis"""
        
        suggestions = []
        
        # Performance-based suggestions
        if execution_metrics['execution_time_seconds'] > 5:
            suggestions.append("Query execution time is high (>5s). Consider adding indexes or optimizing joins.")
        
        if execution_metrics['rows_per_second'] < 1000:
            suggestions.append("Low throughput detected. Consider query optimization or hardware scaling.")
        
        # Structure-based suggestions
        query_str = str(parsed_query).upper()
        
        if 'SELECT *' in query_str:
            suggestions.append("Avoid SELECT * - specify only needed columns to reduce data transfer.")
        
        if 'ORDER BY' in query_str and 'LIMIT' not in query_str:
            suggestions.append("ORDER BY without LIMIT can be expensive. Consider adding LIMIT if appropriate.")
        
        if query_str.count('JOIN') > 3:
            suggestions.append("Multiple joins detected. Consider denormalizing data or using materialized views.")
        
        # Execution plan-based suggestions
        if execution_plan['plan_available'] and execution_plan['total_cost'] > 1000:
            suggestions.append("High query cost detected. Review execution plan for sequential scans.")
        
        return suggestions

# Example usage
database_config = {
    'connection_string': 'postgresql://user:password@localhost:5432/warehouse'
}

analyzer = QueryPerformanceAnalyzer(database_config)

# Analyze a query
query = """
SELECT c.customer_name, SUM(o.order_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2024-01-01'
GROUP BY c.customer_name
ORDER BY total_spent DESC
"""

analysis_result = analyzer.analyze_query_performance(query)
```

### 4.2 Caching Strategy Implementation

#### Multi-Level Caching Framework

```python
# caching_framework.py
import redis
import memcache
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd

class MultiLevelCachingFramework:
    def __init__(self, redis_config: Dict, memcache_config: Dict = None):
        self.redis_client = redis.Redis(**redis_config)
        self.memcache_client = memcache.Client([memcache_config['server']]) if memcache_config else None
        self.logger = logging.getLogger(__name__)
        
        # Cache levels configuration
        self.cache_levels = {
            'L1': {'type': 'memory', 'ttl': 300, 'max_size': '100MB'},      # 5 minutes
            'L2': {'type': 'redis', 'ttl': 3600, 'max_size': '1GB'},       # 1 hour
            'L3': {'type': 'disk', 'ttl': 86400, 'max_size': '10GB'}       # 1 day
        }
    
    def generate_cache_key(self, query: str, parameters: Dict = None, 
                          user_context: Dict = None) -> str:
        """Generate consistent cache key"""
        
        # Normalize query (remove extra whitespace, convert to lowercase)
        normalized_query = ' '.join(query.lower().split())
        
        # Create key components
        key_components = [normalized_query]
        
        if parameters:
            key_components.append(json.dumps(parameters, sort_keys=True))
        
        if user_context:
            # Include user-specific context that affects results
            relevant_context = {
                'user_id': user_context.get('user_id'),
                'roles': sorted(user_context.get('roles', [])),
                'permissions': sorted(user_context.get('permissions', []))
            }
            key_components.append(json.dumps(relevant_context, sort_keys=True))
        
        # Generate hash
        key_string = '|'.join(key_components)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"query_cache:{cache_key}"
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get result from cache (multi-level lookup)"""
        
        # Try L1 cache (memory) first
        if hasattr(self, '_memory_cache'):
            if cache_key in self._memory_cache:
                self.logger.debug(f"Cache hit (L1): {cache_key}")
                return self._memory_cache[cache_key]
        
        # Try L2 cache (Redis)
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                result = json.loads(cached_data)
                self.logger.debug(f"Cache hit (L2): {cache_key}")
                
                # Promote to L1 cache
                self._set_memory_cache(cache_key, result)
                
                return result
        except Exception as e:
            self.logger.warning(f"Redis cache error: {e}")
        
        # Try L3 cache (Memcache) if available
        if self.memcache_client:
            try:
                cached_data = self.memcache_client.get(cache_key)
                if cached_data:
                    self.logger.debug(f"Cache hit (L3): {cache_key}")
                    
                    # Promote to higher levels
                    self._set_memory_cache(cache_key, cached_data)
                    self.redis_client.setex(cache_key, 3600, json.dumps(cached_data))
                    
                    return cached_data
            except Exception as e:
                self.logger.warning(f"Memcache error: {e}")
        
        self.logger.debug(f"Cache miss: {cache_key}")
        return None
    
    def set_cached_result(self, cache_key: str, result: Dict, 
                         cache_level: str = 'L2', custom_ttl: int = None) -> None:
        """Set result in cache"""
        
        ttl = custom_ttl or self.cache_levels[cache_level]['ttl']
        
        try:
            # Always cache in Redis (L2)
            self.redis_client.setex(cache_key, ttl, json.dumps(result))
            
            # Also cache in memory (L1) for frequently accessed data
            if cache_level in ['L1', 'L2']:
                self._set_memory_cache(cache_key, result)
            
            # Cache in Memcache (L3) for large datasets
            if cache_level == 'L3' and self.memcache_client:
                self.memcache_client.set(cache_key, result, time=ttl)
            
            self.logger.debug(f"Cached result ({cache_level}): {cache_key}")
            
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
    
    def _set_memory_cache(self, cache_key: str, result: Dict) -> None:
        """Set result in memory cache with size limits"""
        
        if not hasattr(self, '_memory_cache'):
            self._memory_cache = {}
        
        # Simple size management (remove oldest entries if needed)
        if len(self._memory_cache) > 1000:  # Max 1000 entries
            # Remove 10% of oldest entries
            keys_to_remove = list(self._memory_cache.keys())[:100]
            for key in keys_to_remove:
                del self._memory_cache[key]
        
        self._memory_cache[cache_key] = result
    
    def invalidate_cache(self, pattern: str = None, user_id: str = None) -> int:
        """Invalidate cache entries"""
        
        invalidated_count = 0
        
        if pattern:
            # Invalidate by pattern
            try:
                keys = self.redis_client.keys(f"*{pattern}*")
                if keys:
                    self.redis_client.delete(*keys)
                    invalidated_count = len(keys)
                    self.logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
            except Exception as e:
                self.logger.error(f"Cache invalidation error: {e}")
        
        elif user_id:
            # Invalidate user-specific cache
            try:
                keys = self.redis_client.keys(f"*user_id*{user_id}*")
                if keys:
                    self.redis_client.delete(*keys)
                    invalidated_count = len(keys)
                    self.logger.info(f"Invalidated {invalidated_count} cache entries for user: {user_id}")
            except Exception as e:
                self.logger.error(f"User cache invalidation error: {e}")
        
        return invalidated_count
    
    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics"""
        
        try:
            redis_info = self.redis_client.info()
            
            stats = {
                'redis': {
                    'used_memory': redis_info.get('used_memory_human', 'N/A'),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0)
                }
            }
            
            # Calculate hit rate
            hits = stats['redis']['keyspace_hits']
            misses = stats['redis']['keyspace_misses']
            total_requests = hits + misses
            
            if total_requests > 0:
                stats['redis']['hit_rate'] = (hits / total_requests) * 100
            else:
                stats['redis']['hit_rate'] = 0
            
            # Memory cache stats
            if hasattr(self, '_memory_cache'):
                stats['memory'] = {
                    'entries': len(self._memory_cache),
                    'estimated_size_mb': len(str(self._memory_cache)) / (1024 * 1024)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cache statistics: {e}")
            return {}

# Example usage
redis_config = {
    'host': 'localhost',
    'port': 6379,
    'db': 1
}

memcache_config = {
    'server': '127.0.0.1:11211'
}

cache_framework = MultiLevelCachingFramework(redis_config, memcache_config)

# Cache a query result
query = "SELECT * FROM sales WHERE date >= '2024-01-01'"
parameters = {'start_date': '2024-01-01'}
user_context = {'user_id': 'user123', 'roles': ['analyst']}

cache_key = cache_framework.generate_cache_key(query, parameters, user_context)

# Try to get from cache
cached_result = cache_framework.get_cached_result(cache_key)

if not cached_result:
    # Execute query and cache result
    result = {
        'data': [{'id': 1, 'amount': 100}, {'id': 2, 'amount': 200}],
        'metadata': {'row_count': 2, 'execution_time': 0.5}
    }
    cache_framework.set_cached_result(cache_key, result, 'L2')
```

---

## 📈 Implementation Checklist

### Analytics & Consumption Implementation Checklist

**Phase 1: Architecture Planning**
- [ ] Define user personas and access patterns
- [ ] Select appropriate BI and visualization tools
- [ ] Design data consumption APIs
- [ ] Plan caching and performance optimization strategies
- [ ] Establish security and authentication requirements

**Phase 2: Core Implementation**
- [ ] Set up business intelligence platforms (Superset/Power BI/Tableau)
- [ ] Implement data API services with authentication
- [ ] Create custom dashboard frameworks
- [ ] Build query optimization and performance monitoring
- [ ] Implement multi-level caching strategies

**Phase 3: Advanced Features**
- [ ] Real-time analytics capabilities
- [ ] Self-service analytics tools
- [ ] Embedded analytics for external applications
- [ ] Advanced visualization and reporting
- [ ] Mobile and responsive dashboard support

**Phase 4: Optimization & Monitoring**
- [ ] Performance monitoring and alerting
- [ ] Query optimization and indexing
- [ ] Cache hit rate optimization
- [ ] User experience analytics
- [ ] Cost optimization and resource management

---

## 🎯 Success Metrics

### Analytics & Consumption KPIs

**User Adoption:**
- Dashboard usage frequency: Daily active users
- Query volume: Queries per day/hour
- Self-service adoption: % of users creating own reports
- Mobile usage: % of mobile vs desktop access

**Performance Metrics:**
- Query response time: 95th percentile < 3 seconds
- Dashboard load time: < 5 seconds
- Cache hit rate: > 80%
- API availability: > 99.9% uptime

**Business Impact:**
- Time to insight: Reduction in report generation time
- Decision speed: Faster business decision cycles
- Cost efficiency: Cost per query/dashboard view
- User satisfaction: Net Promoter Score (NPS)

---

## 🔗 Integration Points

### Related Documents
- [Data Storage (05)](05_data_storage.md) - Data warehouse and lake integration
- [Data Processing (06)](06_data_processing.md) - Processed data consumption
- [Testing & Quality (08)](08_testing_quality.md) - Analytics testing strategies
- [Monitoring & Support (10)](10_monitoring_support.md) - Analytics monitoring
- [Advanced Analytics Topics](07b_analytics_consumption_advanced.md) - Real-time analytics and ML insights

### External Dependencies
- Business Intelligence platforms (Tableau, Power BI, Superset)
- Visualization libraries (Plotly, D3.js, Chart.js)
- API frameworks (FastAPI, GraphQL, REST)
- Caching systems (Redis, Memcached)
- Authentication providers (Auth0, Okta, LDAP)

---

**Next Steps:** Proceed to [Advanced Analytics Topics (07b)](07b_analytics_consumption_advanced.md) for real-time analytics, ML-powered insights, and advanced visualization techniques.
