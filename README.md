# Local Lifestyle Agent

A production-ready local lifestyle recommendation system that helps users find venues (starting with afternoon tea) based on natural language queries.

## Features

- **Natural Language Processing**: Parse user intent from conversational queries
- **Multi-iteration Planning**: Intelligent planning with evaluation and replanning
- **Production Infrastructure**: Error handling, logging, metrics, caching, validation
- **Resilient Architecture**: Fallback strategies and graceful degradation
- **Comprehensive Testing**: 263+ unit tests with 100% pass rate

## Architecture

```
User Prompt → Planner (LLM) → Executor (Google Places) → Evaluator → Orchestrator → Final Plan
```

### Core Components

- **Planner**: Normalizes user intent and generates execution plans using LLM
- **Executor**: Executes tool calls against Google Places API
- **Evaluator**: Scores and ranks candidate venues
- **Orchestrator**: Coordinates the complete recommendation workflow

### Infrastructure

- **Error Handler**: Unified error handling with retry logic and exponential backoff
- **Config Manager**: Environment-based configuration with validation
- **Structured Logger**: JSON logging with request ID tracking and sensitive data masking
- **Metrics Collector**: Prometheus-compatible metrics for monitoring
- **Cache**: LRU+TTL caching for API responses
- **Data Validator**: Input validation and malicious content detection

## Quick Start

### Prerequisites

- **Python 3.10+** (Python 2.x is NOT supported)
- OpenAI API key (for LLM, supports gpt-4o-mini or gpt-4o)
- Google Places API key

### Installation

```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="sk-..."
export GOOGLE_PLACES_API_KEY="AIza..."
```

**Important**: Always use `python3` command or activate the virtual environment. The system `python` command may point to Python 2.7 which is not compatible.

### Configuration

Create a `config.json` file (optional, see `config.example.json`):

```json
{
  "openai_api_key": "your-key",
  "google_places_api_key": "your-key",
  "openai_model": "gpt-4o-mini",
  "llm_timeout_seconds": 30,
  "places_timeout_seconds": 10,
  "cache_enabled": true,
  "cache_ttl_seconds": 3600,
  "log_level": "INFO"
}
```

Environment variables take precedence over config file values.

### Run Examples

**Quick Start (Recommended)**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux

# Run the script
python scripts/quick_start.py
```

**Complete Example (All Features)**:
```bash
python scripts/complete_example.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=local_lifestyle_agent --cov-report=html

# Run specific test module
pytest tests/test_orchestrator.py -v
```

## Project Structure

```
Project1/
├── local_lifestyle_agent/          # Main package
│   ├── infrastructure/             # Infrastructure modules
│   │   ├── error_handler.py        # Error handling
│   │   ├── config.py               # Configuration management
│   │   ├── logger.py               # Structured logging
│   │   ├── metrics.py              # Metrics collection
│   │   ├── cache.py                # Caching
│   │   └── validator.py            # Data validation
│   ├── adapters/                   # External API adapters
│   │   └── google_places.py        # Google Places API
│   ├── schemas.py                  # Pydantic models
│   ├── llm_client.py               # OpenAI API wrapper
│   ├── planner.py                  # Intent normalization and planning
│   ├── executor.py                 # Tool execution
│   ├── evaluator.py                # Candidate evaluation
│   ├── orchestrator.py             # Main orchestration loop
│   └── renderer.py                 # Output rendering
├── scripts/                        # Example scripts
│   ├── quick_start.py              # Quick start example
│   └── complete_example.py         # Complete feature demo
├── tests/                          # Test suite (263+ tests)
├── docs/                           # Documentation
│   ├── TravelAgentDesign.md        # Design document
│   ├── deployment.md               # Deployment guide
│   └── troubleshooting.md          # Troubleshooting guide
├── requirements.txt                # Python dependencies
├── config.example.json             # Example configuration
├── USAGE_GUIDE.md                  # Detailed usage guide
└── README.md                       # This file
```

## Configuration Options

### API Keys
- `OPENAI_API_KEY` / `openai_api_key`: OpenAI API key (required)
- `GOOGLE_PLACES_API_KEY` / `google_places_api_key`: Google Places API key (required)

### Timeouts
- `LLM_TIMEOUT_SECONDS` / `llm_timeout_seconds`: LLM API timeout (default: 30)
- `PLACES_TIMEOUT_SECONDS` / `places_timeout_seconds`: Places API timeout (default: 10)

### Retry Configuration
- `MAX_RETRIES` / `max_retries`: Maximum retry attempts (default: 3)
- `RETRY_BASE_DELAY` / `retry_base_delay`: Base delay for exponential backoff (default: 1.0)

### Cache Configuration
- `CACHE_ENABLED` / `cache_enabled`: Enable caching (default: true)
- `CACHE_TTL_SECONDS` / `cache_ttl_seconds`: Cache TTL (default: 3600)
- `CACHE_MAX_SIZE` / `cache_max_size`: Maximum cache entries (default: 1000)

### Logging Configuration
- `LOG_LEVEL` / `log_level`: Logging level (default: INFO)
- `LOG_FORMAT` / `log_format`: Log format - json or text (default: json)
- `LOG_FILE` / `log_file`: Log file path (optional)

## Error Handling

The system includes comprehensive error handling:

- **Automatic Retries**: Transient errors (timeouts, rate limits) are automatically retried with exponential backoff
- **Fallback Strategies**: When LLM is unavailable, rule-based fallback generates default plans
- **Graceful Degradation**: Component failures don't crash the system
- **Structured Error Responses**: All errors return consistent ErrorResponse objects

## Monitoring

Prometheus-compatible metrics are available:

- `requests_total`: Total number of requests
- `request_duration_seconds`: Request duration histogram
- `api_calls_total`: External API calls counter
- `cache_hits_total` / `cache_misses_total`: Cache performance
- `errors_total`: Error counter by type
- `active_requests`: Current active requests gauge
