# AgenticMaid Legacy

Backward compatibility and legacy API methods for the AgenticMaid framework. This package provides support for older AgenticMaid implementations and helps users migrate from legacy versions to the modern modular architecture.

## Overview

AgenticMaid Legacy is designed to ease the transition from older versions of AgenticMaid to the new modular architecture. It provides backward compatibility wrappers, legacy API methods, and migration utilities to help users upgrade their existing implementations without breaking changes.

## Features

- **Legacy API Compatibility**: Support for older API calling methods
- **Configuration Migration**: Tools to convert legacy configurations to modern format
- **Backward Compatibility Wrappers**: Maintain compatibility with existing code
- **MCP STDIO Adapter**: Legacy MCP communication methods
- **Deprecation Warnings**: Clear guidance on deprecated functionality
- **Migration Utilities**: Automated tools to help with the upgrade process

## Installation

### Basic Installation

```bash
pip install agenticmaid-legacy
```

### With Additional Features

```bash
# With MCP support
pip install agenticmaid-legacy[mcp]

# With full modern AgenticMaid integration
pip install agenticmaid-legacy[full]

# Development tools
pip install agenticmaid-legacy[dev]
```

## Quick Start

### Using Legacy API Methods

```python
from agenticmaid_legacy import LegacyAgenticMaid

# Initialize legacy client
client = LegacyAgenticMaid({
    "ai_service": {
        "provider": "Google",
        "model": "gemini-2.5-pro",
        "api_key": "your_api_key"
    }
})

client.initialize()

# Use legacy API calling method
response = client.call_api_legacy(
    endpoint="chat",
    data={"message": "Hello, world!"},
    method="POST"
)

print(response)
```

### Configuration Migration

```python
from agenticmaid_legacy import LegacyConfigConverter, migrate_to_modern

# Convert configuration programmatically
legacy_config = {
    "ai_service": {
        "provider": "Google",
        "model": "gemini-2.5-pro",
        "api_key": "your_api_key"
    },
    "mcp_config": {
        "server1": {"url": "http://localhost:8001"}
    }
}

modern_config = LegacyConfigConverter.convert_to_modern(legacy_config)
print(modern_config)

# Or migrate configuration files
migrate_to_modern("legacy_config.json", "modern_config.json")
```

### Using Legacy MCP Adapter

```python
from agenticmaid_legacy import MCPStdioAdapter

# Initialize legacy MCP adapter
adapter = MCPStdioAdapter()

# Execute legacy MCP commands
result = adapter.execute_command("list_tools")
print(result)
```

## Migration Guide

### Step 1: Install Legacy Package

```bash
pip install agenticmaid-legacy
```

### Step 2: Migrate Configuration

```python
from agenticmaid_legacy import migrate_to_modern

# Migrate your existing configuration
migrate_to_modern("old_config.json", "new_config.json")
```

### Step 3: Update Imports

**Before (Legacy):**
```python
from agenticmaid import AgenticMaid
```

**After (Modern with Legacy Support):**
```python
from agenticmaid_legacy import LegacyAgenticMaid as AgenticMaid
```

### Step 4: Gradual Migration

Use the legacy package as a bridge while gradually updating your code:

```python
from agenticmaid_legacy import LegacyAgenticMaid
from agenticmaid import AgenticMaid  # Modern version

# Start with legacy
legacy_client = LegacyAgenticMaid(legacy_config)

# Gradually migrate to modern
modern_config = LegacyConfigConverter.convert_to_modern(legacy_config)
modern_client = AgenticMaid(modern_config)
```

## Legacy API Reference

### LegacyAgenticMaid Class

The main legacy compatibility class:

```python
class LegacyAgenticMaid:
    def __init__(self, config=None):
        """Initialize with legacy configuration format."""
        
    def initialize(self):
        """Initialize the client (synchronous)."""
        
    def call_api_legacy(self, endpoint: str, data: dict = None, method: str = "POST"):
        """Legacy API calling method."""
        
    def run_task_legacy(self, task_name: str, params: dict = None):
        """Legacy task execution method."""
        
    def get_config_legacy(self):
        """Get configuration in legacy format."""
        
    def set_config_legacy(self, config: dict):
        """Set configuration in legacy format."""
```

### LegacyConfigConverter Class

Configuration format conversion utilities:

```python
class LegacyConfigConverter:
    @staticmethod
    def convert_to_modern(legacy_config: dict) -> dict:
        """Convert legacy configuration to modern format."""
        
    @staticmethod
    def convert_to_legacy(modern_config: dict) -> dict:
        """Convert modern configuration to legacy format."""
```

### MCPStdioAdapter Class

Legacy MCP communication adapter:

```python
class MCPStdioAdapter:
    def execute_command(self, command: str, params: dict = None):
        """Execute legacy MCP command."""
        
    def list_tools(self):
        """List available MCP tools."""
        
    def call_tool(self, tool_name: str, arguments: dict = None):
        """Call a specific MCP tool."""
```

## Configuration Format Differences

### Legacy Format

```json
{
  "ai_service": {
    "provider": "Google",
    "model": "gemini-2.5-pro",
    "api_key": "your_api_key"
  },
  "mcp_config": {
    "server1": {
      "url": "http://localhost:8001"
    }
  },
  "messaging_config": {
    "telegram": {
      "bot_token": "your_bot_token"
    }
  }
}
```

### Modern Format

```json
{
  "ai_services": {
    "default_service": {
      "provider": "Google",
      "model": "gemini-2.5-pro",
      "api_key": "your_api_key"
    }
  },
  "default_llm_service_name": "default_service",
  "mcp_servers": {
    "server1": {
      "url": "http://localhost:8001"
    }
  },
  "messaging_clients": {
    "telegram": {
      "type": "telegram",
      "enabled": true,
      "config": {
        "bot_token": "your_bot_token"
      }
    }
  }
}
```

## Deprecation Warnings

The legacy package includes deprecation warnings to help guide migration:

```python
import warnings
from agenticmaid_legacy import deprecated_function

@deprecated_function
def old_method():
    """This method is deprecated."""
    pass

# Using deprecated functionality will show warnings
old_method()  # Shows deprecation warning
```

## Command Line Tools

### Configuration Migration Tool

```bash
# Migrate configuration file
agenticmaid-migrate legacy_config.json modern_config.json
```

## Best Practices

### 1. Gradual Migration

Don't migrate everything at once. Use the legacy package as a bridge:

1. Install `agenticmaid-legacy`
2. Update imports to use legacy wrappers
3. Migrate configuration format
4. Gradually update code to use modern APIs
5. Remove legacy dependencies

### 2. Testing During Migration

Test both legacy and modern implementations side by side:

```python
# Test legacy implementation
legacy_result = legacy_client.call_api_legacy("endpoint", data)

# Test modern implementation
modern_result = await modern_client.run_mcp_interaction(messages, service)

# Compare results
assert legacy_result["status"] == "success"
assert modern_result is not None
```

### 3. Configuration Validation

Validate configurations after migration:

```python
from agenticmaid_legacy import LegacyConfigConverter

# Validate conversion
legacy_config = load_legacy_config()
modern_config = LegacyConfigConverter.convert_to_modern(legacy_config)
back_to_legacy = LegacyConfigConverter.convert_to_legacy(modern_config)

# Ensure round-trip conversion works
assert legacy_config == back_to_legacy
```

## Troubleshooting

### Common Migration Issues

1. **Import Errors**: Make sure all required packages are installed
2. **Configuration Errors**: Use the migration tool to convert configurations
3. **API Changes**: Check deprecation warnings for guidance
4. **Dependency Conflicts**: Install legacy package in a separate environment if needed

### Getting Help

- Check the [Migration Guide](MIGRATION.md) for detailed instructions
- Review [examples/](examples/) for migration examples
- Report issues on [GitHub Issues](https://github.com/agenticmaid/agenticmaid-legacy/issues)

## Roadmap

### Deprecation Timeline

- **v1.0**: Full legacy support with deprecation warnings
- **v1.5**: Reduced legacy support, migration tools enhanced
- **v2.0**: Legacy support removed, migration tools only

### Migration Support

The legacy package will be maintained for at least 12 months to provide adequate migration time.

## Development

### Running Tests

```bash
pip install agenticmaid-legacy[dev]
pytest
```

### Code Style

```bash
black agenticmaid_legacy/
flake8 agenticmaid_legacy/
mypy agenticmaid_legacy/
```

## License

MIT License - see LICENSE file for details.

## Links

- [Main Project](https://github.com/agenticmaid/agenticmaid)
- [Core Package](https://github.com/agenticmaid/agenticmaid-core)
- [Migration Guide](MIGRATION.md)
- [Documentation](https://github.com/agenticmaid/agenticmaid-legacy#readme)
- [Issues](https://github.com/agenticmaid/agenticmaid-legacy/issues)
- [PyPI](https://pypi.org/project/agenticmaid-legacy/)
