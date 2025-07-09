"""
Unit tests for the ClientFactory class.
"""

import pytest
from unittest.mock import Mock, patch

from messaging_system.clients.client_factory import ClientFactory
from messaging_system.clients.base_client import BaseClient
from messaging_system.core.exceptions import ConfigurationError, ValidationError


class MockTestClient(BaseClient):
    """Mock client for testing factory."""
    
    def get_client_type(self) -> str:
        return "test_client"
    
    async def connect(self) -> bool:
        return True
    
    async def disconnect(self) -> bool:
        return True
    
    async def send_message(self, message) -> bool:
        return True
    
    async def receive_message(self, timeout=None):
        return None


class TestClientFactory:
    """Test cases for ClientFactory."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear any existing registrations
        ClientFactory._client_types.clear()
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clear registrations
        ClientFactory._client_types.clear()
    
    def test_register_client_type(self):
        """Test registering a client type."""
        ClientFactory.register_client_type("test", MockTestClient)
        
        assert "test" in ClientFactory._client_types
        assert ClientFactory._client_types["test"] == MockTestClient
    
    def test_register_invalid_client_type(self):
        """Test registering an invalid client type."""
        class InvalidClient:
            pass
        
        with pytest.raises(ValueError):
            ClientFactory.register_client_type("invalid", InvalidClient)
    
    def test_unregister_client_type(self):
        """Test unregistering a client type."""
        ClientFactory.register_client_type("test", MockTestClient)
        ClientFactory.unregister_client_type("test")
        
        assert "test" not in ClientFactory._client_types
    
    def test_get_available_types(self):
        """Test getting available client types."""
        ClientFactory.register_client_type("test1", MockTestClient)
        ClientFactory.register_client_type("test2", MockTestClient)
        
        available_types = ClientFactory.get_available_types()
        assert "test1" in available_types
        assert "test2" in available_types
        assert len(available_types) == 2
    
    def test_create_client_success(self):
        """Test successful client creation."""
        ClientFactory.register_client_type("test", MockTestClient)
        
        config = {"test_param": "value"}
        
        with patch('messaging_system.core.config_validator.ConfigValidator.validate_client_config'):
            client = ClientFactory.create_client("test_client", "test", config)
        
        assert isinstance(client, MockTestClient)
        assert client.client_id == "test_client"
        assert client.config == config
    
    def test_create_client_unsupported_type(self):
        """Test creating client with unsupported type."""
        with pytest.raises(ConfigurationError) as exc_info:
            ClientFactory.create_client("test_client", "unsupported", {})
        
        assert "Unsupported client type" in str(exc_info.value)
    
    @patch('messaging_system.core.config_validator.ConfigValidator.validate_client_config')
    def test_create_client_validation_error(self, mock_validate):
        """Test client creation with validation error."""
        ClientFactory.register_client_type("test", MockTestClient)
        mock_validate.side_effect = ValidationError("Invalid config")
        
        with pytest.raises(ConfigurationError) as exc_info:
            ClientFactory.create_client("test_client", "test", {})
        
        assert "Invalid configuration" in str(exc_info.value)
    
    def test_create_client_instantiation_error(self):
        """Test client creation with instantiation error."""
        class FailingClient(BaseClient):
            def __init__(self, client_id, config):
                raise Exception("Instantiation failed")
            
            def get_client_type(self):
                return "failing"
            
            async def connect(self):
                return True
            
            async def disconnect(self):
                return True
            
            async def send_message(self, message):
                return True
            
            async def receive_message(self, timeout=None):
                return None
        
        ClientFactory.register_client_type("failing", FailingClient)
        
        with patch('messaging_system.core.config_validator.ConfigValidator.validate_client_config'):
            with pytest.raises(ConfigurationError) as exc_info:
                ClientFactory.create_client("test_client", "failing", {})
        
        assert "Failed to create" in str(exc_info.value)
    
    def test_create_clients_from_config_success(self):
        """Test creating multiple clients from configuration."""
        ClientFactory.register_client_type("test", MockTestClient)
        
        config = {
            "client1": {
                "type": "test",
                "enabled": True,
                "config": {"param1": "value1"}
            },
            "client2": {
                "type": "test",
                "enabled": True,
                "config": {"param2": "value2"}
            }
        }
        
        with patch('messaging_system.core.config_validator.ConfigValidator.validate_client_config'):
            clients = ClientFactory.create_clients_from_config(config)
        
        assert len(clients) == 2
        assert "client1" in clients
        assert "client2" in clients
        assert isinstance(clients["client1"], MockTestClient)
        assert isinstance(clients["client2"], MockTestClient)
    
    def test_create_clients_from_config_disabled_client(self):
        """Test creating clients with disabled client."""
        ClientFactory.register_client_type("test", MockTestClient)
        
        config = {
            "client1": {
                "type": "test",
                "enabled": True,
                "config": {}
            },
            "client2": {
                "type": "test",
                "enabled": False,
                "config": {}
            }
        }
        
        with patch('messaging_system.core.config_validator.ConfigValidator.validate_client_config'):
            clients = ClientFactory.create_clients_from_config(config)
        
        assert len(clients) == 1
        assert "client1" in clients
        assert "client2" not in clients
    
    def test_create_clients_from_config_missing_type(self):
        """Test creating clients with missing type."""
        config = {
            "client1": {
                "enabled": True,
                "config": {}
            }
        }
        
        with pytest.raises(ConfigurationError) as exc_info:
            ClientFactory.create_clients_from_config(config)
        
        assert "missing 'type' field" in str(exc_info.value)
    
    def test_create_clients_from_config_creation_error(self):
        """Test creating clients with creation error."""
        ClientFactory.register_client_type("test", MockTestClient)
        
        config = {
            "client1": {
                "type": "unsupported",
                "enabled": True,
                "config": {}
            }
        }
        
        with pytest.raises(ConfigurationError) as exc_info:
            ClientFactory.create_clients_from_config(config)
        
        assert "Failed to create client" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
