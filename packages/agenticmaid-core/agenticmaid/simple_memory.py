import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class SimpleMemoryProvider(ABC):
    """Abstract base class for simple memory providers."""
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a value for a given key."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any):
        """Sets a value for a given key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Deletes a key from memory."""
        pass

    @abstractmethod
    def append(self, key: str, value: Any):
        """Appends a value to a list associated with a key."""
        pass

class JsonProvider(SimpleMemoryProvider):
    """A simple key-value memory store backed by a local JSON file."""

    def __init__(self, config: Dict[str, Any]):
        self.file_path = config.get("file_path")
        if not self.file_path:
            raise ValueError("JsonProvider config must include 'file_path'.")

        self._lock = threading.Lock()
        self._data = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        with self._lock:
            if not os.path.exists(self.file_path):
                return {}
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content:
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, IOError):
                return {}

    def _save_memory(self):
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
            except IOError as e:
                print(f"Error saving memory file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value
        self._save_memory()

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._save_memory()
            return True
        return False

    def append(self, key: str, value: Any):
        current_value = self._data.get(key)
        if isinstance(current_value, list):
            current_value.append(value)
        else:
            self._data[key] = [value]
        self._save_memory()

class SimpleMemory:
    """
    Factory class for creating and managing a simple memory provider.
    """
    def __init__(self, config: Dict[str, Any]):
        provider_name = config.get("provider")
        if not provider_name:
            raise ValueError("SimpleMemory config must include a 'provider'.")

        provider_config = config.get("config", {})

        if provider_name == "json_file":
            self.provider = JsonProvider(provider_config)
        # Future providers can be added here
        # elif provider_name == "redis":
        #     self.provider = RedisProvider(provider_config)
        else:
            raise ValueError(f"Unsupported simple memory provider: {provider_name}")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a value for a given key."""
        return self.provider.get(key, default)

    def set(self, key: str, value: Any):
        """Sets a value for a given key."""
        self.provider.set(key, value)

    def delete(self, key: str) -> bool:
        """Deletes a key from memory."""
        return self.provider.delete(key)

    def append(self, key: str, value: Any):
        """Appends a value to a list associated with a key."""
        self.provider.append(key, value)