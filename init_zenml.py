import os
from pathlib import Path
from zenml.config.global_config import GlobalConfiguration
from zenml.zen_stores import LocalZenStoreConfiguration

# Set up paths
ZENML_DIR = Path(".zen")
ZENML_DIR.mkdir(exist_ok=True, parents=True)

# Configure local store
local_store_path = ZENML_DIR / "local" / "store"
local_store_path.parent.mkdir(parents=True, exist_ok=True)

# Create a new GlobalConfiguration
config = GlobalConfiguration()

# Set up local store configuration
store_config = LocalZenStoreConfiguration(
    type="local",
    url=str(local_store_path.absolute())
)

# Configure the store
config.set_default_store(store_config)

# Set default project and stack
config.set_default_project("default")
config.activate_stack(config.active_stack_name)

print("✅ ZenML configuration initialized successfully!")
print(f"Store location: {local_store_path.absolute()}")
