import os
from pathlib import Path
from zenml.client import Client
from zenml.enums import StoreType, StackComponentType
from zenml.zen_stores import LocalZenStoreConfiguration
from zenml.config.global_config import GlobalConfiguration

def initialize_zenml():
    """Initialize ZenML with local store configuration."""
    try:
        # Set up paths
        ZENML_DIR = Path(".zen")
        ZENML_DIR.mkdir(exist_ok=True, parents=True)
        
        # Configure local store path
        local_store_path = ZENML_DIR / "local" / "store"
        local_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize GlobalConfiguration
        config = GlobalConfiguration()
        
        # Only initialize if not already done
        if not (ZENML_DIR / "config.yaml").exists():
            # Set up local store configuration
            store_config = LocalZenStoreConfiguration(
                type=StoreType.LOCAL,
                url=f"sqlite:///{local_store_path.absolute()}/zenml.db"
            )
            
            # Configure the store
            config.set_default_store(store_config)
            
            # Initialize the client
            client = Client()
            
            # Set default project
            try:
                client.get_project("default")
            except KeyError:
                client.create_project("default")
            
            print("✅ ZenML configuration initialized!")
            print(f"Store location: {local_store_path.absolute()}")
        else:
            print("ℹ️  ZenML is already initialized.")
            
        return True
        
    except Exception as e:
        print(f"❌ Error initializing ZenML: {str(e)}")
        return False

if __name__ == "__main__":
    initialize_zenml()
