#!/bin/bash

# Create necessary directories
mkdir -p .zen/local/store

# Initialize ZenML if not already done
if [ ! -f ".zen/config.yaml" ]; then
    echo "Initializing ZenML..."
    python -c "
from zenml.client import Client
from zenml.zen_stores import LocalZenStoreConfiguration
import os

# Configure local store
store_config = LocalZenStoreConfiguration(
    type='local',
    path=os.path.abspath('.zen/local/store')
)

# Initialize ZenML
Client.initialize(store_config)
print('✅ ZenML initialized successfully')
"
fi

echo "✅ Setup complete!"
