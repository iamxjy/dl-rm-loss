#!/bin/bash
# Helper script to set up personal wandb configuration
# This creates a config file with your API key that can be used in jobs

WANDB_PERSONAL_DIR="/proj/inf-scaling/iris/iris_wandb_config"
WANDB_API_KEY_FILE="$WANDB_PERSONAL_DIR/api_key.txt"

echo "=== Setting up personal wandb configuration ==="
echo "Config directory: $WANDB_PERSONAL_DIR"
echo ""

# Create directories
mkdir -p "$WANDB_PERSONAL_DIR"

# Clear any existing wandb settings from wrong user
if [ -f "$WANDB_PERSONAL_DIR/settings" ]; then
    echo "Found existing wandb settings. Removing to start fresh..."
    rm -f "$WANDB_PERSONAL_DIR/settings"
fi

# Check if API key file already exists
if [ -f "$WANDB_API_KEY_FILE" ]; then
    echo "API key file already exists at: $WANDB_API_KEY_FILE"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing API key."
        exit 0
    fi
fi

echo ""
echo "Please enter your wandb API key."
echo "Get it from: https://wandb.ai/authorize"
echo ""
read -sp "API key: " api_key
echo ""

if [ -z "$api_key" ]; then
    echo "✗ No API key provided. Exiting."
    exit 1
fi

# Save API key to file (with restricted permissions)
echo "$api_key" > "$WANDB_API_KEY_FILE"
chmod 600 "$WANDB_API_KEY_FILE"

echo ""
echo "✓ API key saved to: $WANDB_API_KEY_FILE"
echo ""
echo "The run_grpo_multi_node.sh script will automatically load this API key."
echo ""
echo "✓ This file is gitignored and will NOT be committed to git"
echo "  (safe to push your code without exposing your API key)"
echo ""
echo "Your wandb runs will now log to your personal account!"
echo ""
echo "To verify, check that the API key file exists:"
echo "  ls -l $WANDB_API_KEY_FILE"

