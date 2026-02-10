#!/bin/bash
# Colors
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting Live Translation Launcher...${NC}"

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate venv
source venv/bin/activate

# Run launcher
python launcher.py
