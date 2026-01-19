#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting Live Translation System Setup...${NC}"

# 1. Check Prerequisites
echo -e "\n${BLUE}1. Checking prerequisites...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: Node.js/npm is not installed."
    exit 1
fi

# 2. Backend Setup
echo -e "\n${BLUE}2. Setting up Python Backend...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading AI models (this may take a while)..."
python -m src.download_models

# 3. Frontend Setup
echo -e "\n${BLUE}3. Setting up React Frontend...${NC}"
cd ui
echo "Installing Node modules..."
npm install
cd ..

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "You can now run the system using: ${GREEN}./bin/start.sh${NC}"
