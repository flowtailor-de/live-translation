#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting Live Translation System Setup...${NC}"

# 1. Check Prerequisites & Install if missing
echo -e "\n${BLUE}1. Checking prerequisites...${NC}"

# Helper function to install via Homebrew
install_if_missing() {
    CMD=$1
    PKG=$2
    if ! command -v $CMD &> /dev/null; then
        echo -e "${BLUE}$PKG is missing. Checking for Homebrew...${NC}"
        if command -v brew &> /dev/null; then
            echo -e "${GREEN}Homebrew found. Installing $PKG...${NC}"
            brew install $PKG
        else
            echo -e "${RED}Error: $PKG is not installed and Homebrew was not found.${NC}"
            echo "Please install Homebrew (https://brew.sh) or install $PKG manually."
            exit 1
        fi
    else
        echo "✅ $PKG is installed"
    fi
}

install_if_missing "python3" "python"
install_if_missing "npm" "node"

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
