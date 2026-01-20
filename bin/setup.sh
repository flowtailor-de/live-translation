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

echo "Installing dependencies (including piper-tts with native Apple Silicon support)..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Detect CPU architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [ "$ARCH" = "arm64" ]; then
    echo "✅ Apple Silicon detected - piper-tts will run natively (no Rosetta needed)"
elif [ "$ARCH" = "x86_64" ]; then
    echo "✅ Intel Mac detected"
else
    echo -e "${RED}Warning: Unknown architecture: $ARCH${NC}"
fi

# Clean up old piper binary if it exists (no longer needed with piper-tts Python package)
if [ -d "bin/piper" ]; then
    echo "Removing old Piper binary (now using piper-tts Python package)..."
    rm -rf bin/piper
fi

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
