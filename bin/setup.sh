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

# Check Python version - piper-tts requires Python 3.10-3.12
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
    echo -e "${BLUE}⚠️  Python $PYTHON_VERSION detected. piper-tts works best with Python 3.10-3.12${NC}"
    
    # Check if python3.12 or python3.11 is available
    if command -v python3.12 &> /dev/null; then
        echo -e "${GREEN}Found python3.12, using it instead...${NC}"
        PYTHON_CMD="python3.12"
    elif command -v python3.11 &> /dev/null; then
        echo -e "${GREEN}Found python3.11, using it instead...${NC}"
        PYTHON_CMD="python3.11"
    elif command -v python3.10 &> /dev/null; then
        echo -e "${GREEN}Found python3.10, using it instead...${NC}"
        PYTHON_CMD="python3.10"
    else
        echo -e "${BLUE}Installing Python 3.12 via Homebrew...${NC}"
        brew install python@3.12
        PYTHON_CMD="/opt/homebrew/opt/python@3.12/bin/python3.12"
    fi
else
    PYTHON_CMD="python3"
fi

# 2. Backend Setup
echo -e "\n${BLUE}2. Setting up Python Backend...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
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
