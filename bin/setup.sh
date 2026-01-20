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
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt




echo "Downloading Piper TTS binary (Apple Silicon)..."
if [ ! -f "bin/piper/piper" ]; then
    mkdir -p bin/piper_temp
    curl -L "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_aarch64.tar.gz" -o bin/piper_temp/piper.tar.gz
    tar -xzf bin/piper_temp/piper.tar.gz -C bin/
    rm -rf bin/piper_temp
    echo "Piper binary installed to bin/piper/"
else
    echo "Piper binary already exists."
fi

# Fix Piper's espeak-ng library dependency (required for macOS)
echo "Setting up espeak-ng library for Piper..."
if [ ! -f "bin/piper/libespeak-ng.1.dylib" ]; then
    # Install espeak-ng via Homebrew if not present
    if ! brew list espeak-ng &>/dev/null; then
        echo "Installing espeak-ng via Homebrew..."
        brew install espeak-ng
    fi
    
    # Find and copy the espeak-ng library
    ESPEAK_LIB=""
    if [ -f "/opt/homebrew/lib/libespeak-ng.1.dylib" ]; then
        ESPEAK_LIB="/opt/homebrew/lib/libespeak-ng.1.dylib"
    elif [ -f "/usr/local/lib/libespeak-ng.1.dylib" ]; then
        ESPEAK_LIB="/usr/local/lib/libespeak-ng.1.dylib"
    fi
    
    if [ -n "$ESPEAK_LIB" ]; then
        cp "$ESPEAK_LIB" bin/piper/
        echo "Copied espeak-ng library to bin/piper/"
    else
        echo -e "${RED}Warning: Could not find libespeak-ng.1.dylib${NC}"
    fi
fi

# Fix the library path in piper binary to use @executable_path
if [ -f "bin/piper/piper" ] && [ -f "bin/piper/libespeak-ng.1.dylib" ]; then
    echo "Fixing Piper library paths..."
    install_name_tool -change @rpath/libespeak-ng.1.dylib @executable_path/libespeak-ng.1.dylib bin/piper/piper 2>/dev/null || true
    echo "✅ Piper library paths fixed"
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
