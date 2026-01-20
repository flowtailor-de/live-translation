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




# NOTE: The official Piper "macos_aarch64" release is broken - it actually contains x86_64 binaries
# So we always use the x86_64 version and run it via Rosetta 2 on Apple Silicon Macs
# The Python code in synthesizer.py handles running via "arch -x86_64" automatically

PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_x64.tar.gz"

# Detect CPU architecture for espeak-ng library path
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [ "$ARCH" = "arm64" ]; then
    echo "Apple Silicon detected - Piper will run via Rosetta 2"
    # We need to install x86_64 espeak-ng for Rosetta compatibility
    NEED_X86_ESPEAK=true
elif [ "$ARCH" = "x86_64" ]; then
    NEED_X86_ESPEAK=false
else
    echo -e "${RED}Error: Unsupported architecture: $ARCH${NC}"
    exit 1
fi

echo "Downloading Piper TTS binary (x86_64 for Rosetta compatibility)..."

# Check if existing piper binary has the wrong architecture
NEED_DOWNLOAD=false
if [ -f "bin/piper/piper" ]; then
    PIPER_ARCH=$(file bin/piper/piper | grep -o 'x86_64\|arm64' | head -1)
    if [ "$PIPER_ARCH" != "$ARCH" ]; then
        echo "Existing Piper binary is for $PIPER_ARCH, but this machine is $ARCH. Re-downloading..."
        rm -rf bin/piper
        NEED_DOWNLOAD=true
    else
        echo "Piper binary already exists and matches architecture."
    fi
else
    NEED_DOWNLOAD=true
fi

if [ "$NEED_DOWNLOAD" = true ]; then
    mkdir -p bin/piper_temp
    curl -L "$PIPER_URL" -o bin/piper_temp/piper.tar.gz
    tar -xzf bin/piper_temp/piper.tar.gz -C bin/
    rm -rf bin/piper_temp
    echo "Piper binary installed to bin/piper/"
fi

# Fix Piper's espeak-ng library dependency (required for macOS)
echo "Setting up espeak-ng library for Piper..."

# We always need x86_64 library since Piper binary is x86_64
# Check if library exists and has correct architecture (must be x86_64)
NEED_LIB=false
if [ -f "bin/piper/libespeak-ng.1.dylib" ]; then
    LIB_ARCH=$(file bin/piper/libespeak-ng.1.dylib | grep -o 'x86_64\|arm64' | head -1)
    if [ "$LIB_ARCH" != "x86_64" ]; then
        echo "Existing espeak-ng library is $LIB_ARCH, but Piper needs x86_64. Re-copying..."
        rm -f bin/piper/libespeak-ng.1.dylib
        NEED_LIB=true
    else
        echo "espeak-ng library already exists and is x86_64."
    fi
else
    NEED_LIB=true
fi

if [ "$NEED_LIB" = true ]; then
    if [ "$NEED_X86_ESPEAK" = true ]; then
        # On Apple Silicon, we need x86_64 espeak-ng library
        # This requires x86_64 Homebrew installation
        X86_BREW="/usr/local/bin/brew"
        
        if [ -f "$X86_BREW" ]; then
            echo "Using x86_64 Homebrew to install espeak-ng..."
            if ! arch -x86_64 $X86_BREW list espeak-ng &>/dev/null; then
                arch -x86_64 $X86_BREW install espeak-ng
            fi
            ESPEAK_LIB="/usr/local/lib/libespeak-ng.1.dylib"
        else
            echo -e "${BLUE}x86_64 Homebrew not found at $X86_BREW${NC}"
            echo "Installing x86_64 Homebrew for Rosetta compatibility..."
            arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            echo "Installing espeak-ng via x86_64 Homebrew..."
            arch -x86_64 /usr/local/bin/brew install espeak-ng
            ESPEAK_LIB="/usr/local/lib/libespeak-ng.1.dylib"
        fi
    else
        # On Intel Mac, use native Homebrew
        if ! brew list espeak-ng &>/dev/null; then
            echo "Installing espeak-ng via Homebrew..."
            brew install espeak-ng
        fi
        ESPEAK_LIB="/usr/local/lib/libespeak-ng.1.dylib"
    fi
    
    if [ -f "$ESPEAK_LIB" ]; then
        cp "$ESPEAK_LIB" bin/piper/
        echo "Copied espeak-ng library to bin/piper/"
    else
        echo -e "${RED}Warning: Could not find libespeak-ng.1.dylib at $ESPEAK_LIB${NC}"
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
