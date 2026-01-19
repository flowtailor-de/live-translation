#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Cleanup function to kill child processes
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo -e "${BLUE}🚀 Starting Live Translation System...${NC}"

# 1. Start Backend
echo -e "${GREEN}Starting Python Backend...${NC}"
source venv/bin/activate
python -m src.main &
BACKEND_PID=$!

# 2. Start Frontend
echo -e "${GREEN}Starting React Frontend...${NC}"
cd ui
npm run dev -- --host &
FRONTEND_PID=$!
cd ..

echo -e "${BLUE}System is running! Access the UI at http://localhost:5173${NC}"
echo -e "Press Ctrl+C to stop.\n"

# Wait for processes
wait
