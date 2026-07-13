RED='\e[31m'
GREEN='\e[32m'
BLUE='\e[34m'
BOLD='\e[1m'
YELLOW='\e[33m'
RESET='\e[0m' # Resets all text attributes

OK_CHECKED="${GREEN}${BOLD}[OK]${RESET}"
ERROR_CHECKED="${RED}${BOLD}[ERROR]${RESET}"

print_in_color() {
    local color_code="$1"
    local message="$2"
    echo -e "${color_code}${message}${RESET}"
}

print_ok() {
    local message="$1"
    print_in_color "${GREEN}${BOLD}" "${OK_CHECKED} ${message}"
}

print_warning() {
    local message="$1"
    print_in_color "${YELLOW}${BOLD}" "${WARNING_CHECKED} ${message}"
}

print_error() {
    local message="$1"
    print_in_color "${RED}${BOLD}" "${ERROR_CHECKED} ${message}"
}

print_exit() {
    local message="$1"
    print_error "$message"
    exit 1
}