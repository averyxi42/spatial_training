check_pkgs() {
    for pkg in "$@"; do
        if dpkg -s "$pkg" >/dev/null 2>&1; then
            echo -e "[${GREEN}${BOLD}OK${RESET}] $pkg is installed"
        else
            echo "[MISSING] $pkg is NOT installed"
            return 1   # exit function immediately
        fi
    done
    return 0   # all good
}

# Example usage:
# pkgs_to_check=(curl git vim)
# if check_pkgs "${pkgs_to_check[@]}"; then
#     echo "All packages are installed."
# else
#     echo "Some packages are missing."
# fi