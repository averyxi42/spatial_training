# Function to get yes/no input from user with default option
get_yn_input() {
    local prompt="$1"
    local default_option="${2:-Y}"
    local -a yes_action=("${!3}")   # expand array passed by name
    local -a no_action=("${!4}")

    while :; do
        read -rp "$prompt" answer
        answer="${answer:-$default_option}"

        case "$answer" in
            [Yy]* )
                "${yes_action[@]}"
                return 0
                ;;
            [Nn]* )
                "${no_action[@]}"
                return 1
                ;;
            * )
                echo "Invalid response (please enter y or n)"
                ;;
        esac
    done
}

get_str_input() {
    local prompt="$1"
    local var_name="$2"

    while :; do
        # Quote the prompt so a full string with spaces is supported
        read -rp "$prompt" answer

        if [ -n "$answer" ]; then
            eval "$var_name='$answer'"
            return 0
        else
            echo "Input cannot be empty. Please try again."
        fi
    done
}

choose_option() {
    local prompt="$1"
    shift
    local options=("$@") index=0 key tty tput_term
    [ ${#options[@]} -eq 0 ] && return 1

    if ! tty=$(tty); then
        echo "No TTY available for interactive selection" >&2
        return 1
    fi

    tput_term="${TERM:-xterm}"

    # Use a dedicated descriptor tied to the terminal so output still appears even if stdout is piped.
    exec 3<> "$tty" || return 1

    tput -T "$tput_term" civis <&3 >&3
    trap 'tput -T "$tput_term" cnorm <&3 >&3; exec 3>&-; echo' EXIT

    # Render prompt on its own line
    printf '%s\n' "$prompt" >&3

    local bold_cyan='\033[1;36m' reset='\033[0m'

    while true; do
        # Clear the options line then render options
        printf '\r\033[K' >&3
        for i in "${!options[@]}"; do
            if [[ $i -eq $index ]]; then
                printf ' %b[%s]%b ' "$bold_cyan" "${options[$i]}" "$reset" >&3
            else
                printf '  %s  ' "${options[$i]}" >&3
            fi
        done

        IFS= read -rsn1 -u 3 key || key=""
        case "$key" in
            "")  # Enter
                printf '\n' >&3
                break
                ;;
            $'\x1b') # Arrow prefix
                IFS= read -rsn2 -u 3 key
                case "$key" in
                    "[D") ((index=(index-1+${#options[@]})%${#options[@]})) ;; # Left
                    "[C") ((index=(index+1)%${#options[@]})) ;;               # Right
                esac
                ;;
        esac
    done

    tput -T "$tput_term" cnorm <&3 >&3
    trap - EXIT
    exec 3>&-
    printf '%s\n' "${options[$index]}"
}
