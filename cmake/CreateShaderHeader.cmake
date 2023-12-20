macro(ENDIAN_SWAP)
    set(endian_swap_temp "")
    while (NOT ("${dword}" STREQUAL ""))
        # Take the first 8 characters from the string
        string(SUBSTRING ${dword} 0 2 endian_swap_byte)
        # Remove the first 8 characters from the string
        string(SUBSTRING ${dword} 2 -1 dword)
        string(PREPEND endian_swap_temp "${endian_swap_byte}")
    endwhile()
    set(dword "${endian_swap_temp}")
endmacro()

file(READ ${SHADER_SPIRV} input_hex HEX)
string(LENGTH ${input_hex} SPIRV_NIBBLE_COUNT)
math(EXPR SHADER_TOKENS "${SPIRV_NIBBLE_COUNT} / 8")

set(SHADER_CODE "")
set(COUNTER 0)

# Iterate through each of the 32 bit tokens from the source file
while (NOT ("${input_hex}" STREQUAL ""))
    if (COUNTER GREATER 8)
        # Write a newline so that all of the array initializer
        # gets spread across multiple lines.
        string(APPEND SHADER_CODE "\n                    ")
        set(COUNTER 0)
    endif()

    # Take the first 8 characters from the string
    string(SUBSTRING ${input_hex} 0 8 dword)
    # Remove the first 8 characters from the string
    string(SUBSTRING ${input_hex} 8 -1 input_hex)
    ENDIAN_SWAP()
    # Write the hex string to the line with an 0x prefix
    string(APPEND SHADER_CODE "0x${dword},")

    # Increment the element counter before the newline.
    math(EXPR COUNTER "${COUNTER}+1")
endwhile()


get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
get_filename_component(SHADER_TYPE ${SHADER_FILE} EXT)
string(SUBSTRING ${SHADER_TYPE} 1 -1 SHADER_TYPE)


# namespace @EXAMPLE_NAME@ {
#     namespace @SHADER_NAME@ {
#         constexpr std::array<uint32_t, @SHADER_TOKENS@> @SHADER_TYPE@ {{
#             @SHADER_CODE@
#         }};
#     }
# }


configure_file("${CONFIG_FILE}" ${SHADER_HEADER_NAME} @ONLY)


