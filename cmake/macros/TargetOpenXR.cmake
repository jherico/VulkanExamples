#  Created by Bradley Austin Davis on 2024/01/20
#  Copyright 2024
#
#  Distributed under the Apache License, Version 2.0.
#  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
#
macro(TARGET_OPENXR)
    find_package(OpenXR CONFIG REQUIRED)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenXR::headers OpenXR::openxr_loader)
endmacro()
