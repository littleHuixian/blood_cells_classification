cmake_minimum_required(VERSION 3.19)
project(five_classification_demo LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)

# ONNX Runtime 1.16.3 本机安装目录
set(ONNXRUNTIME_DIR
    "D:/onnxruntime-win-x64-1.16.3"
    CACHE PATH "ONNX Runtime installation root")

# OpenCV 4.5.3 本机安装目录（MinGW 64-bit 构建）
set(OpenCV_DIR
    "D:/OpenCV453/build/install/x64/mingw/lib"
    CACHE PATH "Path containing OpenCVConfig.cmake")
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs)

# ONNX Runtime 动态库：运行用 D 盘 DLL，链接用本工程内生成的 MinGW 导入库
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/include"
)

qt_standard_project_setup()

qt_add_executable(five_classification_demo
    WIN32 MACOSX_BUNDLE
    main.cpp
    bloodcell_classifier.cpp
    bloodcell_classifier.h
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

target_include_directories(five_classification_demo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(five_classification_demo
    PRIVATE
        Qt::Core
        Qt::Widgets
        onnxruntime
        ${OpenCV_LIBS}
)

# 将运行所需的 ONNX Runtime / OpenCV DLL 复制到可执行文件目录
if(WIN32)
    set(OPENCV_RUNTIME_DIR "D:/OpenCV453/build/install/x64/mingw/bin")
    add_custom_command(TARGET five_classification_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:five_classification_demo>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_RUNTIME_DIR}/libopencv_core453.dll"
            "$<TARGET_FILE_DIR:five_classification_demo>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_RUNTIME_DIR}/libopencv_imgproc453.dll"
            "$<TARGET_FILE_DIR:five_classification_demo>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_RUNTIME_DIR}/libopencv_imgcodecs453.dll"
            "$<TARGET_FILE_DIR:five_classification_demo>"
        VERBATIM
    )
endif()

install(TARGETS five_classification_demo
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET five_classification_demo
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
